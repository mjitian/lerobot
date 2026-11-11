#!/usr/bin/env python

"""Convert the local Tiangong2Pro pose dataset into a LeRobot v3.0 dataset.

The source dataset uses one parquet file and three JPEG directories per episode.
This converter:

- combines the 34 non-head state components into ``observation.state``;
- combines the left/right 10-D actions into ``action``;
- encodes the three JPEG streams as LeRobot v3 MP4 video features;
- writes v3 tasks, stats, episode metadata, and the dataset info file.

Example:

    PYTHONPATH=src python3 tools/convert_pose_dataset_to_v30.py \
        --source dataset_2026-10-08_13-55-37_pose \
        --output dataset_2026-10-08_13-55-37_pose_v30
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from lerobot.datasets import CODEBASE_VERSION
from lerobot.datasets.compute_stats import (
    DEFAULT_QUANTILES,
    get_feature_stats,
    sample_images,
)
from lerobot.datasets.feature_utils import (
    create_empty_dataset_info,
    get_hf_features_from_features,
)
from lerobot.datasets.io_utils import write_episodes, write_info, write_stats, write_tasks
from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH
from lerobot.datasets.video_utils import get_video_duration_in_s, get_video_info


TASK = (
    "Sort all test tubes by color, move all red test tubes from the dark-blue rack "
    "to the orange rack first, then move all purple test tubes to the light-blue rack. "
    "Continue until all test tubes are sorted."
)

SOURCE_CAMERAS = {
    "observation.images.head": "observation_images_head",
    "observation.images.left_wrist": "observation_images_left_wrist",
    "observation.images.right_wrist": "observation_images_right_wrist",
}

STATE_COLUMNS = [
    "observation_states_joint_angle_left",
    "observation_states_joint_angle_right",
    "observation_states_ee_pose_left",
    "observation_states_ee_pose_right",
    "observation_states_gripper_left",
    "observation_states_gripper_right",
]

STATE_NAMES = [
    *[f"observation_states_joint_angle_left.j{i}" for i in range(7)],
    *[f"observation_states_joint_angle_right.j{i}" for i in range(7)],
    *[f"observation_states_ee_pose_left.e{i}" for i in range(9)],
    *[f"observation_states_ee_pose_right.e{i}" for i in range(9)],
    "observation_states_gripper_left",
    "observation_states_gripper_right",
]

ACTION_COLUMNS = ["action_left", "action_right"]
ACTION_NAMES = [
    *[f"action_left.e{i}" for i in range(10)],
    *[f"action_right.e{i}" for i in range(10)],
]

VIDEO_ENCODER_INFO = {
    "video.g": 15,
    "video.crf": 23,
    "video.preset": "veryfast",
    "video.fast_decode": 0,
    "video.video_backend": "pyav",
    "video.extra_options": {"profile:v": "high", "bf": 2},
}


@dataclass(frozen=True)
class SourceEpisode:
    episode_index: int
    parquet_path: Path
    length: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset_2026-10-08_13-55-37_pose"),
        help="Source pose dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_2026-10-08_13-55-37_pose_v30"),
        help="Destination LeRobot v3 dataset directory.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid partial outputs from an interrupted conversion.",
    )
    return parser.parse_args()


def _discover_episodes(source: Path) -> list[SourceEpisode]:
    paths = sorted((source / "data" / "chunk-000").glob("episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under {source}")

    episodes = []
    for expected_index, path in enumerate(paths):
        episode_index = int(path.stem.rsplit("_", 1)[-1])
        if episode_index != expected_index:
            raise ValueError(
                f"Episode indices must be contiguous: expected {expected_index}, found {episode_index} in {path}"
            )
        episodes.append(SourceEpisode(episode_index, path, pq.ParquetFile(path).metadata.num_rows))
    return episodes


def _validate_source(source: Path, episodes: list[SourceEpisode], fps: int) -> None:
    required_columns = {
        *STATE_COLUMNS,
        *ACTION_COLUMNS,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    expected_global_index = 0

    for episode in tqdm(episodes, desc="Validate source", unit="episode"):
        table = pq.read_table(episode.parquet_path)
        missing = sorted(required_columns - set(table.column_names))
        if missing:
            raise ValueError(f"{episode.parquet_path} is missing required columns: {missing}")
        if table.num_rows != episode.length:
            raise ValueError(f"Unexpected row count in {episode.parquet_path}")

        frame_index = table["frame_index"].to_numpy(zero_copy_only=False)
        if not np.array_equal(frame_index, np.arange(episode.length)):
            raise ValueError(f"Frame indices are not contiguous in {episode.parquet_path}")

        episode_index = table["episode_index"].to_numpy(zero_copy_only=False)
        if not np.all(episode_index == episode.episode_index):
            raise ValueError(f"Mixed episode indices in {episode.parquet_path}")

        global_index = table["index"].to_numpy(zero_copy_only=False)
        expected = np.arange(expected_global_index, expected_global_index + episode.length)
        if not np.array_equal(global_index, expected):
            raise ValueError(f"Global frame indices are not contiguous in {episode.parquet_path}")
        expected_global_index += episode.length

        timestamp = table["timestamp"].to_numpy(zero_copy_only=False)
        expected_timestamp = np.arange(episode.length, dtype=np.float32) / fps
        if not np.allclose(timestamp, expected_timestamp, atol=1e-5):
            raise ValueError(f"Unexpected timestamps in {episode.parquet_path}")

        for image_key, source_camera in SOURCE_CAMERAS.items():
            image_dir = source / "images" / source_camera / f"episode_{episode.episode_index:06d}"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Missing source image directory: {image_dir}")
            image_count = sum(1 for _ in image_dir.glob("frame_*.jpg"))
            if image_count != episode.length:
                raise ValueError(
                    f"{image_dir} has {image_count} JPEG frames, expected {episode.length} "
                    f"({image_key})"
                )


def _source_arrays(table: pa.Table, columns: list[str]) -> np.ndarray:
    parts = [np.asarray(table[column].to_pylist(), dtype=np.float32) for column in columns]
    return np.concatenate(parts, axis=1)


def _build_data(
    output: Path,
    episodes: list[SourceEpisode],
    video_features: dict[str, dict],
    fps: int,
) -> tuple[dict[str, dict], np.ndarray, np.ndarray]:
    all_states = []
    all_actions = []
    all_episode_indices = []
    all_frame_indices = []

    for episode in tqdm(episodes, desc="Convert state/action", unit="episode"):
        columns = [
            *STATE_COLUMNS,
            *ACTION_COLUMNS,
            "frame_index",
        ]
        table = pq.read_table(episode.parquet_path, columns=columns)
        states = _source_arrays(table, STATE_COLUMNS)
        actions = _source_arrays(table, ACTION_COLUMNS)

        if states.shape != (episode.length, 34):
            raise ValueError(f"Unexpected state shape for episode {episode.episode_index}: {states.shape}")
        if actions.shape != (episode.length, 20):
            raise ValueError(f"Unexpected action shape for episode {episode.episode_index}: {actions.shape}")

        all_states.append(states)
        all_actions.append(actions)
        all_episode_indices.append(
            np.full(episode.length, episode.episode_index, dtype=np.int64)
        )
        all_frame_indices.append(np.arange(episode.length, dtype=np.int64))

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    episode_indices = np.concatenate(all_episode_indices)
    frame_indices = np.concatenate(all_frame_indices)
    total_frames = len(states)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (34,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (20,),
            "names": ACTION_NAMES,
        },
        **video_features,
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }

    hf_features = get_hf_features_from_features(features)
    frame_data = {
        "observation.state": states,
        "action": actions,
        "timestamp": np.empty(total_frames, dtype=np.float32),
        "frame_index": frame_indices,
        "episode_index": episode_indices,
        "index": np.arange(total_frames, dtype=np.int64),
        "task_index": np.zeros(total_frames, dtype=np.int64),
    }

    # Timestamps restart at zero for every episode.
    timestamps = np.empty(total_frames, dtype=np.float32)
    offset = 0
    for episode in episodes:
        timestamps[offset : offset + episode.length] = (
            np.arange(episode.length, dtype=np.float32) / fps
        )
        offset += episode.length
    frame_data["timestamp"] = timestamps

    hf_dataset = datasets.Dataset.from_dict(frame_data, features=hf_features)
    data_path = output / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        hf_dataset.with_format("arrow")[:],
        data_path,
        compression="snappy",
        use_dictionary=True,
    )

    return features, states, actions


def _video_target(
    output: Path,
    video_key: str,
    episode_index: int,
) -> Path:
    relative_path = DEFAULT_VIDEO_PATH.format(
        video_key=video_key,
        chunk_index=0,
        file_index=episode_index,
    )
    return output / relative_path


def _count_video_frames(video_path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if not value or value == "N/A":
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
    return int(value)


def _encode_video(
    source: Path,
    output: Path,
    episode: SourceEpisode,
    video_key: str,
    source_camera: str,
    fps: int,
    crf: int,
    preset: str,
) -> tuple[str, int, int]:
    target = _video_target(output, video_key, episode.episode_index)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        try:
            if _count_video_frames(target) == episode.length:
                return video_key, episode.episode_index, target.stat().st_size
        except (ValueError, subprocess.CalledProcessError):
            pass

    image_dir = source / "images" / source_camera / f"episode_{episode.episode_index:06d}"
    partial = target.with_name(f"{target.stem}.partial.mp4")
    if partial.exists():
        partial.unlink()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(image_dir / "frame_%06d.jpg"),
        "-frames:v",
        str(episode.length),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps),
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        encoded_frames = _count_video_frames(partial)
        if encoded_frames != episode.length:
            raise ValueError(
                f"Encoded {encoded_frames} frames for {video_key} episode "
                f"{episode.episode_index}, expected {episode.length}"
            )
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    return video_key, episode.episode_index, target.stat().st_size


def _encode_videos(
    source: Path,
    output: Path,
    episodes: list[SourceEpisode],
    fps: int,
    workers: int,
    crf: int,
    preset: str,
) -> None:
    jobs = [
        (episode, video_key, source_camera)
        for episode in episodes
        for video_key, source_camera in SOURCE_CAMERAS.items()
    ]
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _encode_video,
                source,
                output,
                episode,
                video_key,
                source_camera,
                fps,
                crf,
                preset,
            ): (video_key, episode.episode_index)
            for episode, video_key, source_camera in jobs
        }
        with tqdm(total=len(futures), desc="Encode videos", unit="video") as progress:
            for future in as_completed(futures):
                video_key, episode_index, size = future.result()
                total_bytes += size
                progress.update(1)
                if progress.n % 25 == 0:
                    progress.set_postfix_str(
                        f"{total_bytes / (1024**3):.2f} GiB encoded"
                    )

    print(f"Video output size: {total_bytes / (1024**3):.2f} GiB")


def _feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([len(values)], dtype=np.int64),
    }
    for quantile in DEFAULT_QUANTILES:
        stats[f"q{int(quantile * 100):02d}"] = np.quantile(values, quantile, axis=0)
    return stats


def _video_feature_stats(
    source: Path,
    source_camera: str,
    episodes: list[SourceEpisode],
    num_samples: int = 300,
) -> dict[str, np.ndarray]:
    """Sample RGB frames across the dataset and compute per-channel statistics."""
    sample_indices = np.linspace(0, len(episodes) - 1, num_samples).round().astype(int)
    frame_positions = (0, 1, 2)
    image_paths = []

    for sample_idx, episode_idx in enumerate(sample_indices):
        episode = episodes[int(episode_idx)]
        frame_index = {
            0: 0,
            1: episode.length // 2,
            2: episode.length - 1,
        }[frame_positions[sample_idx % len(frame_positions)]]
        image_paths.append(
            source
            / "images"
            / source_camera
            / f"episode_{episode.episode_index:06d}"
            / f"frame_{frame_index:06d}.jpg"
        )

    images = sample_images([str(path) for path in image_paths])
    stats = get_feature_stats(
        images,
        axis=(0, 2, 3),
        keepdims=True,
        quantile_list=DEFAULT_QUANTILES,
    )
    return {
        key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
        for key, value in stats.items()
    }


def _build_video_features(first_video_by_key: dict[str, Path]) -> dict[str, dict]:
    features = {}
    for video_key in SOURCE_CAMERAS:
        video_info = get_video_info(first_video_by_key[video_key])
        video_info.update(VIDEO_ENCODER_INFO)
        features[video_key] = {
            "dtype": "video",
            "shape": (
                video_info["video.height"],
                video_info["video.width"],
                video_info["video.channels"],
            ),
            "names": ["height", "width", "channels"],
            "info": video_info,
        }
    return features


def _write_episode_metadata(
    output: Path,
    episodes: list[SourceEpisode],
) -> None:
    rows = {
        "episode_index": [],
        "tasks": [],
        "length": [],
        "data/chunk_index": [],
        "data/file_index": [],
        "dataset_from_index": [],
        "dataset_to_index": [],
        "meta/episodes/chunk_index": [],
        "meta/episodes/file_index": [],
    }
    for video_key in SOURCE_CAMERAS:
        rows[f"videos/{video_key}/chunk_index"] = []
        rows[f"videos/{video_key}/file_index"] = []
        rows[f"videos/{video_key}/from_timestamp"] = []
        rows[f"videos/{video_key}/to_timestamp"] = []

    frame_offset = 0
    for episode in episodes:
        rows["episode_index"].append(episode.episode_index)
        rows["tasks"].append([TASK])
        rows["length"].append(episode.length)
        rows["data/chunk_index"].append(0)
        rows["data/file_index"].append(0)
        rows["dataset_from_index"].append(frame_offset)
        rows["dataset_to_index"].append(frame_offset + episode.length)
        rows["meta/episodes/chunk_index"].append(0)
        rows["meta/episodes/file_index"].append(0)

        video_path = _video_target(output, next(iter(SOURCE_CAMERAS)), episode.episode_index)
        duration = get_video_duration_in_s(video_path)
        for video_key in SOURCE_CAMERAS:
            rows[f"videos/{video_key}/chunk_index"].append(0)
            rows[f"videos/{video_key}/file_index"].append(episode.episode_index)
            rows[f"videos/{video_key}/from_timestamp"].append(0.0)
            rows[f"videos/{video_key}/to_timestamp"].append(duration)
        frame_offset += episode.length

    features = datasets.Features(
        {
            "episode_index": datasets.Value("int64"),
            "tasks": datasets.Sequence(datasets.Value("string")),
            "length": datasets.Value("int64"),
            "data/chunk_index": datasets.Value("int64"),
            "data/file_index": datasets.Value("int64"),
            "dataset_from_index": datasets.Value("int64"),
            "dataset_to_index": datasets.Value("int64"),
            "meta/episodes/chunk_index": datasets.Value("int64"),
            "meta/episodes/file_index": datasets.Value("int64"),
            **{
                key: datasets.Value("int64")
                for key in rows
                if key.startswith("videos/") and key.endswith(("chunk_index", "file_index"))
            },
            **{
                key: datasets.Value("float64")
                for key in rows
                if key.startswith("videos/") and key.endswith(("from_timestamp", "to_timestamp"))
            },
        }
    )
    episode_dataset = datasets.Dataset.from_dict(rows, features=features)
    write_episodes(episode_dataset, output)


def _write_metadata(
    source: Path,
    output: Path,
    episodes: list[SourceEpisode],
    features: dict[str, dict],
    states: np.ndarray,
    actions: np.ndarray,
    fps: int,
) -> None:
    tasks = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([TASK], name="task"),
    )
    write_tasks(tasks, output)

    stats = {
        "observation.state": _feature_stats(states),
        "action": _feature_stats(actions),
        **{
            video_key: _video_feature_stats(source, source_camera, episodes)
            for video_key, source_camera in SOURCE_CAMERAS.items()
        },
    }
    write_stats(stats, output)

    info = create_empty_dataset_info(
        codebase_version=CODEBASE_VERSION,
        fps=fps,
        features=features,
        use_videos=True,
        robot_type="tiangong2pro",
        chunks_size=1000,
    )
    info.total_episodes = len(episodes)
    info.total_frames = sum(episode.length for episode in episodes)
    info.total_tasks = 1
    info.splits = {"train": f"0:{len(episodes)}"}
    write_info(info, output)
    _write_episode_metadata(output, episodes)


def _validate_output(
    output: Path,
    episodes: list[SourceEpisode],
) -> None:
    expected_files = [
        output / "meta" / "info.json",
        output / "meta" / "stats.json",
        output / "meta" / "tasks.parquet",
        output / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        output / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0),
    ]
    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing v3 output files: {missing}")

    info = json.loads((output / "meta" / "info.json").read_text())
    if info["codebase_version"] != CODEBASE_VERSION:
        raise ValueError(f"Unexpected v3 codebase version: {info['codebase_version']}")
    if info["features"]["observation.state"]["shape"] != [34]:
        raise ValueError("observation.state is not 34-dimensional")
    if "observation_states_joint_angle_head" in info["features"]:
        raise ValueError("Head joint state must not be included in observation.state")
    if np.any(np.asarray(info["features"]["observation.state"]["shape"]) != 34):
        raise ValueError("Invalid observation.state shape")

    episode_table = pq.read_table(expected_files[3])
    if episode_table.num_rows != len(episodes):
        raise ValueError("Episode metadata row count does not match source")
    if not np.array_equal(
        episode_table["dataset_to_index"].to_numpy()[-1:],
        np.array([sum(episode.length for episode in episodes)]),
    ):
        raise ValueError("Final episode dataset_to_index is incorrect")

    for episode in tqdm(episodes, desc="Validate videos", unit="episode"):
        for video_key in SOURCE_CAMERAS:
            video_path = _video_target(output, video_key, episode.episode_index)
            if not video_path.is_file():
                raise FileNotFoundError(f"Missing output video: {video_path}")
            if _count_video_frames(video_path) != episode.length:
                raise ValueError(
                    f"Frame count mismatch for {video_path}: expected {episode.length}"
                )


def main() -> None:
    args = _parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {output}. Use --resume to continue an interrupted conversion."
        )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    episodes = _discover_episodes(source)
    _validate_source(source, episodes, args.fps)
    print(
        f"Source validated: {len(episodes)} episodes, "
        f"{sum(episode.length for episode in episodes)} frames"
    )

    output.mkdir(parents=True, exist_ok=True)

    # Use the first video in each camera to derive dimensions and v3 video info.
    preview_paths = {}
    for video_key, source_camera in SOURCE_CAMERAS.items():
        first_image = (
            source
            / "images"
            / source_camera
            / "episode_000000"
            / "frame_000000.jpg"
        )
        with Image.open(first_image) as image:
            width, height = image.size
        preview_paths[video_key] = (width, height)
        if (width, height) != (640, 480):
            raise ValueError(
                f"Expected 640x480 source frames, found {width}x{height} for {video_key}"
            )

    # Encode all videos first so the feature metadata can be derived from an
    # actual output file.
    _encode_videos(
        source=source,
        output=output,
        episodes=episodes,
        fps=args.fps,
        workers=args.workers,
        crf=args.crf,
        preset=args.preset,
    )

    first_video_by_key = {
        video_key: _video_target(output, video_key, episodes[0].episode_index)
        for video_key in SOURCE_CAMERAS
    }
    video_features = _build_video_features(first_video_by_key)

    # Build data after video encoding so an interrupted run resumes without
    # rewriting the large data table unnecessarily.
    features, states, actions = _build_data(
        output=output,
        episodes=episodes,
        video_features=video_features,
        fps=args.fps,
    )
    _write_metadata(
        source=source,
        output=output,
        episodes=episodes,
        features=features,
        states=states,
        actions=actions,
        fps=args.fps,
    )
    _validate_output(output, episodes)
    print(f"Conversion complete: {output}")


if __name__ == "__main__":
    main()
