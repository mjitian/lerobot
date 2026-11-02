#!/usr/bin/env bash
python -m lerobot.async_inference.robot_client \
    --robot.type=tiangong2pro \
    --task="'Sort all test tubes by color: move all red test tubes from the dark-blue rack to the orange rack first, then move all purple test tubes to the light-blue rack. Continue until all test tubes are sorted.'" \
    --server_address=192.168.41.100:8080 \
    --policy_type=pi05 \
    --pretrained_name_or_path=/home/nibs/lerobot_ws/lerobot/checkpoints/179999_lerobot \
    --policy_device=cuda \
    --actions_per_chunk=16 \
    --chunk_size_threshold=0.6 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True \
    --fps 15
