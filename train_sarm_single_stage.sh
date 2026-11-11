rm -rf outputs/train/sarm_single

# lerobot-train \
#   --dataset.repo_id=/home/nibs/lerobot_ws/lerobot/dataset_2026-10-08_13-55-37_pose_v30 \
#   --reward_model.type=sarm \
#   --reward_model.annotation_mode=single_stage \
#   --reward_model.image_key=observation.images.head \
#   --output_dir=outputs/train/sarm_single \
#   --batch_size=4 \
#   --steps=5000 \
#   --wandb.enable=false \
  # --wandb.enable=true \
  # --wandb.project=sarm \
  # --wandb.entity=mjitian-mjitain
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=8

lerobot-train \
  --dataset.repo_id=/home/nibs/lerobot_ws/lerobot/dataset_2026-10-08_13-55-37_pose_v30 \
  --reward_model.type=sarm \
  --reward_model.annotation_mode=single_stage \
  --reward_model.image_key=observation.images.head \
  --output_dir=outputs/train/sarm_single \
  --batch_size=2 \
  --num_workers=1 \
  --prefetch_factor=1 \
  --steps=5000 \
  --wandb.enable=false
