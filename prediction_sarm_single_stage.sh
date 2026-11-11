python -m lerobot.rewards.sarm.compute_rabc_weights \
  --dataset-repo-id your-username/your-dataset \
  --reward-model-path your-username/sarm-model \
  --visualize-only \
  --num-visualizations 5 \
  --head-mode sparse \
  --output-dir ./sarm_viz