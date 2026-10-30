#!/usr/bin/env bash

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps 15
