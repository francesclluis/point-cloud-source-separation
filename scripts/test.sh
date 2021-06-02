#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id rgb-depth_2_mix "
OPTS+="--list_val data/test.csv "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_vision sparseresnet18 "
OPTS+="--frame_pool maxpool "
OPTS+="--num_channels 16 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 1 "
OPTS+="--stride_frames 15 "
OPTS+="--frameRate 15 "
OPTS+="--rgbs_feature 1 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
OPTS+="--batch_size_per_gpu 1 "
OPTS+="--workers 0 "

python -u main.py $OPTS
