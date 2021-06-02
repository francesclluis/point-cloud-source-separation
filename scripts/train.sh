#!/bin/bash

OPTS=""
OPTS+="--id default "
OPTS+="--list_train data/train.csv "
OPTS+="--list_val data/val.csv "

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
OPTS+="--rgbs_feature 0 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--workers 12 "
OPTS+="--batch_size_per_gpu 40 "
OPTS+="--lr_vision 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 200 "

# display
OPTS+="--disp_iter 20 "
OPTS+="--num_val 256 "
OPTS+="--eval_epoch 1 "

python -u main.py $OPTS
