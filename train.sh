#!/bin/bash

export MASTER_ADDR=localhost

python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/self_forcing_dmd.yaml \
    --logdir logs/self_forcing_dmd