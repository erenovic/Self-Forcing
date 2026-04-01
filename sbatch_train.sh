#!/bin/bash

#SBATCH --job-name=self_forcing_camera
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpumem:80g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=24g
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ecetin@student.ethz.ch

export MASTER_ADDR=localhost

python -m torch.distributed.run \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/self_forcing_dmd.yaml \
    --logdir logs/self_forcing_dmd