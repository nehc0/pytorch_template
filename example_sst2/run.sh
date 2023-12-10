#!/bin/sh

# torchrun automatically spawns the processes!

# single-node, multi-worker
# for example, 1 machine, which has 1 GPU
# run the command below
torchrun --standalone --nnodes=1 --nproc_per_node=1 ./main.py

# multi-node, multi-worker
# for example, 2 machines, where one has 4 GPUs and the other has 2 GPUs
# run the fist command below on the first machine
#torchrun --nnodes=2 --node_rank=0 --nproc-per-node=4 --rdzv-id=$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR ./main.py
# and run the second command below on the second machine
#torchrun --nnodes=2 --node_rank=1 --nproc-per-node=2 --rdzv-id=$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR ./main.py