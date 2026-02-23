#!/bin/bash

for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=$((i % 8)) wandb agent "${ENTITY}"/sheaf_new/"$1" &
done