#!/bin/bash
set -ex

# cd /home/jinyanc/github/srt-slurm
# srtctl apply -f recipies/h200/8k1k/bs128-agg-tp.yaml

srtctl apply -f recipies/h100/1k1k/mtp/h100-fp8-2p4d-max-dep-mtp.yaml
# srtctl apply -f recipies/h100/1k1k/stp/h100-fp8-2p4d-max-tp.yaml
# srtctl apply -f recipies/h100/8k1k/mtp/h100-fp8-2p2d-max-dep-mtp.yaml
