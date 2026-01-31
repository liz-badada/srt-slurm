#!/bin/bash
set -ex

# cd /home/jinyanc/github/srt-slurm

# srtctl apply -f recipes/h100/1k1k/mtp/h100-fp8-1p2d-max-dep-mtp.yaml

srtctl apply -f recipes/h100/1k1k/stp/h100-fp8-1p2d-max-tp.yaml

# srtctl apply -f recipes/h100/8k1k/mtp/h100-fp8-1p1d-max-dep-mtp.yaml
