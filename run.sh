#!/bin/bash
set -ex


# cd /home/jinyanc/github/srt-slurm


################################ 1k1k ################################
# 1p2d-ctx_tp-gen_tp ok
# srtctl apply -f recipes/h100/1k1k/stp/h100-fp8-1p2d-max-tp.yaml
# srtctl apply -f recipes/h100/1k1k/mtp/h100-fp8-1p2d-max-tp-mtp.yaml

# 1p1d-ctx_tp-gen_dep
# srtctl apply -f recipes/h100/1k1k/stp/h100-fp8-1p1d-max-dep.yaml
# srtctl apply -f recipes/h100/1k1k/mtp/h100-fp8-1p1d-max-dep-mtp.yaml
################################# 1k1k ################################


################################ 1k8k ################################
# 1p2d ok
# srtctl apply -f recipes/h100/1k8k/stp/h100-fp8-1p2d-max-tp.yaml
# srtctl apply -f recipes/h100/1k8k/mtp/h100-fp8-1p2d-max-tp-mtp.yaml

# 1p1d
# srtctl apply -f recipes/h100/1k8k/stp/h100-fp8-1p1d-max-dep.yaml
# srtctl apply -f recipes/h100/1k8k/mtp/h100-fp8-1p1d-max-dep-mtp.yaml
################################# 1k8k ################################


################################ 8k1k ################################
# 1p1d-ctx_tp-gen_tp ok
# srtctl apply -f recipes/h100/8k1k/stp/h100-fp8-1p1d-max-tp.yaml
# srtctl apply -f recipes/h100/8k1k/mtp/h100-fp8-1p1d-max-tp-mtp.yaml

# 1p1d-ctx_tp-gen_dep
srtctl apply -f recipes/h100/8k1k/stp/h100-fp8-1p1d-max-dep.yaml
srtctl apply -f recipes/h100/8k1k/mtp/h100-fp8-1p1d-max-dep-mtp.yaml
################################ 8k1k ################################













# legacy
# srtctl apply -f recipes/h100/1k1k/mtp/h100-fp8-1p2d-max-dep-mtp.yaml
# srtctl apply -f recipes/h100/1k1k/stp/h100-fp8-1p2d-max-tp.yaml
# srtctl apply -f recipes/h100/8k1k/mtp/h100-fp8-1p1d-max-dep-mtp.yaml
