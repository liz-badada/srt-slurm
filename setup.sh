#!/bin/bash
set -ex

cd $HOME/github

cd srt-slurm && pip install -e . --break-system-packages && echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc && source ~/.bashrc

make setup ARCH=x86_64