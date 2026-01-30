#!/bin/bash
set -ex

cd $HOME/github/srt-slurm

# Try pip install with --break-system-packages, fallback without it
pip install -e . --break-system-packages 2>/dev/null || pip install -e . --user

# Add local bin to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
fi

# Export for current session
export PATH=$PATH:$HOME/.local/bin

make setup ARCH=x86_64

echo "✅ Setup complete! Run 'source ~/.bashrc' or start a new terminal."


# enroot import docker://lmsysorg/sglang:v0.5.8-cu130-runtime