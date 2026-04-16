#!/bin/bash
# vast_setup.sh — Run this ONCE on the vast.ai instance after SSH in
#
# Usage:
#   ssh -p 40127 root@87.197.140.238
#   curl -sL https://raw.githubusercontent.com/pmmathias/krr-chat/experiment/autoregressive-krr/vast_setup.sh | bash
#
# Or manually:
#   bash vast_setup.sh

set -e
echo "=== vast.ai setup for AR-KRR GPU experiments ==="

# Clone repo + checkout experiment branch
if [ ! -d "krr-chat" ]; then
    git clone https://github.com/pmmathias/krr-chat.git
fi
cd krr-chat
git checkout experiment/autoregressive-krr
git pull

# Install Python deps (PyTorch should be pre-installed in the vast.ai PyTorch template)
pip install --quiet gensim tokenizers datasets

# Verify GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Quick test (D=4096, should take ~2 min):"
echo "  python src/autoregressive/train_gpu.py --D 4096 --cg-maxiter 100"
echo ""
echo "Full run (D=12288, 2L, 3M tokens, ~15 min):"
echo "  python src/autoregressive/train_gpu.py --D 12288 --layers 2"
echo ""
echo "D-scaling sweep:"
echo "  for D in 6144 12288 24576 49152; do"
echo "    python src/autoregressive/train_gpu.py --D \$D --output data/autoregressive/model_gpu_D\${D}.pkl"
echo "  done"
