# Sheaf Neural Networks

## Installaton

```bash
git clone https://github.com/betyavan/SheafNNs.git
cd SheafNN

conda create -n sheaf python=3.11
conda activate sheaf
pip install -r requirements.txt

# where ${CUDA} should be replaced by either cpu, cu118, or cu121 depending on your PyTorch installation
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html

pip install ninja wheel
pip install git+https://github.com/pyg-team/pyg-lib.git

pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

## Run
Чтобы запустить процесс обучения: `bash scripts/run_cora.sh`