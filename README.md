# Sheaf Neural Networks

## Installaton

```bash
git clone https://github.com/betyavan/SheafNNs.git
cd SheafNN

conda create -n sheaf python=3.11
conda activate sheaf
pip install -r requirements.txt

conda install -c dglteam/label/cu118 dgl
```

## Run
Чтобы запустить процесс обучения: `bash scripts/run_cora.sh`