# Sheaf Neural Networks

## Installaton

```
git clone https://github.com/betyavan/SheafNNs.git
cd SheafNN

conda create -n sheaf python=3.11
conda activate sheaf
pip install -r requirements.txt

pip install ninja wheel
pip install git+https://github.com/pyg-team/pyg-lib.git
```

## Run
Чтобы запустить процесс обучения: `bash scripts/run_cora.sh`