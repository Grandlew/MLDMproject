\# Robot-Ready DeiT-Tiny — Intermediate Checkpoint



\*\*Paper:\*\* Data-Efficient Image Transformers (DeiT).  

\*\*Paper PDF (local path for reference):\*\* /mnt/data/2012.12877v2.pdf



\## Quickstart (Windows PowerShell)

```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

python train.py --config configs/imagenette\_deit\_tiny.yaml --epochs 2 --amp --save\_dir runs/

python eval.py --checkpoint runs/deit\_tiny.ckpt --amp



