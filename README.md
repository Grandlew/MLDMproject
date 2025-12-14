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

### Deployment Results (Windows, ONNXRuntime CPU)
- Model: DeiT-Tiny + segmentation head (img_size=160)
- Source: logs/urban_run/timing.csv
- Average FPS: **148.54**
- p95 inference time: **6.96 ms**
- Ground truth: not provided (mIoU n/a)

Command used:
.\.venv\Scripts\python.exe .\deploy\log_stream.py --model .\runs\deit_tiny.onnx --source .\samples\urban.mp4 --out_dir .\logs\urban_run
.\.venv\Scripts\python.exe .\deploy\metrics_csv.py --log_dir .\logs\urban_run --csv_out .\reports\log_metrics.csv




