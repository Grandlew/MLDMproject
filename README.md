
# Robot-Ready DeiT-Tiny (Cityscapes) — Reproduction & Deployment 

**Paper reproduced:** Data-Efficient Image Transformers (DeiT, ICML’21).  
**Task:** Road-scene **semantic segmentation** on **Cityscapes** with a lightweight ViT (DeiT-Tiny) + deployable **ONNX**.  
**Status:** Implementation complete; training/eval done; ONNX + demo overlay & metrics committed.
**Dataset link:** https://disk.360.yandex.ru/d/gONYDzRB7lf4-g
## Quick Reproduce

```powershell
# venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
git lfs install

# prepare Cityscapes
.\.venv\Scripts\python.exe .\scripts\prepare_cityscapes.py --out .\data\cityscapes_512x1024

# train (fast, 256px)
.\.venv\Scripts\python.exe .\train.py --config .\configs\cityscapes_deit_tiny.yaml --amp

# evaluate (overall + per-class)
.\.venv\Scripts\python.exe .\eval.py --checkpoint .\runs_city\deit_tiny_best.ckpt --amp
.\.venv\Scripts\python.exe .\eval.py --checkpoint .\runs_city\deit_tiny_best.ckpt --amp --per_class > .\reports\city_per_class.txt

# qualitative grid
.\.venv\Scripts\python.exe -m scripts.vis_preds --ckpt .\runs_city\deit_tiny_best.ckpt --out .\reports\city_val_vis.jpg

# export ONNX (dynamic shapes)
.\.venv\Scripts\python.exe -m scripts.export_onnx --ckpt .\runs_city\deit_tiny_best.ckpt --out .\runs\deit_tiny_dynamic.onnx --img_size 256 --dynamic

# runtime on sample video + CSV
$log = ".\logs\urban_run_256"
.\.venv\Scripts\python.exe .\deploy\log_stream.py --model .\runs\deit_tiny_dynamic.onnx --source .\samples\urban.mp4 --img_size 256 --out_dir $log --thr 0.5
.\.venv\Scripts\python.exe .\deploy\metrics_csv.py --log_dir $log --csv_out .\reports\log_metrics_256.csv



