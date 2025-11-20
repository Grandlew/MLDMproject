#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw_freiburg
cd data/raw_freiburg

# Try the official downloader
if [ ! -f download_freiburg_forest_annotated.sh ]; then
  wget -q http://deepscene.cs.uni-freiburg.de/static/datasets/download_freiburg_forest_annotated.sh
  chmod +x download_freiburg_forest_annotated.sh
fi
bash download_freiburg_forest_annotated.sh || true

# Fallback: download the tarball directly if the script didn’t fetch it
if [ ! -f freiburg_forest_annotated.tar.gz ]; then
  wget -q http://deepscene.cs.uni-freiburg.de/static/datasets/freiburg_forest_annotated.tar.gz
fi

tar -xzf freiburg_forest_annotated.tar.gz
rm -f freiburg_forest_annotated.tar.gz*
echo "Freiburg Forest extracted to data/raw_freiburg/freiburg_forest_annotated"
