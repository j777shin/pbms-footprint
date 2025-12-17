# Building Footprint Segmentation (ReFineNet) + Sentinel‑2 / OSM / YOLOv8

This repo segments building footprints using a **ReFineNet (ResNet50 backbone)** model and adds an optional geospatial pipeline to:

- Fetch **Sentinel‑2** imagery around a coordinate (Sentinel Hub)
- Predict footprints
- Clip results to a **radius (e.g. 100 m)** around that coordinate
- Optionally use **OpenStreetMap** building footprints to filter/validate results
- Optionally fuse results with **YOLOv8** (if you provide building-trained weights)

## Requirements

- Python **3.12** recommended
- PyTorch + torchvision installed for your platform (CPU/CUDA/ROCm)

## Install

### Create an environment (recommended)

```bash
conda create -n building-seg python=3.12 -y
conda activate building-seg
```

### Install core dependencies

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Install optional geospatial dependencies (Sentinel‑2 / OSM / GeoJSON)

```bash
python -m pip install -r requirements-geo.txt
```

If `rasterio`/`geopandas` fails on macOS, use conda-forge:

```bash
conda install -c conda-forge rasterio geopandas shapely pyproj -y
python -m pip install osmnx sentinelhub
```

### Install optional YOLOv8 dependency (fusion)

```bash
python -m pip install -r requirements-yolo.txt
```

## Sentinel Hub credentials (Client ID / Secret)
Create a file named `.env` in the repo root:

```bash
SENTINELHUB_CLIENT_ID=...
SENTINELHUB_CLIENT_SECRET=...
```

(`.env` is ignored by `.gitignore`.)

## Set coordinates (center point)

Edit `coords.yaml` (repo root). Example:

```yaml
lat: 16.040115454292344
lon: 108.22135618664987
radius_m: 50
```

The lat/lon is the **center** of the circle; predictions are clipped to `radius_m`.

## Run end-to-end and get PNG results

### Quickstart: dataset → train → inference 

From the repo root:

```bash
conda activate building-seg
cd /Users/jshin6/code/building-footprint-segmentation-resnet50
```

1) Prepare dataset (Sentinel‑2 + OSM weak labels):

```bash
python scripts/prepare_s2_osm_dataset.py \
  --center-lat 16.040115454292344 \
  --center-lon 108.22135618664987 \
  --sample-radius-m 3000 \
  --n-train 200 --n-val 40 --n-test 40 \
  --out-root data/s2_osm/building_patches
```

2) Train + export weights:

```bash
python train_refinenet.py \
  --config configs/train_s2_osm.yaml \
  --epochs 10 \
  --log-root logs \
  --export-dir weights \
  --export-name best.pt
```

3) Inference + PNG outputs:

```bash
python run_nearby.py \
  --coords coords.yaml \
  --dotenv .env \
  --weights weights/best.pt \
  --outdir geo_outputs \
  --use-osm --osm-filter
```

### Recommended: `run_nearby.py` (one command)
```bash
python run_nearby.py \
  --coords coords.yaml \
  --dotenv .env \
  --weights /path/to/best.pt \
  --outdir geo_outputs \
  --use-osm \
  --osm-filter \
  --osm-building industrial \
  --yolo-weights /path/to/yolo_buildings.pt \
  --fuse-mode intersection
```

This runs the full pipeline and writes PNG + GeoJSON outputs.

```bash
python run_nearby.py \
  --coords coords.yaml \
  --weights /path/to/best.pt \
  --outdir geo_outputs \
  --use-osm --osm-filter
```

### Step 1: Create a weakly-labeled dataset (Sentinel‑2 + OSM)

This creates `data/s2_osm/building_patches/{train,val,test}/{images,labels}` in the same format the trainer expects.

```bash
python scripts/prepare_s2_osm_dataset.py \
  --center-lat 16.040115454292344 \
  --center-lon 108.22135618664987 \
  --sample-radius-m 3000 \
  --n-train 200 --n-val 40 --n-test 40 \
  --out-root data/s2_osm/building_patches
```

### Step 2: Train and export `weights/best.pt`

```bash
python train_refinenet.py \
  --config configs/train_s2_osm.yaml \
  --epochs 10 \
  --log-root logs \
  --export-dir weights \
  --export-name best.pt
```

Then run inference using the exported weights:

```bash
python run_nearby.py --coords coords.yaml --weights weights/best.pt --outdir geo_outputs
```

**Outputs** (in `geo_outputs/`):

- `s2_rgb.png`
- `mask_refinenet.png`
- `mask_fused.png`
- `overlay_fused.png`
- `predicted_buildings.geojson`
- `osm_buildings.geojson` (if `--use-osm`)

### Alternative: module CLI

```bash
python -m building_footprint_segmentation.geo.cli \
  --coords coords.yaml \
  --refinenet-weights /path/to/best.pt \
  --use-osm --osm-filter \
  --outdir geo_outputs
```

```bash
python run_nearby.py \
  --coords coords.yaml \
  --weights /path/to/best.pt \
  --yolo-weights /path/to/yolo_buildings.pt \
  --fuse-mode intersection \
  --outdir geo_outputs
```

Fusion modes:
- `intersection`: keep pixels both models agree on (usually cleaner, fewer false positives)
- `union`: keep pixels from either model (usually higher recall)

## OSM filtering (“certain buildings”)

If you want only a specific OSM building tag (example: `building=industrial`):

```bash
python run_nearby.py \
  --coords coords.yaml \
  --weights /path/to/best.pt \
  --use-osm --osm-filter \
  --osm-building industrial \
  --outdir geo_outputs
```

## Notes / troubleshooting

- **Python 3.13** often causes issues with some scientific packages. Use **Python 3.12**.
- If you see `ModuleNotFoundError: torchvision`, install `torchvision` matching your `torch` build.
- Sentinel‑2 is **10 m/pixel** (RGB). Small buildings may be only a few pixels wide; best quality typically comes from fine-tuning on Sentinel‑2 or using stronger priors (OSM/YOLO).