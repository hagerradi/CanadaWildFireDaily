# Copilot instructions for `mila-wildfires`

## Commands

Run commands from the repository root.

```bash
pip install -r requirements.txt
```

There is no declared `pytest`, `ruff`, `mypy`, or similar lint/test config in this repository. The main runnable entry points are:

```bash
# Topography preprocessing (requires GDAL-ready environment)
python -m data_preparation.raw_data_preprocessing.topography_rasters_main 2024 --mode local
python -m data_preparation.raw_data_preprocessing.topography_rasters_main 2024 --mode distributed --task-id $SLURM_ARRAY_TASK_ID

# Build per-fire H5 files with weather, fuel, and topography
python -m data_preparation.features_generation.grid_main 2024 --mode local
python -m data_preparation.features_generation.grid_main 2024 --mode distributed --task-id $SLURM_ARRAY_TASK_ID --chunk-size 10

# Append Sentinel-2 imagery into existing H5 files
python -m data_preparation.satellite_generation.satellite_main 2024 --mode local
python -m data_preparation.satellite_generation.satellite_main 2024 --mode distributed --task-id $SLURM_ARRAY_TASK_ID

# Inject quality masks into H5 days with missing/corrupt pixels
python -m data_preparation.diagnosis.quality_diagnosis

# Regenerate tile/day overlap metadata for a year
python -m data_preparation.metadata_generation.tile_mapper 2024

# Precompute offline PyTorch samples before training
python -m samples_generation.data_generator_main --type simple --config configs/default.yaml
python -m samples_generation.data_generator_main --type timeseries --config configs/default.yaml

# Train, then automatically test best_checkpoint.pt if it was created
python -m main --config configs/default.yaml

# SLURM-friendly multi-run training wrapper
python -m parallel_main --config configs/default.yaml --arch unet_segformer --run_id 1
```

## High-level architecture

`configs/settings.py` is the central path/config hub for raw data, outputs, metadata, and generated sample folders. The placeholder paths there must be replaced with absolute local paths; many modules import these constants directly, and the module creates output directories on import.

The data pipeline is staged:

1. `data_preparation/raw_data_preprocessing/` creates 256x256 topography rasters per fire/tile.
2. `data_preparation/features_generation/grid_main.py` creates one HDF5 file per fire under `H5_OUTPUT_FOLDER` and populates tile/day groups with fire masks, weather, VIIRS vegetation indices, topography, and SCANFI fuel layers.
3. `data_preparation/satellite_generation/satellite_main.py` appends Sentinel-2 bands into those existing H5 files.
4. `data_preparation/diagnosis/quality_diagnosis.py` writes optional `quality_mask` datasets for bad tile-days.
5. `data_preparation/metadata_generation/tile_mapper.py` builds `fires_metadata/tile_dob_mapper_<year>.json`, which is then used to merge overlapping fires safely.
6. `samples_generation/` reads H5 + mapper data and materializes offline `.pt` samples into `Samples/` or `Timeseries_Samples/`.
7. `src/train.py` chooses the dataloader and model from `config.model.architecture`; `main.py` trains and immediately runs final evaluation if `best_checkpoint.pt` exists in the checkpoint directory.

Training does **not** read raw H5 files directly. It reads precomputed `.pt` samples from `settings.SAMPLE_FOLDER` or `settings.TIMESERIES_SAMPLE_FOLDER`.

## Key conventions

- Spatial normalization is built around a permanent tile grid: EPSG:3347 Canada Lambert, `GRID_SIZE = 256`, `PIXEL_SIZE = 90`. Tile IDs are `tile_<col>_<row>`, and mapper keys are `tile_<col>_<row>_DOB_<year>_<dob>`.
- Each H5 file is `fire_<fire_id>.h5`. Within it, the expected structure is `tile_*/coords`, `tile_*/static_features`, and `tile_*/days/day_###/{features,satellite}`. Changes to preprocessing should preserve this layout because sample generation depends on these paths.
- Overlapping fires are intentional and are merged at sample-build time. `tile_mapper.py` groups all fires that share the same tile and day-of-burn, and `create_stratified_splits()` unions those overlaps into "super-fires" so train/val/test splits do not leak spatially overlapping fires across splits.
- `quality_mask` is sparse by design: it is only written when a day has invalid pixels. Sample generators skip any tile-day that already contains `quality_mask`.
- `satellite_main.py` is resumable through marker files in `SATELLITE_STATUS_FOLDER`: `success_<fire_id>.txt` means skip on rerun, and `fail_<fire_id>.txt` is deleted before retrying.
- Architecture choice changes the expected sample format:
  - `unet`, `unet_attention`, and `unet_segformer` use offline samples from `Samples/`
  - `unet_age` also uses `Samples/`, but the dataloader must return `delta_t`
  - `unet_convlstm` uses `Timeseries_Samples/`
- `training.use_cumuarea` changes the task from binary new-fire prediction to a 3-class setup (background / old fire / new fire), which changes the loss, metrics, and checkpoint-selection metric in `Trainer`.
- `training.use_cyclical_aspect` encodes `aspect` as sin/cos in sample generation, so channel counts are not always the obvious count from the YAML alone.
- `fuel_viirs.py` hardcodes `ee.Initialize(project='widlfires-vegetation')`. If Earth Engine access is being adjusted for another environment, that project ID is one of the first places to check.
