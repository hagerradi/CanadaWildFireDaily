# Wildfire Daily Propagation

This repository contains a complete, end-to-end deep learning pipeline for forecasting daily wildfire spread. It processes raw tabular/vector fire perimeters, generates topographical rasters via GDAL, downloads satellite imagery, and trains models to predict future fire boundaries.

---

## Part 1: Data Acquisition & Configuration

To run this project, you must download the core datasets and configure the project to locate them.

### 1.1 Download the Data
1. **Fire Growth Points:** Download the target year folders from the Open Science Framework.
   * **Download Link:** [OSF Fire Growth Points](https://osf.io/f48ry/overview)
2. **Covariates Data:** Download the `Data_Samples` folder containing the necessary weather, fuel, and topography files.
   * **Download Link:** [Google Drive Data Samples](https://drive.google.com/drive/folders/1j7DQDEBpojiUjJxMKHAAXhGIA1sdNwRe?usp=drive_link)

### 1.2 Directory Structure
Extract the downloaded files so your raw data directory looks exactly like this:
```text
DATA_FOLDER/
├── DEM_API/
│   └── DEM_Tiles/
├── ERA5/
├── SCANFI/
│   └── 2020/
└── Fire_growth_points/
    ├── Firegrowth_pts_v1_1_2024/   # From OSF
    ├── Firegrowth_pts_v1_1_2023/   # From OSF
    ├── Firegrowth_pts_v1_1_2022/   # From OSF
    ├── Firegrowth_pts_v1_1_2021/   # From OSF
    └── Firegrowth_pts_v1_1_2020/   # From OSF
```

### 1.3 Path Configuration (settings.py)
Before running any scripts, open configs/settings.py. This file acts as the central nervous system for the pipeline. You must update the following three variables with their **absolute** paths on your machine:

* `PROJECT_FOLDER`: The absolute path to the root repository folder that contains all of the code (e.g., the src, configs, and data_preparation directories).
* `DATA_FOLDER`: The absolute path to the directory containing the raw downloaded datasets, structured exactly as shown in Section 1.2 above.
* `OUTPUT_FOLDER`: The absolute path to the directory where all generated data will be saved. This is where the scripts will output the final .h5 PyTorch datasets.

** Note on Folder Creation:** You only need to manually set up the directories containing the downloaded raw data (like `ERA5/`, `SCANFI/`, etc.). As soon as you run any script in this project, `settings.py` will automatically create the entire required `OUTPUT_FOLDER` structure (including the DEM, Satellite Status, and Metadata directories) if they do not already exist.

---

## Part 2: Environment Setup

### 2.1 Virtual Environment
Create and activate a fresh Python virtual environment:
```bash
# Create the environment
python -m venv env_wildfires_spread

# Activate (Linux/Mac)
source env_wildfires_spread/bin/activate

# Activate (Windows)
env_wildfires_spread\Scripts\activate
```

### 2.2 Install Python Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### 2.3 GDAL Installation (Topography Generation)
Exceptionally, building the raw topography rasters requires the **Geospatial Data Abstraction Library (GDAL)**. Because GDAL has complex system-level dependencies, creating a dedicated Conda environment is the most stable and recommended approach.

You can use the following pipeline to install Miniconda, speed up the solver, and install GDAL safely:

**1. Download and Install Miniconda (If not already installed)**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_22.11.1-1-Linux-x86_64.sh
bash Miniconda3-py310_22.11.1-1-Linux-x86_64.sh

source ~/.bashrc
```

**2. Configure the Fast Libmamba Solver**
Standard Conda can be very slow when resolving spatial libraries. Set up `libmamba` to make the installation incredibly fast:
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

**3. Create the Environment and Install GDAL**
Create the environment and install GDAL alongside the required spatial Python libraries from `conda-forge`:
```bash
# Create and activate the environment
conda create -p conda_wildfires_env python=3.10
conda activate conda_wildfires_env

# Install GDAL and spatial dependencies
conda install -c conda-forge gdal rasterio pyproj pandas numpy
```

**4. Verify the Installation**
Ensure the GDAL command-line tools are recognized by your system:
```bash
gdalwarp --version
gdaldem --version
```

### 2.4 Google Earth Engine Authentication
The Features Generation pipeline (specifically for downloading `fuel_viirs` data) requires access to Google Earth Engine via the Python API. 

**Option A: Local Environment Setup**
If you are running this on a personal laptop or desktop with a web browser:
1. With your Python virtual environment activated, run:
```bash
earthengine authenticate
```
2. The terminal will output a URL. Open it in your browser, log in using a Google account registered for Earth Engine, and paste the authorization token back into your terminal.

**Option B: Remote / HPC Cluster Setup (Headless)**
If you are running this on a distant environment (like a SLURM cluster) without a web browser, you must authenticate locally first and securely transfer your token.

1. **Authenticate Locally:** Run `earthengine authenticate` on your personal laptop.
2. **Locate the Credentials File:** 
    * Mac/Linux: `~/.config/earthengine/credentials`
    * Windows: `C:\Users\YourName\.config\earthengine\credentials`
3. SSH into your cluster and create a secure, hidden directory:
   ```bash
   mkdir -p ~/.config/earthengine/
   chmod 700 ~/.config/earthengine/
   ```
4. **Transfer the Token:** Use Secure Copy (`scp`) from your local laptop terminal to push the file to your cluster (replace the paths/usernames with your own):
```bash
scp "C:/Users/YourName/.config/earthengine/credentials" username@login.server.edu:~/.config/earthengine/
```
5. **Lock Down the File:** Ensure only your user account has permission to read the token:
```bash
chmod 600 ~/.config/earthengine/credentials
```

**Configuring the GEE Cloud Project:**
Google Earth Engine now requires a registered Cloud Project to route API requests. In our codebase, the GEE initialization is set up like this:
```python
ee.Initialize(project='widlfires-vegetation')
```
* **External Users:** You must [create your own GEE Cloud Project](https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup). Once created, simply update the `project='...'` parameter in the features generation scripts (`fuel_viirs.py`) to match your new Project ID.

---

## Part 3: Raw Data Preprocessing (Topography)
Before we can build our dataset, we need to generate uniform topographical grids for every fire. This pipeline uses GDAL to crop a massive DEM mosaic, project it to EPSG:3347 (Meters), calculate Slope and Aspect, and create an averaged 3x3 background DEM.

### 3.1 Build the DEM Mosaic (VRT)
First, you need to stitch all of your individual downloaded DEM tiles into a single Virtual Raster (VRT). This allows the Python script to seamlessly crop fire boxes that overlap multiple tile boundaries.

Activate your GDAL Conda environment and run the build command at the same directory level as your DEM_Tiles folder (typically inside DATA_FOLDER/DEM_API/):

```bash
# Activate the GDAL environment you created in Part 2
conda activate conda_wildfires_env

# Navigate to the folder containing DEM_Tiles
cd /path/to/your/DATA_FOLDER/DEM_API/

# Build the Virtual Raster
gdalbuildvrt dem_mosaic.vrt DEM_Tiles/*.tif
```

### 3.2 Generate the Fire Topography
Now, navigate back to your `PROJECT_FOLDER` (the root of this repository) to run the GDAL generation script. This step processes the raw topographical data (elevation, slope, aspect) for your fires and **must be completed before generating the final grid files**.

You can run this script either locally on your machine or distributed across a cluster.

**Option A: Local Execution (Single Machine)**
Ideal for testing or processing a small subset of data. This mode processes the topography for the fires sequentially on your current machine.
```bash
cd /path/to/your/PROJECT_FOLDER/

python -m data_preparation.raw_data_preprocessing.topography_rasters_main 2024 \
  --mode local
```

**Option B: Distributed Execution (SLURM Cluster)**
Ideal for processing entire years. This uses SLURM array task IDs to process the heavy GDAL raster operations in parallel.
```bash
cd /path/to/your/PROJECT_FOLDER/

python -m data_preparation.raw_data_preprocessing.topography_rasters_main 2024 \
  --mode distributed \
  --task-id $SLURM_ARRAY_TASK_ID
```

### Parameters Explained
* **`YEAR` (Positional):** The target year to process (e.g., `2024`). You must run this command for **all target years** in your dataset independently (e.g., 2020 through 2024).
* **`--mode`:** 
  * `local`: Runs sequentially on your current machine.
  * `distributed`: Tells the script to look for a specific subset of fires based on the `--task-id`, allowing hundreds of nodes to process the dataset simultaneously without overlapping.
* **`--task-id`:** Required if using `--mode distributed`. It maps to your cluster's job array index (e.g., `$SLURM_ARRAY_TASK_ID`), determining exactly which topographical chunk the current job is responsible for.

---

## Part 4: Data Preparation (The H5 Pipeline)

Once the raw rasters are generated, the data is packaged into structured `.h5` files. This is split into three sequential steps to ensure data integrity and memory efficiency.

### 4.1 Features Generation (Base H5 Grids)
This script constructs the foundational `.h5` file for each fire. It precisely maps daily weather forcing (ERA5), fuel data, and the topographical rasters generated in Part 3 onto a standardized 2D spatial grid.

You can run the script either locally on your machine or distributed across a cluster (highly recommended for large datasets). 

**Option A: Local Execution (Single Machine)**
Ideal for testing or processing a small subset of data. This mode processes the fires sequentially on your current machine.
```bash
cd /path/to/your/PROJECT_FOLDER/

python -m data_preparation.features_generation.grid_main 2024 \
  --mode local
```

**Option B: Distributed Execution (SLURM Cluster)**
Ideal for processing entire years. This uses SLURM array task IDs to process multiple chunks of fires in parallel.
```bash
cd /path/to/your/PROJECT_FOLDER/

python -m data_preparation.features_generation.grid_main 2024 \
  --mode distributed \
  --task-id $SLURM_ARRAY_TASK_ID \
  --chunk-size 10
```

### Parameters Explained
* **`YEAR` (Positional):** The target year to process (e.g., `2024`). You must run this command for **all target years** in your dataset independently.
* **`--mode`:** 
  * `local`: Runs sequentially on your current machine.
  * `distributed`: Tells the script to look for a specific subset of fires based on the `--task-id`, allowing hundreds of nodes to process the dataset simultaneously without overlapping.
* **`--task-id`:** Required if using `--mode distributed`. It maps to your cluster's job array index (e.g., `$SLURM_ARRAY_TASK_ID`), determining exactly which chunk of fires the current job is responsible for.
* **`--chunk-size`:** Used with `--mode distributed`. Defines how many fires a single job or process should handle. Generating these dense 3D/4D `.h5` files is heavily I/O bound. During our testing, processing batches of **5 to 10 fires** per parallel job yielded optimal performance.


### 4.2 Satellite Generation (Sentinel-2 Imagery)

This script queries the Microsoft Planetary Computer to download Sentinel-2 multispectral imagery (B02, B03, B04, B06, B11, B12) that aligns spatially and temporally with the grids generated in the previous step. It directly appends the new `satellite/` group and Cloud Cover statistics into your existing `.h5` files.

**Option A: Local Execution**
```bash
cd /path/to/your/PROJECT_FOLDER/

python -m data_preparation.satellite_generation.satellite_main 2024 \
  --mode local
```

**Option B: Distributed Execution (SLURM Cluster)**
```bash
cd /path/to/your/PROJECT_FOLDER/

python -m data_preparation.satellite_generation.satellite_main 2024 \
  --mode distributed \
  --task-id $SLURM_ARRAY_TASK_ID
```
*(Note: The `YEAR`, `--mode`, and `--task-id` parameters function exactly as described in the previous sections).*

**Key Execution Notes:**
* **Parallel Execution (1 Fire per Job):** When running on a cluster, we highly recommend allocating exactly **one fire per parallel job** (no chunking). This maximizes download throughput and isolates API failures.
* **API Throttling & Fault Tolerance:** Because the script sends thousands of requests to the Planetary Computer STAC API, connections can occasionally be throttled, dropped, or timed out. 
* **Resumability (Status Tracking):** To handle dropped connections, the script uses a marker system. It writes empty trace files (`success_{fire_id}.txt` or `fail_{fire_id}.txt`) to the `SATELLITE_STATUS_FOLDER`. 
  * If your pipeline crashes or times out due to API limits, you can simply **re-run the exact same command**. The code will instantly bypass successful fires, delete the fail flags of the broken ones, and retry the downloads automatically.


### 4.3 Quality Control & Diagnosis
Wildfire datasets rely on overlapping multiple different data sources, which can occasionally contain missing features or corrupted pixels (`NaN`/`Inf`). This script scans every fully-built `.h5` file and performs a rigorous pixel-by-pixel check across all static and dynamic features.

Run the diagnosis script:
```bash
cd /path/to/your/PROJECT_FOLDER/

python -m data_preparation.diagnosis.quality_diagnosis
```

**Key Execution Notes:**
* **Dynamic Mask Injection:** If a specific day contains any corrupted or missing data, the script generates and embeds a binary `quality_mask` (where `1 = bad pixel`) directly into that day's group within the `.h5` file.
* **Storage Efficiency:** To save disk space, perfectly clean days do not receive a mask. The PyTorch Dataloader is programmed to assume a day is perfectly clean unless it finds a `quality_mask` present.
* **Model Integration:** During training, the samples builder will automatically detect these `quality_mask` flags and skip corrupted days, ensuring the model is only fed complete data.

### 4.4 Tile Mapper Generation (Metadata)

This script scans the fires (from the dataframes) and builds a mapping dictionary (`tile_dob_mapper_{year}.json`). It groups together any individual fires that share the exact same 256x256 grid tile on the exact same Day of Burn (DOB). 

This metadata is critical for the samples generation (next section). It allows the pipeline to safely merge overlapping fire masks into a single "Super-Fire" environment, preventing spatial data leakage between the training and testing sets.

```bash
cd /path/to/your/PROJECT_FOLDER/

python -m data_preparation.metadata_generation.tile_mapper 2024
```
*(Note: Unlike previous steps, this script is very fast and runs locally in a single pass without needing SLURM distribution.)*

**Key Execution Notes:**
* **Output Location:** The resulting JSON files are saved directly into your `fires_metadata/` folder.
* **Pre-Computed Files Provided:** We have already provided the generated mapper files for **2020 through 2024** directly in this repository! You **do not** need to run this script unless you are processing brand new years or modifying the spatial grid parameters.

---

## Part 5: Samples Generation

This is the final data preparation step before model training. This script reads the raw daily `.h5` files and the JSON tile mappers (from Part 4) to compile finalized, ready-to-train PyTorch samples. 

If you are planning to split your README into two separate documents (a Data one and a Model one), adding a highly visible "callout" or "warning" block is the absolute best way to handle this. It immediately catches the reader's eye and prevents them from running the script and getting a crash if they skipped the data download steps.

Here is how you can update that introduction to explicitly state the CSV requirement:

***

## Part 5: Samples Generation

This is the final data preparation step before model training. This script reads the raw daily `.h5` files, the JSON tile mappers (from Part 4), and the original **Fire Growth CSV files** to compile finalized, ready-to-train PyTorch samples. 

> ⚠️ **Important Data Requirement:** > This step absolutely requires the raw Fire Growth CSV files (e.g., `Firegrowth_pts_v1_1_2024.csv`). If you have not downloaded these yet, please refer to **Part 1: Data Acquisition** to download them from the Open Science Framework (OSF) and place them in your `DATA_FOLDER`.

Crucially, this step handles the **Train/Validation/Test splitting** using the CSV fire growth data. It uses the tile mappers to guarantee that geographically overlapping fires are kept strictly within the same fold, ensuring zero spatial data leakage between your training and testing sets.

To generate the dataset, run the following command from the root of your project:

```bash
cd /path/to/your/PROJECT_FOLDER/

python -m samples_generation.data_generator_main --config configs/default.yaml --type simple
```

### Script Arguments:
* `--config`: The path to your configuration file (e.g., `configs/default.yaml`).
* `--type`: The formatting style of the generated samples. 
  * `choices=["simple", "timeseries"]`
  * **`simple`**: Generates standard single-step spatial inputs (Day $T \rightarrow$ Predict Day $T+1$). Best for standard U-Net architectures.
  * **`timeseries`**: Generates sequential temporal inputs (e.g., Days $T, T+1, T+2 \rightarrow$ Predict Day $T+3$). Best for spatio-temporal architectures like ConvLSTM.

**Key Execution Notes:**
* **Dataset Normalization:** During generation, the script automatically calculates the global mean and standard deviation for all features across the training split and saves a `.json` file. This ensures the validation and test sets are normalized exclusively using training statistics.
* **Output Location:** The script saves each generated sample as an individual PyTorch (`.pt`) file inside designated split subfolders (e.g., `train`, `val`, `test`) within either your `SAMPLE_FOLDER` or `TIMESERIES_SAMPLE_FOLDER`. Each file contains a dictionary with the following:
  * `x`: The input feature tensor(s).
  * `y`: The ground truth target mask.
  * `delta_t`: The satellite image age in days (included for `simple` samples only).
* **Optimization Note:** It is possible to build a PyTorch Dataset that reads directly from the raw daily `.h5` files during training using the code provided in `samples_generation/data_generator.py` and `samples_generation/data_generator_timeseries.py`. However, we pre-compute and save these ready-to-batch `.pt` tensors purely for optimization purposes to significantly accelerate the training loop and maximize GPU utilization.

## Part 6: Modeling

With the samples generated, the model is ready to train.

### 6.1 Configuration (`configs/default.yaml`)
All training hyperparameters, hardware settings, and logging preferences are centralized in `configs/default.yaml`. Before training, you can adjust this file to suit your needs.

### 6.2 Model Selection & Customization
Multiple model architectures are implemented in the `src/models/` directory to handle different temporal and spatial requirements. 

To switch between architectures (which will automatically configure the corresponding dataloaders, such as swapping from single-day static prediction to a 3-day sliding window), open your `configs/default.yaml` and update the `architecture` parameter under the `model` section to one of the following options:

1. **Standard UNet** (`architecture: 'unet'`): 
   The baseline spatial U-Net model.
2. **Age-Encoding UNet** (`architecture: 'unet_age'`): 
   A U-Net that explicitly encodes the satellite age (the time gap in days between the fire event and the satellite acquisition).
3. **Spatiotemporal UNet** (`architecture: 'unet_convlstm'`): 
   A U-Net featuring a **ConvLSTM** bottleneck for recurrent time-series processing (e.g., 3-day sliding window forecasting).
4. **Attention UNet** (`architecture: 'unet_attention'`): 
   A U-Net utilizing attention gates in the skip connections to help the model focus on the most critical spatial features and suppress irrelevant background noise.
5. **UNet-SegFormer** (`architecture: 'unet_segformer'`): 
   A hybrid vision-transformer architecture that replaces the standard CNN encoder with SegFormer's Mix Vision Transformer (MiT), paired with a standard U-Net decoder for heavy pixel-level accuracy. 

### 6.3 Training the Model
The main entry point for the training pipeline is `main.py`, located at the root of the project. 

To run the training loop locally:
```bash
cd /path/to/your/PROJECT_FOLDER/

python -m main --config configs/default.yaml
```

**Key Training Features:**
* **Custom Loss:** Automatically initializes a custom `CombinedLoss` (Focal + Dice) to handle the extreme class imbalance of wildfire pixels.
* **Automatic Checkpointing:** The `Trainer` monitors the Validation IoU. Whenever the model improves, it automatically overwrites and saves `best_checkpoint.pt` to your configured `checkpoint_dir`.
* **Comet.ml Integration:** If `enabled: true` in your config, the pipeline will automatically log learning rates, loss curves, and epoch-by-epoch evaluation metrics directly to your Comet dashboard. It also uploads visual grid predictions at the end of epochs so you can watch the model learn.

### 5.4 Automatic Evaluation (Testing)
At the end of the `train` loop, `main.py` automatically looks for the `best_checkpoint.pt` generated during training. 

If found, it initiates the `test()` protocol on the holdout test split. This step computes the final unbiased Macro IoU, F1 scores, Precision, and Recall. Furthermore, it uploads high-resolution 4-pane visual predictions (Previous Fire Mask, Ground Truth, Model Prediction, and Probability Heatmap) to CometML for your final visual analysis.