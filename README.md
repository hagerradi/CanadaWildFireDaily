# Wildfire Propagation Project

## Data Setup

To run this project, you must download two specific datasets and arrange them according to the directory structure below.

### 1. Fire Growth Points
Download the **Fire growth points** folder from the Open Science Framework.
* **Download Link:** [OSF Fire Growth Points](https://osf.io/f48ry/overview)

### 2. Sample Data (Google Drive)
Download the `Data_Samples` folder containing the necessary covariates files.
* **Download Link:** [Google Drive Data Samples](https://drive.google.com/drive/folders/1j7DQDEBpojiUjJxMKHAAXhGIA1sdNwRe?usp=drive_link)

### Required Directory Structure
For the scripts to reference the data correctly, ensure the folders inside `Data_Samples` are placed at the **same level** as your notebooks.

```text
.
├── DEM_API/
├── ERA5/
├── Fuel_Disturbance/
├── SCANFI/
├── Fire growth points/     # Downloaded from OSF
├── requirements.txt
└── notebook.ipynb
```

## Project Setup

Follow these steps to configure your environment and data directory correctly.

### 1. Environment & GDAL Installation
This project relies on the **Geospatial Data Abstraction Library (GDAL)** for terrain analysis (DEM, slope, and aspect).

* **Download GDAL:** * [OSGeo4W Installer](https://trac.osgeo.org/osgeo4w/) (Recommended for Windows users)
    * [GDAL Binaries](https://gdal.org/en/stable/download.html) (Official site)
* **Installation Support:** If you are unfamiliar with setting up GDAL environment variables, follow this [Step-by-Step Video Guide](https://www.youtube.com/watch?v=oCeffhtywco).
* **Documentation:** Refer to the [gdaldem documentation](https://gdal.org/en/stable/programs/gdaldem.html) for technical details on raster processing.

### 2. Python Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```