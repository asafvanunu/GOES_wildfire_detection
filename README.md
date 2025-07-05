# Wildfire Detection Using GOES Satellite Imagery

A Python-based tool for wildfire detection using GOES satellite imagery and geospatial data. This repository includes functions for reading satellite data, preparing area of interest (AOI) shapefiles, running a machine learning model for fire detection, and visualizing results on interactive maps.

---

## 🚀 Features

* Detects wildfires from GOES-16/17/18 MCMI imagery
* Supports AOI shapefiles or manual coordinate entry
* Pre-trained CatBoost model included
* Interactive map visualization (with background imagery and overlayed fire predictions)
* Modular code for easy extension

---

## 📂 Repository Structure

```
wildfire-detection/
│
├── data/                  # Example GOES-18 MCMI image and AOI shapefile
├── notebooks/             # Jupyter Notebook example
├── src/                   # Python functions (wildfire_detection.py)
├── environment.yml        # For Conda users
├── requirements.txt       # For Pip users
├── README.md
└── LICENSE
```

---

## 💻 Installation

### Option 1: Using Conda

```bash
conda env create -f environment.yml
conda activate wildfire-detection
```

### Option 2: Using Pip

```bash
pip install -r requirements.txt
```
### Additional recommended Library:

Install the GOES Python library separately:

```bash
pip install goes
```

More information: [GOES Python Library](https://pypi.org/project/GOES/)

Python version: `3.9.16`

---

## 📝 Example Usage

```python
import geopandas as gpd
from wildfire_detection import predict_fire_for_AOI, plot_fire_prediction_on_interactive_map

AOI_path = r"..\data\AOI\AOI.shp"  # Path to AOI shapefile
MCMI_path = r"..\data\GOES_18\OR_ABI-L2-MCMIPC-M6_G18_s202407071036.nc"  # MCMI file path
ACM_path = r"..\data\GOES_18\OR_ABI-L2-ACMC-M6_G18_s202407071036.nc"  # ACM file path

AOI = gpd.read_file(AOI_path)

fire_raster = predict_fire_for_AOI(
    MCMI_path=MCMI_path,
    AOI_path=AOI_path,
    ACM_path=ACM_path,
    save_raster=False,
    output_path=None
)

plot_fire_prediction_on_interactive_map(prediction_raster=fire_raster, AOI=AOI)
```

---

## 📊 Dependencies

* rioxarray
* xarray
* pandas
* geopandas
* numpy
* joblib
* shapely
* matplotlib
* contextily
* geoviews
* holoviews
* bokeh

---


## 🙏 Acknowledgments

* GOES satellite data: NOAA
* Geospatial tools: GeoPandas, rioxarray
* Machine Learning: CatBoost
* **GOES Python Library:** [https://pypi.org/project/GOES/](https://pypi.org/project/GOES/)
* **GOES Satellite Data Archive:** [https://www.ssec.wisc.edu/datacenter/goes-archive/#GOES16](https://www.ssec.wisc.edu/datacenter/goes-archive/#GOES16)

---


## 📬 Contact
For questions, suggestions, or collaboration, please contact: **[asafyu@post.bgu.ac.il](mailto:asafyu@post.bgu.ac.il)**
