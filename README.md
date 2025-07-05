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

Python version: `3.9.16`

---

## 📝 Example Usage

The `/notebooks/wildfire_example.ipynb` notebook demonstrates:

1. Loading MCMI and AOI data
2. Running wildfire detection
3. Visualizing fire predictions on an interactive map

Example code snippet:

```python
from src.wildfire_detection import detect_fire, create_aoi_polygon

# Create AOI from coordinates
aoi = create_aoi_polygon(x_min=-100, x_max=-99, y_min=35, y_max=36)

# Run detection
result = detect_fire(MCMI_path='data/MCMI_sample.nc', ACM_path=None, AOI_path='data/AOI_sample.shp')
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

## 🔖 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

* GOES satellite data: NOAA
* Geospatial tools: GeoPandas, rioxarray
* Machine Learning: CatBoost

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Contact

For questions or collaborations, please contact: \[Your Name or GitHub username]
