# Topologically-Constrained Multi-3DGS Navigator for Indoor Environments

This is the code repository for a MSc Geomatics thesis paper that proposes the use of IndoorGML topology to constrain and determine navigation between discrete 3DGS scenes of indoor environments.


## Prerequisites

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Code Language |
| [COLMAP](https://colmap.github.io/) | Sparse 3D reconstruction |
| [FFmpeg](https://ffmpeg.org/) | Video frame extraction |
| [Lichtfeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) | Free 3DGS reconstruction tool |
| PostgreSQL + PostGIS | Spatial topology database |

Note: 
- The base IndoorGML model that is ingested by indoorgml_to_postgis.py was manually created, using STEMLab's Ineditor (available at https://github.com/STEMLab/InEditor). 
- InViewer-Desktop (https://github.com/STEMLab/InViewer-Desktop) may also be used for visualising the IndoorGML files.
- A reference floor plan was used to ensure metric accuracy for alignment purposes. see `data/floor_plan_data.json`

## Installation

See requirements.txt.
COLMAP and FFmpeg folders should be installed to code/tools/.
Lichtfeld Studio can be separately installed and used to generate the 3DGS reconstruction.

## Usage

### 1. Input Video to Sparse Reconstruction
Run `video_to_sparse.py`.
It automates the pipeline: frame extraction (FFmpeg), feature extraction, feature matching, sparse SfM, undistortion. 
Edit the script's `VIDEO_PATH`, `OUTPUT_DIR`, and `COLMAP_PATH` variables before running.

### 2. 3DGS Reconstructions
The output `undistorted` folder is structured for ingestion by Lichtfeld Studio.
Parameters for the 3DGS generation can be tuned to user's preferences.

### 3. Create the IndoorGML model
Sample IndoorGML model is given in `data/`.
For manually created IndoorGMLs via InEditor, additionally run `coordinate_replacer.py` to create an aligned IndoorGML.
Ensure you have a valid `data/floor_plan_data.json` with point-pair correspondences between the COLMAP sparse output and a reference frame (eg: floor plan).

### 4. Parse IndoorGML to Database
Run `indoorgml_to_postgis.py` on the aligned IndoorGML.
This populates the PostGIS database with the necessary geometry and attributes to run the web viewer.
Edit the database parameters to suit your setup.

### 5. Alignment Calculation
Run `transformation.py`.
This reads the point-pair correspondences from `data/floor_plan_data.json` and calculates the transformation matrix per IndoorGML Cell.
The transformation matrix and axis permutation are then written into the PostGIS database.

Note: Re-running `indoorgml_to_postgis.py` after this step will wipe the transformation matrix from the database.

### 6. Web Server Prototype
Note: First ensure that the 3DGS ply files are downloaded from the Release (at https://github.com/mjteo95/3dgs-topology/releases/tag/PLY-files), and placed in `code/src/static/splats/`.

Run `server.py`.
Then open `http://localhost:8000` in a browser to launch the 3DGS viewer.

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves `viewer.html` |
| `/api/scene-data` | GET | Returns rooms, states, and transitions as JSON |
| `/static/splats/{id}.ply` | GET | Streams a pre-trained 3DGS PLY file |

## Data

| File | Description |
|------|-------------|
| `data/floor_plan_data.json` | Room definitions (C1–C4) and DXF-to-sparse correspondence points |
| `data/aligned-bk.gml` | IndoorGML model with aligned coordinates |
| `data/manual-bk.gml` | Hand-authored IndoorGML model (source) |
| `data/transformation_report.txt` | Alignment metrics |
| `data/0470_synthetic_pc.ply` | Synthetic reference point cloud for room 0470 |
| `code/src/static/splats/*.ply` | 3DGS point clouds (rooms 0450, 0470, 0807) |

## Rooms

Three rooms are included in the dataset:

| ID | PLY size |
|----|----------|
| 0450 | ~80 MB |
| 0470 | ~80 MB |
| 0807 | ~125 MB |

These are published as a Release (see https://github.com/mjteo95/3dgs-topology/releases/tag/PLY-files) due to file size constraints in the repository. Ensure these files are downloaded and placed in `code/src/static/splats/` in order for `server.py` to function.

## Project Structure

```
main/
├── code/
│   ├── src/
│   │   ├── server.py   
│   │   ├── transformation.py          
│   │   ├── coordinate_replacer.py     
│   │   ├── indoorgml_to_postgis.py   
│   │   ├── video_to_sparse.py
│   │   ├── viewer.html
│   │   └── static/splats/
│   └── tools/
│       ├── colmap/ 
│       ├── ffmpeg/
│       └── Lichtfeld Studio/
├── data/
├── requirements.txt
└── README.md
```
