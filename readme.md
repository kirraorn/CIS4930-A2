# Kepler Exoplanet Visual Analytics Dashboard

An interactive multi-view visual analytics dashboard built with Python FastAPI and D3.js v7 for exploring exoplanet transit light curves from NASA's Kepler Space Telescope dataset.

## System Architecture
- **Backend:** Python FastAPI (`backend/main.py`) serving precomputed summary metrics, normalized noise scores, and downsampled time-series JSON data over HTTP.
- **Frontend:** Single-page web client (`frontend/index.html`) built with D3.js v7, HTML5, and Tailwind CSS, featuring three coordinated views and smooth transitions.

## Repository Structure
```text
CIS4930-A2/
├── backend/
│   ├── main.py
│   └── data/
│       └── Exoplanet_Train.csv
└── frontend/
    └── index.html
```

### 2. Backend Setup
Navigate to the `backend` directory, install dependencies, and start the FastAPI uvicorn server:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate            # WINDOWS: venv\Scripts\activate
pip install fastapi uvicorn pandas numpy
uvicorn main:app --reload --port 8000
```

The API will start at http://localhost:8000

### 2. Frontend Setup
```bash
cd frontend
python3 -m http.server 5173
```

Open your browser and navigate to http://localhost:5173
