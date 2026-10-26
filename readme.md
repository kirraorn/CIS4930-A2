# Kepler Exoplanet Visual Analytics Dashboard

An interactive multi-view visual analytics dashboard built with Python FastAPI and D3.js v7 for exploring exoplanet transit light curves from NASA's Kepler Space Telescope dataset.

## System Architecture
- **Backend:** Python FastAPI serving precomputed summary metrics and downsampled flux time series over JSON endpoints.
- **Frontend:** Single-page dashboard built with D3.js v7, HTML5, and Tailwind CSS.

## Setup & Running Instructions

### 1. Prerequisites
Ensure you have Python 3.9+ installed and your dataset placed in `backend/data/Exoplanet_Train.csv` (or `backend/Exoplanet_Train.csv`).

### 2. Backend Setup
Navigate to the `backend` directory, install dependencies, and start the FastAPI uvicorn server:
```bash
cd backend
pip install fastapi uvicorn pandas numpy
uvicorn main:app --reload --port 8000