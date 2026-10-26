import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

app = FastAPI(
    title="Kepler Exoplanet Visual Analytics API",
    description="Backend API serving pre-aggregated Kepler light-curve features and time-series endpoints.",
    version="5.0.0"
)

# Enable CORS for localhost and external deployed frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join("data", "Exoplanet_Train.csv")
raw_df: Optional[pd.DataFrame] = None
summary_df: Optional[pd.DataFrame] = None


@app.on_event("startup")
def load_and_process_data():
    """Load Kepler CSV dataset and precompute robust visual metrics."""
    global raw_df, summary_df
    file_to_load = DATA_PATH if os.path.exists(DATA_PATH) else "Exoplanet_Train.csv"
    
    if not os.path.exists(file_to_load):
        print(f"Warning: {file_to_load} not found. Please ensure dataset is present.")
        return

    print(f"Loading Kepler dataset from {file_to_load}...")
    raw_df = pd.read_csv(file_to_load)

    flux_cols = [c for c in raw_df.columns if c.startswith("FLUX.")]
    labels = raw_df["LABEL"].astype(int).values
    flux_matrix = raw_df[flux_cols].values

    # Feature Engineering
    means = np.mean(flux_matrix, axis=1)
    stds = np.std(flux_matrix, axis=1)
    mins = np.min(flux_matrix, axis=1)

    # Avoid zero division
    stds_safe = np.where(stds == 0, 1.0, stds)

    # Calculate normalized features
    noise_level = stds
    dip_scores = (means - mins) / stds_safe

    # Clip 99th percentile extreme outliers for visualization stability
    noise_cutoff = np.percentile(noise_level, 99)
    dip_cutoff = np.percentile(dip_scores, 99)

    summary_df = pd.DataFrame({
        "star_id": np.arange(len(raw_df)),
        "label": labels,
        "is_exoplanet": (labels == 2).astype(bool),
        "mean_flux": np.round(means, 2),
        "noise_level": np.round(np.clip(noise_level, 0, noise_cutoff), 2),
        "transit_depth_score": np.round(np.clip(dip_scores, 0, dip_cutoff), 2),
    })
    print(f"Successfully preprocessed {len(summary_df)} stars ({sum(summary_df['is_exoplanet'])} confirmed exoplanets).")


@app.get("/")
def root():
    return {
        "status": "online",
        "dataset": "Kepler Exoplanet Time-Series",
        "total_stars": len(summary_df) if summary_df is not None else 0,
        "endpoints": ["/api/stars/summary", "/api/stars/{star_id}"]
    }


@app.get("/api/stars/summary")
def get_stars_summary(
    label: Optional[int] = Query(None, description="Filter by class: 2 for Exoplanet, 1 for Standard Star"),
    limit: Optional[int] = Query(None, description="Limit max returned records")
):
    """Serves lightweight summary array for fast rendering across multi-view dashboard."""
    if summary_df is None:
        raise HTTPException(status_code=500, detail="Data not initialized on server.")
    
    filtered = summary_df
    if label is not None:
        filtered = filtered[filtered["label"] == label]
    if limit is not None:
        filtered = filtered.iloc[:limit]

    return {
        "total_count": len(filtered),
        "exoplanet_count": int(filtered["is_exoplanet"].sum()),
        "non_exoplanet_count": int((~filtered["is_exoplanet"]).sum()),
        "stars": filtered.to_dict(orient="records")
    }


@app.get("/api/stars/{star_id}")
def get_star_details(star_id: int, downsample: Optional[int] = Query(4, description="Downsampling step size for fast rendering")):
    """Serves details-on-demand time-series flux array for a single selected star."""
    if raw_df is None or summary_df is None:
        raise HTTPException(status_code=500, detail="Data not initialized on server.")
    
    if star_id < 0 or star_id >= len(raw_df):
        raise HTTPException(status_code=404, detail=f"Star ID {star_id} does not exist.")

    flux_cols = [c for c in raw_df.columns if c.startswith("FLUX.")]
    star_summary = summary_df.iloc[star_id].to_dict()
    flux_values = raw_df.iloc[star_id][flux_cols].values.astype(float)

    # Downsample time series for ultra-fast rendering over network
    if downsample > 1:
        time_steps = np.arange(1, len(flux_values) + 1)[::downsample]
        flux_values = flux_values[::downsample]
    else:
        time_steps = np.arange(1, len(flux_values) + 1)

    time_series = [{"time": int(t), "flux": float(f)} for t, f in zip(time_steps, flux_values)]
    return {"metadata": star_summary, "time_series": time_series}