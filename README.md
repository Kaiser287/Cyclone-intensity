# Cyclone Intensity & Track

Estimate tropical cyclone intensity from a single infrared satellite image and forecast the storm's track 12–48 hours ahead, served through an interactive Streamlit app.



| Task | Model | Input | Output |
|---|---|---|---|
| Intensity estimation | ResNet18 (CNN, 1-channel) | 128×128 IR satellite image | Max sustained wind (kt) + Saffir–Simpson category |
| Track forecast | LightGBM ensemble (8 regressors) | Last 3 positions (−48h, −24h, now) + current intensity | Positions at +12/24/36/48h with uncertainty cone |

## Results

**Intensity model** (held-out test set, with test-time augmentation):

| Metric | Value |
|---|---|
| RMSE | 11.87 kt |
| MAE | 8.44 kt |

Full numbers are in `source/models/intensity_metrics.json`, per-sample predictions in `source/models/test_predictions.csv`, and the training curve in `source/models/intensity_history.csv`. Track model metrics are in `source/models/track_metrics.json`.

## How it works

### 1. Intensity estimation (deep learning)

- **Data:** [The Cyclone Image Dataset](https://www.kaggle.com/datasets/vaukaofworlds/thecycloneimagedataset) (Kaggle, v3): 21,076 satellite images of 128×128 px with 4 channels, labelled with basin, storm ID, position, time, Vmax, R35 and MSLP.
- **Input:** only channel 0 (infrared). Images use a 0–255 scale with *dark = cold cloud tops* polarity, and are normalised with dataset statistics stored inside the checkpoint.
- **Model:** ResNet18 adapted to a single input channel, regressing standardised Vmax.
- **Inference:** optional test-time augmentation (TTA) averages predictions over transformed copies of the image.
- **Checkpoint:** `intensity_best.pt`, stored in FP16 (22 MB) so it fits in the repo without Git LFS.

The predicted wind speed is mapped to a category:

| Category | Wind (kt) |
|---|---|
| Tropical Depression (TD) | < 34 |
| Tropical Storm (TS) | 34–63 |
| Category 1 | 64–82 |
| Category 2 | 83–95 |
| Category 3 | 96–112 |
| Category 4 | 113–136 |
| Category 5 | ≥ 137 |

### 2. Track forecast (gradient boosting)

- **Features (12):** current position, 24h and 48h displacement, acceleration, translation speed, heading (sin/cos) and current intensity.
- **Model:** one LightGBM regressor per (horizon × direction): Δlat and Δlon at 12, 24, 36 and 48 hours, giving 8 models in total.
- **Uncertainty:** each forecast point carries a cone radius (km) that widens with lead time.
- **Fallback:** if the model file is missing, the predictor falls back to a persistence (constant velocity) forecast, so the app never crashes.

## Project structure

Cyclone-intensity/
├── app.py                         # Streamlit app
├── requirements.txt
└── source/
    ├── inference/
    │   └── predictor.py           # IntensityPredictor: preprocessing, TTA, track integration
    └── models/
        ├── intensity_model.py     # ResNet18 definition
        ├── track_model.py         # TrackPredictor (LightGBM + persistence fallback)
        ├── intensity_best.pt      # trained CNN weights (FP16)
        ├── intensity_stats.json   # normalisation statistics
        ├── track_lgbm.pkl         # trained LightGBM ensemble
        └── *_metrics.json, *.csv  # evaluation results

## Run locally

git clone https://github.com/Kaiser287/Cyclone-intensity.git
cd Cyclone-intensity
pip install -r requirements.txt
streamlit run app.py

Then open http://localhost:8501.

## Use from Python

from source.inference.predictor import IntensityPredictor

p = IntensityPredictor("source/models/intensity_best.pt")

# image: 128x128 (or 128x128x4, channel 0 is selected automatically)
bt, approx = p.to_bt(image)
res = p.predict(bt, tta=True)
print(res["wind_speed"], res["category"])

# track: positions at -48h, -24h and now as (lat, lon)
history = [(15.0, 130.0), (16.0, 128.5), (17.2, 127.0)]
track = p.track.predict(history, res["wind_speed"])
for pt in track["points"]:
    print(pt)   # {'hour': 12, 'lat': ..., 'lon': ..., 'cone_km': ...}

## Limitations

- Intensity is estimated from a **single image**. The model sees no temporal evolution, which operational methods such as the Dvorak technique rely on.
- The track model only uses the storm's own recent motion and intensity, with no steering-flow or environmental data, so errors grow quickly beyond 24h.
- Trained on one dataset. Images from other satellites or with different calibration may need re-normalisation.
- This is a portfolio/research project and **not** a substitute for official forecasts.

## Tech stack

PyTorch · LightGBM · scikit-learn · NumPy · pandas · h5py · Streamlit · trained on Google Colab (T4 GPU)
