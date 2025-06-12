# Air Quality Prediction System

This project is a web-based application that predicts air quality parameters using machine learning. It provides predictions for various air pollutants including CO, NO, NO2, O3, SO2, PM2.5, PM10, and NH3.

## Features

- Real-time air quality prediction
- Multi-step forecasting (24 steps ahead)
- Web interface for easy input of current air quality parameters
- Support for multiple air pollutants:
  - Carbon Monoxide (CO)
  - Nitric Oxide (NO)
  - Nitrogen Dioxide (NO2)
  - Ozone (O3)
  - Sulfur Dioxide (SO2)
  - Particulate Matter 2.5 (PM2.5)
  - Particulate Matter 10 (PM10)
  - Ammonia (NH3)

## Prerequisites

- Python 3.x
- Flask
- Pandas
- scikit-learn
- joblib

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd air-quality-prediction
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Ensure you have the following files in your project directory:
   - `pipeline.joblib` (preprocessing pipeline)
   - `rf.joblib` (trained random forest model)
   - `FINAL.csv` (historical air quality data)

## Project Structure

```
air-quality-prediction/
├── main.py              # Main Flask application
├── pipeline.joblib      # Preprocessing pipeline
├── rf.joblib           # Trained model
├── FINAL.csv           # Historical data
├── templates/          # HTML templates
│   ├── index.html
│   ├── about.html
│   ├── air.html
│   └── result.html
└── static/            # Static files (CSS, JS, images)
```

## Usage

1. Start the Flask application:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter the current air quality parameters in the web interface.

4. The system will provide predictions for the next 24 steps for each pollutant.

## How It Works

1. The application uses a preprocessing pipeline to normalize the input data
2. A Random Forest model makes predictions for multiple steps ahead
3. The predictions are transformed back to their original scale
4. Results are displayed in a user-friendly format

## Model Details

- The system uses a Random Forest model for predictions
- Features include lagged values of all pollutants (24 lags)
- Predictions are made for 24 steps ahead
- Data is preprocessed using a standard scaler

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
