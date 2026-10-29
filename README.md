# Iris Flower Classification — Python + Machine Learning

A beginner-friendly machine-learning project that predicts an Iris flower species from four measurements. It uses scikit-learn's built-in Iris dataset, Logistic Regression, feature scaling, a train/test split, and a Streamlit interface.

## Features
- Trains a classification model using a train/test split.
- Scales numeric features with `StandardScaler`.
- Reports test accuracy, precision/recall/F1, and a confusion matrix.
- Saves the trained model and scaler with Joblib.
- Provides a simple web UI for predictions and estimated class probabilities.

## Requirements
- Python 3.10 or newer recommended.

## Run locally

1. Open a terminal in this folder.
2. Create and activate a virtual environment (recommended):

   Windows:
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:
   ```bash
   python train_model.py
   ```

5. Launch the web app:
   ```bash
   streamlit run app.py
   ```

The browser will open the app. If it does not, use the local URL printed in the terminal.


