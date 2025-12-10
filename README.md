# House Price Prediction

A simple, lightweight web app that predicts house prices using machine learning. This project includes a training notebook, a saved model, and a Bootstrap 5 front-end to collect property features and call a prediction API.

**Built with:** Python, scikit-learn, Flask (recommended), Bootstrap 5

**Key Features:**
- **Predictive model**: Trained regressors (Linear Regression, Random Forest, SVR, AdaBoost) evaluated in `Houseprice.ipynb`.
- **Responsive UI**: Modern Bootstrap 5 landing page and prediction form at `templates/index.html`.
- **Simple API**: Post JSON to `/predict` to receive a numeric prediction.
- **Pickle model**: Example of saving/loading a model with `pickle` (see cells that write/read `house_price_regression_dataset.pkl`).

**Project Structure (important files)**
- `Houseprice.ipynb` : Jupyter notebook with data exploration, model training, evaluation, and sample prediction helper.
- `house_price_regression_dataset.csv` : Dataset used for training.
- `templates/index.html` : Bootstrap 5 UI for the web front-end.
- `house_price_regression_dataset.pkl` : (Optional) Pickled model produced by the notebook.

**Requirements**
- Python 3.8+
- Required packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `flask` (if serving the site), `pickle` (standard lib)

Install with pip:
```
python -m pip install -r requirements.txt
```
If you don't have a `requirements.txt`, install the main packages:
```
python -m pip install numpy pandas scikit-learn matplotlib flask
```

**Quick start (Flask example)**
1. Ensure the pickled model `house_price_regression_dataset.pkl` exists (or train the model in `Houseprice.ipynb`).
2. Create a minimal Flask app (example):

```python
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('house_price_regression_dataset.pkl','rb') as f:
	model = pickle.load(f)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	data = request.get_json()
	features = [
		data['Square_Footage'],
		data['Num_Bedrooms'],
		data['Num_Bathrooms'],
		data['Year_Built'],
		data['Lot_Size'],
		data['Garage_Size'],
		data['Neighborhood_Quality']
	]
	arr = np.array([features])
	pred = model.predict(arr)[0]
	return jsonify({'prediction': float(pred)})

if __name__ == '__main__':
	app.run(debug=True)
```

Open `http://127.0.0.1:5000/` and use the form on the landing page to send requests to `/predict`.

**Notes & Tips**
- The notebook contains cells that evaluate multiple models (LR, DT, RF, SVR, AdaBoost). Choose the model you prefer and pickle it for serving.
- Make sure the input order and feature preprocessing used by the served model matches what the front-end sends. If you used scaling/encoding during training, apply the same transforms at inference time.
- If you plan to deploy, consider storing the model in a more robust format (joblib) and adding input validation on the server.

**License & Contribution**
- This repository does not declare a license. Add one if you plan to share it publicly.
- Contributions: open an issue or submit a PR with improvements.

---
Generated README for local development and rapid testing. Update sections to match your final deployment and chosen model.

