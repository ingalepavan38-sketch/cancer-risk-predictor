from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model and label encoder
try:
    model = joblib.load('model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    label_encoder = None

# Feature names (based on the training data)
FEATURE_NAMES = [
    'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
    'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
    'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
    'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
    'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
    'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'
]

@app.route('/')
def home():
    return jsonify({
        "message": "Cancer Risk Prediction API",
        "status": "Running",
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/health": "GET - Health check"
        }
    })

@app.route('/health')
def health():
    model_status = "Loaded" if model is not None else "Not Loaded"
    return jsonify({
        "status": "healthy",
        "model_status": model_status
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or label_encoder is None:
            return jsonify({"error": "Models not loaded"}), 500
        
        # Get data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = []
        for feature_name in FEATURE_NAMES:
            if feature_name not in data:
                return jsonify({"error": f"Missing feature: {feature_name}"}), 400
            features.append(data[feature_name])
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction_encoded = model.predict(features_array)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Get prediction probability if available
        try:
            probabilities = model.predict_proba(features_array)[0]
            prob_dict = {}
            for idx, class_name in enumerate(label_encoder.classes_):
                prob_dict[class_name] = float(probabilities[idx])
        except:
            prob_dict = None
        
        response = {
            "prediction": prediction,
            "risk_level": prediction,
            "probabilities": prob_dict
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
