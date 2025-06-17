from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import firebase_admin
from firebase_admin import credentials, db
import os

# Setup Flask
app = Flask(__name__)
CORS(app)

# Roboflow setup
ROBOFLOW_MODEL_ID = "snackinspector/2"  # Change if needed
ROBOFLOW_API_KEY = "iSFhDbkkI8CDPGS14ib2"
client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY)

# Firebase setup
if not firebase_admin._apps:
    cred = credentials.Certificate("snackinspector-firebase-adminsdk-fbsvc-2ea6eaee79.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://snackinspector-default-rtdb.asia-southeast1.firebasedatabase.app'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(path)

        result = client.infer(path, model_id=ROBOFLOW_MODEL_ID)

        if 'predictions' in result and len(result['predictions']) > 0:
            raw_label = result['predictions'][0]['class']
            label = raw_label.upper().replace(" ", "_")
            confidence = result['predictions'][0]['confidence']
        else:
            label = "UNKNOWN"
            confidence = 0

        ref = db.reference(f'ingredients/{label}')
        data = ref.get()

        packaging = data.get('IngredientPackaging') if data else "Not available"
        common = data.get('IngredientFriendly') if data else "Not available"

        return jsonify({
            'prediction': label,
            'confidence': confidence,
            'ingredients_packaging': packaging,
            'ingredients_common': common
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({'error': str(e)}), 500

# Tell Render how to start the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
