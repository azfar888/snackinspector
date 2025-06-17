import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import os

# Setup Flask
app = Flask(__name__)
CORS(app)

# Firebase setup
if not firebase_admin._apps:
   cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://snackinspector-default-rtdb.asia-southeast1.firebasedatabase.app'
})
# 🔧 Allow time skew (JWT workaround)
from google.auth.transport.requests import Request
from firebase_admin import _DEFAULT_APP_NAME
firebase_admin.get_app(_DEFAULT_APP_NAME)._credential._clock_skew = 300
ROBOFLOW_API_KEY = "iSFhDbkkI8CDPGS14ib2"
ROBOFLOW_MODEL_ID = "snackinspector/2"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        filepath = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(filepath)

        with open(filepath, 'rb') as image_file:
            response = requests.post(
                f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}?api_key={ROBOFLOW_API_KEY}",
                files={"file": image_file},
                data={"confidence": "0.5", "overlap": "0.3"}
            )

        result = response.json()

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

# Render entrypoint
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

