import os
import json
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Flask
app = Flask(__name__)
CORS(app)

# Decode service account key from base64
b64_key = os.environ.get("GOOGLE_CREDENTIALS_B64")
key_data = json.loads(base64.b64decode(b64_key).decode())

# Firebase Realtime DB
FIREBASE_DB_URL = "https://snackinspector-default-rtdb.asia-southeast1.firebasedatabase.app"

# Create Firebase access token
SCOPES = ["https://www.googleapis.com/auth/firebase.database", "https://www.googleapis.com/auth/userinfo.email"]
credentials = service_account.Credentials.from_service_account_info(key_data, scopes=SCOPES)
credentials.refresh(Request())  # get token
access_token = credentials.token

# Roboflow
ROBOFLOW_MODEL_ID = "snackinspector/2"
ROBOFLOW_API_KEY = "iSFhDbkkI8CDPGS14ib2"

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

        # 🔁 Get ingredients from Firebase Realtime DB (using access token)
        headers = {"Authorization": f"Bearer {access_token}"}
        fb_res = requests.get(f"{FIREBASE_DB_URL}/ingredients/{label}.json", headers=headers)
        data = fb_res.json() if fb_res.ok else {}

        packaging = data.get("IngredientPackaging", "Not available")
        common = data.get("IngredientFriendly", "Not available")

        return jsonify({
            'prediction': label,
            'confidence': confidence,
            'ingredients_packaging': packaging,
            'ingredients_common': common
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({'error': str(e)}), 500

# Run on Render
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
