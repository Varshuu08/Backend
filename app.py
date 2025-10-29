# ==============================================
# AI Health Assistant Backend (Flask + Supabase)
# ==============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ----------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
print("🔍 SUPABASE_URL:", os.getenv("SUPABASE_URL"))
print("🔍 SUPABASE_KEY:", "Loaded" if os.getenv("SUPABASE_KEY") else "Missing")
print("🔍 HF_TOKEN:", "Loaded" if os.getenv("HF_TOKEN") else "Missing")

# Initialize Supabase and Hugging Face clients

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
hf_client = InferenceClient(token=HF_TOKEN)

# ----------------------------------------------
# 2️⃣ Initialize Flask
# ----------------------------------------------
app = Flask(__name__)
CORS(app)
diabetes_model = joblib.load("model/diabetes_model.pkl")
symptom_model = joblib.load("model/symptom_model.pkl")
scaler_model = joblib.load("model/scaler.pkl")

# ----------------------------------------------
# 4️⃣ Ayurvedic / Homeopathy Recommendation Logic
# ----------------------------------------------
herbal_meds = {
    "Diabetes": [
        "Bitter Gourd (Karela) Juice - regulates blood sugar",
        "Fenugreek Seeds - improves glucose tolerance",
        "Neem Leaves Extract - natural insulin booster"
    ],
    "Fever": [
        "Tulsi & Ginger Decoction - boosts immunity",
        "Giloy (Guduchi) - natural antipyretic",
        "Homeopathic: Belladonna 30C for acute fever"
    ],
    "Cold": [
        "Tulsi & Honey - relieves congestion",
        "Homeopathic: Aconite Napellus 30C for early cold"
    ],
    "Cough": [
        "Mulethi (Licorice) - soothes throat",
        "Homeopathic: Drosera 30C for dry cough"
    ],
    "Heart Disease": [
        "Arjuna Bark Extract - strengthens cardiac muscles",
        "Homeopathic: Crataegus Q for heart tone"
    ]
}

# ----------------------------------------------
# 5️⃣ Predict Diabetes Risk
# ----------------------------------------------
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        features = np.array([list(data.values())]).astype(float)
        prediction = diabetes_model.predict(features)[0]
        result = "Positive" if prediction == 1 else "Negative"

        # Store consultation in Supabase
        supabase.table("consultations").insert({
            "type": "diabetes",
            "input_data": data,
            "result": result
        }).execute()

        return jsonify({
            "prediction": result,
            "ayurvedic_tips": herbal_meds.get("Diabetes", [])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------------------------------------
# 6️⃣ Predict Disease from Symptoms
# ----------------------------------------------
@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        data = request.json
        symptoms = data.get("symptoms", [])
        if not symptoms:
            return jsonify({"error": "Please provide symptoms"})

        # Convert symptoms to dataframe for model
        df = pd.DataFrame([symptoms])
        prediction = symptom_model.predict(df)[0]

        ayurvedic_remedies = herbal_meds.get(prediction, ["No herbal remedies found"])

        # Optionally get explanation from Hugging Face
        explanation = hf_client.text_generation(
            f"Suggest simple home remedies for {prediction} in 3 short lines.",
            model="distilgpt2"
        )

        # Store consultation
        supabase.table("consultations").insert({
            "type": "disease",
            "input_data": symptoms,
            "result": prediction
        }).execute()

        return jsonify({
            "predicted_disease": prediction,
            "ayurvedic_or_homeopathy": ayurvedic_remedies,
            "ai_explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------------------------------------
# 7️⃣ Health History Endpoint
# ----------------------------------------------
@app.route('/consultations', methods=['GET'])
def get_consultations():
    try:
        response = supabase.table("consultations").select("*").execute()
        return jsonify(response.data)
    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------------------------------------
# 8️⃣ Run the App
# ----------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

