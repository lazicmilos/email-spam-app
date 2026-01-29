from flask import Flask, jsonify, render_template, request
import pickle
import os
import re  # Added for preprocessing

app = Flask(__name__)

# --- PREPROCESSING FUNCTION (Must match training script) ---
def preprocess_text(text):
    """
    Apply the same cleaning steps used during model training.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase, keep letters, numbers and special spam triggers like £$€!?
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s£$€!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the NEW pre-trained model with error handling
try:
    if not os.path.exists('model.pkl'):
        raise FileNotFoundError("model.pkl not found")
    if not os.path.exists('vectorizer.pkl'):
        raise FileNotFoundError("vectorizer.pkl not found")
    
    # Open with context manager (best practice)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    print("✓ Improved Model and vectorizer loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None
    vectorizer = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Invalid request data'}), 400
        
        raw_text = data['text']
        
        if not raw_text or not raw_text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # 1. APPLY PREPROCESSING (Crucial step for the improved model)
        processed_text = preprocess_text(raw_text)
        
        # 2. VECTORIZE (Using the loaded improved vectorizer)
        text_vec = vectorizer.transform([processed_text])
        
        # 3. PREDICT PROBABILITY
        # We want the probability of being SPAM (class 1)
        # model.predict_proba returns [[prob_ham, prob_spam]]
        proba_spam = model.predict_proba(text_vec)[0, 1]
        
        # 4. DECIDE RESULT
        # You can adjust the threshold here (e.g., 0.5)
        prediction = 1 if proba_spam > 0.5 else 0
        
        result = 'Spam' if prediction == 1 else 'Not Spam'
        confidence = f"{proba_spam*100:.2f}%"
        
        print(f"Input: {raw_text[:30]}... -> Prediction: {result} ({confidence})")
        
        return jsonify({
            'result': result, 
            'confidence': confidence,
            'is_spam': bool(prediction)
        })
    
    except Exception as e:
        print(f"ERROR in /predict: {e}")
        return jsonify({'error': 'Internal server error processing text'}), 500


if __name__ == '__main__':
    app.run(debug=True)