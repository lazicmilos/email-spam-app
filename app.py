import os
import pickle

from flask import Flask, jsonify, render_template, request

# The pickled pipeline references preprocessing.preprocess_text, so this
# module must be importable when the pipeline is loaded.
import preprocessing  # noqa: F401

MODEL_PATH = 'spam_pipeline.pkl'
SPAM_THRESHOLD = 0.5

app = Flask(__name__)


def load_pipeline(path):
    if not os.path.exists(path):
        print(f"✗ {path} not found. Run `python train_model_improve.py` first.")
        return None
    try:
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
    except Exception as e:
        print(f"✗ Error loading {path}: {e}")
        return None
    print(f"✓ Loaded {path}")
    return pipeline


pipeline = load_pipeline(MODEL_PATH)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    data = request.get_json(silent=True)
    if not data or not isinstance(data.get('text'), str):
        return jsonify({'error': 'Request body must be JSON with a "text" string'}), 400

    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Text cannot be empty'}), 400

    try:
        # The pipeline preprocesses, vectorizes and classifies in one step.
        # predict_proba returns [[P(ham), P(spam)]].
        spam_probability = float(pipeline.predict_proba([text])[0, 1])
    except Exception as e:
        print(f"ERROR in /predict: {e}")
        return jsonify({'error': 'Internal server error processing text'}), 500

    is_spam = spam_probability > SPAM_THRESHOLD
    return jsonify({
        'result': 'Spam' if is_spam else 'Not Spam',
        'is_spam': is_spam,
        'spam_probability': round(spam_probability, 4),
    })


if __name__ == '__main__':
    app.run(debug=True)
