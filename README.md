# Email Spam Classifier

AI-powered spam detection using Naive Bayes classification with 97% accuracy.

## Features
- Real-time email classification
- Modern, animated UI with dark mode
- TF-IDF vectorization with n-grams
- High accuracy spam detection

## Tech Stack
- **Backend:** Flask, scikit-learn, Pandas
- **Frontend:** HTML, CSS, JavaScript
- **Model:** Multinomial Naive Bayes with TF-IDF

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Train the model
```bash
python train_model.py
```

4. Run the application
```bash
python app.py
```

5. Open `index.html` in your browser

## Model Performance
- Training Accuracy: 97%
- Uses TF-IDF with 5000 features
- N-gram range: 1-3

## Dataset
Uses the SMS Spam Collection dataset.