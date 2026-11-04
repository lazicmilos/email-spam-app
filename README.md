# Email Spam Classifier

A Flask web app that classifies a message as **Spam** or **Not Spam** and returns the model's spam probability. It uses a Multinomial Naive Bayes model on TF-IDF features (1–3 n-grams), trained on the SMS Spam Collection dataset.

## Features

- Real-time classification through a JSON API (`POST /predict`)
- Spam probability returned with each prediction
- Web UI with dark mode and animations
- Training script with stratified train/test split, 5-fold cross-validation, a classification report, and a confusion matrix
- Inspection of the top spam-indicating n-grams learned by the model

## Tech Stack

| Layer    | Tools                                       |
| -------- | ------------------------------------------- |
| Backend  | Flask                                       |
| ML       | scikit-learn, pandas, NumPy                 |
| Plotting | Matplotlib, seaborn (training script only)  |
| Frontend | HTML, CSS, JavaScript (`templates/index.html`) |

## How It Works

1. **Preprocessing:** text is lowercased, and everything except letters, digits, whitespace and `£ $ € ! ?` is stripped. Currency signs and exclamation marks are kept on purpose because they are strong spam signals.
2. **Vectorization:** `TfidfVectorizer` with up to 5000 features, n-grams 1–3, sublinear TF, English stop words removed, and `max_df=0.85`.
3. **Model:** `MultinomialNB(alpha=0.1)`.
4. **Decision:** a message is labeled spam when `P(spam) > 0.5`. The threshold is set in `app.py`.

The same `preprocess_text` function is used in `train_model_improve.py` and `app.py`. If you change it in one file, change it in the other too, or the app will see different input than the model was trained on.

## Project Structure

```
├── app.py                  # Flask app: serves the UI and the /predict endpoint
├── train_model_improve.py  # Main training script (TF-IDF + NB); saves model.pkl and vectorizer.pkl
├── email_spam.py           # Baseline model (CountVectorizer + NB), kept for comparison
├── check.py                # Dataset inspection: class balance and sample messages
├── test.py                 # Checks that model.pkl and vectorizer.pkl load correctly
├── templates/
│   └── index.html          # Frontend
└── requirements.txt
```

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/lazicmilos/email-spam-app.git
cd email-spam-app
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Download the dataset**

The dataset is not committed to the repo. Download the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle and put `spam.csv` in the project root. The scripts expect the Kaggle format, with columns `v1` (label) and `v2` (text).

**4. Train the model**

```bash
python train_model_improve.py
```

This prints the evaluation metrics and writes `model.pkl` and `vectorizer.pkl`.

**5. Run the app**

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## API

`POST /predict`

Request:

```json
{ "text": "WINNER! You've won a £1000 prize. Click here to claim!" }
```

Response:

```json
{ "result": "Spam", "confidence": "98.73%", "is_spam": true }
```

`confidence` is the model's probability that the message is spam, not its confidence in whatever label it picked. The value above is only an example.

Errors: `400` if the body is missing or the text is empty, and `500` if the model files are not loaded.

## Evaluation

The dataset is imbalanced: about 87% of the messages are ham. A model that always answers "ham" already gets about 87% accuracy, so accuracy alone says little. The metrics that matter are **precision and recall on the spam class**, measured on a held-out test set (stratified 80/20 split, `random_state=42`).

<!-- Fill in with the output of `python train_model_improve.py` -->

| Metric (spam class, test set) | Value |
| ----------------------------- | ----- |
| Precision                     | —     |
| Recall                        | —     |
| F1                            | —     |
| Mean 5-fold CV F1 (train)     | —     |

- **Precision:** of the messages flagged as spam, how many really were spam. Low precision means real messages get blocked.
- **Recall:** of all the real spam, how much the model caught.

## Limitations

- **Domain mismatch:** the model is trained on SMS messages, not email. Email-specific signals such as headers, HTML, links and sender reputation are not used.
- **Synthetic data:** about 80 hand-written spam phrases are added to the training data. They are added before the train/test split, so some of them end up in the test set. That makes test metrics slightly optimistic for this kind of phrase.
- **Fixed threshold:** the 0.5 threshold was not tuned. In production you would choose it based on how costly a wrongly blocked real message is.

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) by Almeida & Hidalgo (UCI Machine Learning Repository): 5,574 SMS messages labeled ham or spam.
