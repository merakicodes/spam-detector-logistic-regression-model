# SMS Spam Checker

A machine learning app that classifies SMS messages as **Spam** or **Not spam** using TF-IDF and logistic regression.

Built in Google Colab and deployed with Streamlit Community Cloud.

## Live Demo

[Try the SMS Spam Checker](https://enqhfc7o6vgef5fg7cgxub.streamlit.app/)

Paste a message, choose a decision threshold, and click **Check message** to see its estimated spam probability and predicted label.

## How It Works

1. **TF-IDF** converts the message into numerical word features.
2. **Logistic regression** estimates the probability of spam.
3. A **decision threshold** converts that probability into a label:
   - Probability ≥ threshold → **Spam**
   - Probability < threshold → **Not spam**

The default threshold is **0.50**. Changing it changes the classification rule, while the model’s probability estimate stays the same.

The app reuses the trained TF-IDF converter and model without retraining.

## Dataset

This project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

The downloaded CSV contains **5,572 labelled messages**:

| Label | Meaning | Messages | Percentage |
|---|---|---:|---:|
| `ham` | Not spam | 4,825 | 86.59% |
| `spam` | Spam | 747 | 13.41% |

For modelling, the labels are encoded as **0 = Not spam** and **1 = Spam**. These counts describe the original CSV before any cleaning.

## Training and Evaluation

Messages are divided into training and test sets using a stratified split to preserve similar class proportions. A fixed random seed makes the split reproducible.

The TF-IDF converter learns its vocabulary and word weights from the **training messages only**. The test messages are transformed using that same converter.

Model evaluation includes:

- **Accuracy:** overall proportion of correct predictions.
- **Precision:** proportion of predicted spam messages that are actually spam.
- **Recall:** proportion of actual spam messages detected.
- **Confusion matrix:** counts of correct predictions, false alarms, and missed spam.

Because the dataset contains mostly legitimate messages, accuracy alone is insufficient. Predicting every message as “Not spam” would achieve **86.59% accuracy** while detecting no spam.

## Example App Tests

At a threshold of **0.50**, the deployed app produced:

| Message | Estimated spam probability | Prediction |
|---|---:|---|
| “Hey, are we still meeting for lunch tomorrow? I can be there around 12.” | 2.4% | Not spam |
| “WINNER! Claim your FREE £1000 cash prize now. Call 09012345678 urgently. Text STOP to unsubscribe.” | 98.3% | Spam |

These examples demonstrate app behavior. They are not a substitute for evaluation across the full test set.

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and prediction logic |
| `spam_model.joblib` | Saved TF-IDF converter and logistic regression model |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## Run Locally

Use **Python 3.13**, matching the deployed environment.

Download or clone the repository, open a terminal in its folder, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local browser address displayed in the terminal.

## Deployment

The app runs on Streamlit Community Cloud with:

- **Branch:** `main`
- **Main file:** `app.py`
- **Python version:** 3.13

Keep `app.py`, `requirements.txt`, and `spam_model.joblib` together in the repository’s root directory.

## Limitations

- Predictions can be wrong.
- Estimated probabilities are model outputs, not guarantees.
- Performance on newer spam styles or other languages has not been established.
- Lower thresholds flag more messages as spam, potentially creating more false alarms.
- Higher thresholds flag fewer messages as spam, potentially missing more spam.

## Tools Used

Python · pandas · scikit-learn · joblib · Google Colab · Streamlit · GitHub

## Acknowledgements

Dataset: *SMS Spam Collection* by Tiago Almeida and José María Gómez Hidalgo, available through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and Kaggle.
