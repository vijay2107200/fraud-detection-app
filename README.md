# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using XGBoost, with SHAP explainability.

## Live Demo
[Click here to try the app](https://your-streamlit-url.streamlit.app)

## Dataset
[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 transactions | 492 fraud cases (0.17%)

## Models Trained
| Model | ROC-AUC | Recall |
|---|---|---|
| Logistic Regression | 0.971 | 89.1% |
| Random Forest | 0.989 | 94.8% |
| XGBoost (best) | 0.993 | 95.6% |

## Tech Stack
Python, scikit-learn, XGBoost, SHAP, Streamlit, pandas, SMOTE

## Project Structure
- `fraud_detection.ipynb` — full ML pipeline (EDA → preprocessing → models → SHAP)
- `app.py` — Streamlit web app
- `xgb_model.pkl` — saved best model