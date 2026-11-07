# Bank Marketing - Term Deposit Prediction (ML Assignment 2)

## a. Problem statement

A Portuguese bank ran phone marketing campaigns to sell term deposits. The goal here is to predict, from a client's details and the campaign contact history, whether the client will subscribe to a term deposit (`y` = yes / no).

This is a binary classification problem. I trained 5 classification models on the same data, compared them using 6 metrics, and built a Streamlit app where the test data can be uploaded and each model's results can be checked.

## b. Dataset description

- **Source:** UCI Machine Learning Repository - Bank Marketing (https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **File used:** `bank.csv` (the smaller version of the dataset)
- **Rows:** 4,521
- **Features:** 16 (plus the target `y`)
- **Missing values:** none
- **Target:** `y` - did the client subscribe? `no` = 4,000, `yes` = 521 (about 11.5% yes, so the classes are imbalanced)

| Type | Columns |
|---|---|
| Numeric (7) | age, balance, day, duration, campaign, pdays, previous |
| Categorical (9) | job, marital, education, default, housing, loan, contact, month, poutcome |

**Preprocessing:** numeric columns are scaled with StandardScaler and categorical columns are one-hot encoded. Both steps are saved inside each model's pipeline, so the app can work directly with the raw CSV columns.

**Train/test split:** 80/20 stratified split (3,616 train rows, 905 test rows) with `random_state=42`. The 905 test rows are saved as `test_data.csv` and were never used for training.

## c. GitHub repository link

https://github.com/LaxmiNarayanaK/bank-term-deposit-classifier

**Live Streamlit app:** https://bank-term-deposit-classifier.streamlit.app/

Repository structure:

```
├── app.py               # Streamlit app
├── train_models.py      # trains the 5 models and saves them
├── requirements.txt
├── README.md
├── test_data.csv        # test split used for evaluation (upload this in the app)
├── data/bank.csv        # original UCI data
├── model/               # saved trained models (.joblib) + metrics.csv
└── .streamlit/config.toml
```

## d. Models used

All 5 models were trained on the same training split and evaluated on the same 905 test rows.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8928 | 0.8913 | 0.5636 | 0.2981 | 0.3899 | 0.3579 |
| Decision Tree | 0.8840 | 0.8085 | 0.4921 | 0.2981 | 0.3713 | 0.3235 |
| KNN | 0.8884 | 0.7997 | 0.5517 | 0.1538 | 0.2406 | 0.2492 |
| Naive Bayes | 0.8210 | 0.7881 | 0.3092 | 0.4519 | 0.3672 | 0.2737 |
| Support Vector Machine | 0.8939 | 0.8830 | 0.5909 | 0.2500 | 0.3514 | 0.3374 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Gave the best balance on this data. It has the highest AUC (0.891), F1 (0.390) and MCC (0.358). Accuracy is 0.893, but recall is only 0.298, so it still misses most of the clients who actually said yes. After one-hot encoding and scaling, a simple linear model seems to work well here. |
| Decision Tree | Fairly close to Logistic Regression on F1 (0.371) and MCC (0.324), with the same recall (0.298). AUC is clearly lower (0.809), because a tree with depth 6 gives fewer distinct probability values, so its ranking of clients is rougher. I limited the depth to stop it overfitting the training data. |
| KNN | The weakest model overall. Accuracy looks fine (0.888), but recall is only 0.154 and MCC is 0.249, so it finds very few of the yes clients. With so few yes cases, most of a client's nearest neighbours are no, so the model mostly predicts no. |
| Naive Bayes | Behaves differently from the other models. It has the best recall (0.452), so it catches the most subscribers, but the lowest precision (0.309) and accuracy (0.821) because it predicts yes too often. It assumes the features are independent, which isn't really true for columns like duration, pdays and previous. |
| Support Vector Machine | Has the highest accuracy (0.894) and precision (0.591), so when it predicts yes it's usually right. But recall is just 0.25, so it plays it safe and misses many subscribers. AUC (0.883) is close to Logistic Regression, so it separates the two classes well. |
| Overall, Winner for your dataset? | **Logistic Regression.** It has the best AUC, F1 and MCC, and those matter more than accuracy here because only about 11.5% of clients say yes. Even predicting "no" for everyone would score about 88% accuracy, so accuracy alone isn't useful. If the bank mainly wanted to reach as many likely subscribers as possible, Naive Bayes would be worth considering for its higher recall. |

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py     # optional - models are already saved in model/
streamlit run app.py
```

In the app: upload `test_data.csv` from the sidebar, choose a model, and check the metrics, confusion matrix, classification report and the comparison of all 5 models.
