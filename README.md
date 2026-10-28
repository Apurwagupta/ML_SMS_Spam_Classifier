# SMS Spam Classifier Using Machine Learning

## Project Overview

This mini project uses Machine Learning to classify SMS messages into two categories: **Spam** and **Ham (Legitimate Message)**. It uses TF-IDF for text feature extraction and a Multinomial Naive Bayes algorithm for classification.

The project is implemented in Python using Google Colab and the SMS Spam Collection dataset.

## Student Details

- **Name:** Apurwa
- **Roll Number:** 06
- **Faculty Guide:** Anil Walke
- **College:** ISBM College of Engineering, Nande, Pune
- **Department:** Artificial Intelligence and Data Science
- **Academic Year:** 2026–2027

## Objectives

- Load and explore an SMS spam dataset.
- Split the dataset into 80% training data and 20% testing data.
- Convert text messages into numerical features using TF-IDF.
- Train a Multinomial Naive Bayes classifier.
- Evaluate the model using accuracy, precision, recall, and a confusion matrix.
- Test the trained model on three new SMS messages.

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Dataset

The project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

The downloaded dataset contains **5,572 SMS messages**, labelled as:

- `ham` — legitimate messages
- `spam` — unwanted or spam messages

### Dataset Distribution

| Message Type | Count |
|---|---:|
| Ham | 4,825 |
| Spam | 747 |
| **Total** | **5,572** |

## Methodology

1. **Data Loading:** Load the SMS dataset into a Pandas DataFrame.
2. **Data Exploration:** Inspect message labels, missing values, and class distribution.
3. **Train-Test Split:** Divide the dataset into 80% training data and 20% testing data.
4. **Feature Extraction:** Use `TfidfVectorizer` to convert text messages into numerical features.
5. **Model Training:** Train a `MultinomialNB` classifier on the training data.
6. **Model Evaluation:** Calculate accuracy, precision, recall, and the confusion matrix.
7. **Prediction:** Classify three new SMS messages as spam or ham.

## Model Performance

The model achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 97.04% |
| Spam Precision | 100.00% |
| Spam Recall | 77.85% |

### Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Actual Ham | 966 | 0 |
| Actual Spam | 33 | 116 |

The model correctly classified 966 legitimate messages and 116 spam messages. It missed 33 spam messages and incorrectly classified zero legitimate messages as spam in this test set.

## Testing with New Messages

The model was tested using three new SMS messages.

| SMS Message | Prediction |
|---|---|
| Congratulations! You won a free prize. Claim now! | Spam |
| Hey, are you coming to college tomorrow? | Ham |
| URGENT! You have won cash. Click the link now! | Spam |

These predictions demonstrate how the trained model can classify previously unseen messages.

## How to Run the Project

1. Download or clone this repository.
2. Open `ML_SMS_Spam_Classifier.ipynb` in Google Colab.
3. Run the notebook cells in order.
4. Allow the dataset to download when prompted by the code.
5. Review the evaluation metrics, confusion matrix, and new-message predictions.

## Files in This Repository

- `ML_SMS_Spam_Classifier.ipynb` — Jupyter notebook containing the code and executed outputs.
- `sms_spam_dataset.csv` — CSV dataset exported by the notebook, if included.
- `spam_classifier_model.pkl` — Saved trained model, if included.
- `tfidf_vectorizer.pkl` — Saved TF-IDF vectorizer, if included.
- `Apurwa_ML_SMS_Spam_Classifier_Report.pdf` — Project report with output screenshots.

*Only list files that you have actually uploaded to the repository.*

## Conclusion

The project demonstrates how Natural Language Processing techniques and Machine Learning can be combined to build an SMS spam detection system. The Multinomial Naive Bayes classifier achieved 97.04% accuracy and 100% precision for spam on the test set. However, its 77.85% spam recall indicates that some spam messages were missed.

This project provides practical experience in text classification, feature extraction, model training, evaluation, and prediction.

## Acknowledgements

- UCI Machine Learning Repository for the SMS Spam Collection dataset.
- Scikit-learn documentation for machine learning tools and evaluation metrics.
- Google Colab for providing the notebook environment.
