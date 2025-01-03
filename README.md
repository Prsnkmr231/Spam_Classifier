# Spam Classifier

This project implements a **Spam Classifier** using Natural Language Processing (NLP) and machine learning techniques. It processes text data, converts it into numerical format using TF-IDF Vectorization, and trains a Multinomial Naive Bayes model to classify messages as spam or non-spam. The implementation also evaluates the model's accuracy.

---

## Prerequisites

### Libraries Used:
- **nltk**: For natural language processing tasks such as tokenization and lemmatization.
- **pandas**: For handling tabular data.
- **re**: For text cleaning using regular expressions.
- **scikit-learn**: For feature extraction, splitting data, and training/testing the model.

Install the required libraries:
```bash
pip install nltk pandas scikit-learn
```

---

## File Description

### Input File
- **spam.csv**: The dataset containing labeled messages for spam detection. Ensure the file has columns:
  - `label`: Indicates whether the message is "spam" or "ham".
  - `message`: The content of the message.

---

## Workflow

### 1. **Data Preparation**
- The input dataset is read using `pandas`.
- Unnecessary columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`) are removed.
- Column names are renamed for clarity:
  - `label`: Spam/Non-spam identifier.
  - `message`: The content of the message.

### 2. **Data Cleaning and Preprocessing**
- Each message is cleaned to remove non-alphabetical characters and converted to lowercase.
- Stopwords are removed using NLTK's stopwords list.
- Lemmatization is applied using NLTK's `WordNetLemmatizer` to standardize words.

### 3. **Feature Extraction**
- **TF-IDF Vectorization** is used to convert the cleaned messages into numerical format with a maximum of 2500 features.

### 4. **Model Training**
- Data is split into training and testing sets with an 80-20 split.
- The **Multinomial Naive Bayes** model is trained on the training set.

### 5. **Model Evaluation**
- Predictions are generated for the test set.
- A confusion matrix and accuracy score are computed to evaluate the model's performance.

## Results
- **Confusion Matrix**: Indicates true positives, true negatives, false positives, and false negatives.
- **Accuracy**: Measures the percentage of correctly classified messages.

---

## Customization
You can:
1. Use stemming instead of lemmatization by uncommenting the `PorterStemmer` code in the text cleaning section.
2. Adjust the `max_features` in `TfidfVectorizer` to experiment with different feature counts.
3. Use a different classifier (e.g., SVM, Random Forest) by replacing `MultinomialNB`.

---

## Dependencies
- Python 3.x
- nltk
- pandas
- scikit-learn

---



