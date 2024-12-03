
# Fake News Detection with Machine Learning

This project aims to classify news articles as either "Fake" or "Real" using machine learning techniques. The classification is based on the title and text of the articles. The process involves reading and preprocessing the data, training different machine learning models, and evaluating their performance.

## Project Overview

1. **Data Preprocessing:**  
   - The dataset is loaded and cleaned by dropping unnecessary columns, handling missing values, and combining the article title with the text for feature extraction.
   
2. **Feature Extraction:**  
   - TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the text data into numerical features for the model.

3. **Model Training:**  
   - Different machine learning models, including Naive Bayes, Logistic Regression, and Random Forest, are trained and evaluated on the data.

4. **Model Evaluation:**  
   - Models are evaluated using accuracy and classification reports.

5. **Saving Models:**  
   - The trained models (Naive Bayes, Logistic Regression) and the TF-IDF vectorizer are saved for future use with new text data.

6. **Prediction Function:**  
   - A function is implemented to predict whether a new text is "Real" or "Fake" based on the trained model and vectorizer.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Models and Files](#models-and-files)
4. [Project Structure](#project-structure)
5. [License](#license)

---

## Installation

To run this project, you'll need to install the following Python libraries:

```bash
pip install pandas scikit-learn pickle
```

Additionally, make sure you have your dataset file (`train.csv`) available in the same directory or adjust the path in the code accordingly.

---

## Usage

### 1. Data Preparation

The data is read from a CSV file `train.csv`:

```python
df = pd.read_csv('train.csv')
```

The code then processes the dataset:
- Drops unnecessary columns (`id`, `author`).
- Handles missing values in the `title`.
- Combines the `title` and `text` into one `combined_text` column.

### 2. Preprocessing & Feature Extraction

The combined text is preprocessed by:
- Lowercasing all text.
- Removing non-alphabetic characters.

TF-IDF is then used to convert the cleaned text into numerical features:

```python
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### 3. Model Training and Evaluation

Several models are trained and evaluated, including:

- **Multinomial Naive Bayes:**
  - The model is trained using the TF-IDF features and evaluated using accuracy and classification report.
  - The trained model is saved to a file named `model.pkl`.

- **Logistic Regression:**
  - Logistic regression is trained and evaluated in a similar manner, and the model is saved to `logistic_regression.pkl`.

- **Random Forest Classifier:**
  - The Random Forest model is also trained and evaluated, though it is not saved in the final version of the code.

### 4. Making Predictions on New Text

To predict whether a new piece of text is real or fake, use the following function:

```python
def predict_new_text(new_text, model, vectorizer):
    # Preprocess and predict using the model and vectorizer
    ...
```

### 5. Example Prediction

Here is an example of how to use the prediction function with the saved models:

```python
new_text = "Stock Market Hits Record High as Economic Recovery Gains Momentum"
prediction = predict_new_text(new_text, model, vectorizer)
print(f"The prediction for the new text is: {prediction}")
```

---

## Models and Files

- **Model Files:**
  - `model.pkl`: The trained Naive Bayes model.
  - `logistic_regression.pkl`: The trained Logistic Regression model.
  - `vectorizer.pkl`: The trained TF-IDF vectorizer.

You can load these models in your Python environment using `pickle`:

```python
import pickle

# Load models
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
```

---

## Project Structure

```
fake_news_detection/
├── train.csv                # Dataset file
├── model.pkl                # Trained Naive Bayes model
├── logistic_regression.pkl  # Trained Logistic Regression model
├── vectorizer.pkl           # TF-IDF vectorizer model
├── app.py                   # Web application code
└── README.md                # This README file
```

---



## Notes


1. **Dataset:**
   Ensure you have the correct dataset (`train.csv`) to train the models. If you're using a custom dataset, you may need to adjust the code accordingly.

   **Project BY**
   Rohan chatse(Data scientists)

