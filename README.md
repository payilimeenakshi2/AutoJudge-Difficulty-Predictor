# AutoJudge: Programming Problem Difficulty Prediction

---

## Problem Overview

AutoJudge is a Machine Learning–based system that automatically predicts the difficulty of programming problems **using their textual descriptions**.

The system takes the following **textual inputs**:
- Problem Description
- Input Description
- Output Description

These text fields are combined, cleaned, and converted into numerical features using **TF-IDF vectorization**.  
Based on this representation, AutoJudge performs:
- **Classification** into difficulty categories (Easy / Medium / Hard)
- **Regression** to predict a continuous difficulty score

The goal of this project is to assist competitive programming platforms, educators, and learners by providing an automated and consistent estimation of problem difficulty without manual intervention.

---

## Dataset Description
The dataset consists of programming problem descriptions along with their difficulty labels and numerical difficulty scores.

Each record includes:

• Problem title  
• Problem description  
• Input description  
• Output description  
• Difficulty class  
• Difficulty score  

★ Dataset Size: 4112 problems

---

## Data Preprocessing
The following preprocessing steps were applied:

✔ Combined all textual fields into a single text column  
✔ Converted text to lowercase  
✔ Removed special characters and extra spaces  
✔ No missing values were present in the dataset  

---

## Feature Engineering
Two types of features were used:

### 1. Text-Based Features
✔ TF-IDF Vectorization (unigrams and bigrams)

### 2. Handcrafted Features
✔ Text length  
✔ Mathematical symbol count  
✔ Keyword frequencies (graph, dp, tree, recursion, greedy)

---

## Label Encoding
The target variable `problem_class` contains categorical labels:

• Easy  
• Medium  
• Hard  

Machine learning models require numerical inputs, so label encoding was applied.

Encoding used:

• Easy → 0  
• Medium → 1  
• Hard → 2  

✔ This transformation enables classification models to learn patterns from the data effectively.

---

## Model Performance and Results

### Classification Results
The following classification models were evaluated:

• Logistic Regression  
• Linear Support Vector Machine (SVM)  
• Random Forest Classifier  

★ Best Classification Model  
✔ Random Forest Classifier  
✔ Accuracy → ~52%

Random Forest achieved the highest accuracy and balanced performance across
Easy, Medium, and Hard classes compared to linear models.

---

### Regression Results
The following regression models were evaluated:

• Linear Regression  
• Random Forest Regressor  
• Gradient Boosting Regressor  

★ Best Regression Model  
✔ Random Forest Regressor  
✔ Mean Absolute Error (MAE) → ~1.70  
✔ R² Score → ~0.14  

Lower MAE indicates better prediction accuracy of the difficulty score.
Random Forest Regressor outperformed linear regression by capturing
non-linear relationships in the feature space.

---

## Saved Trained Models
The trained models and preprocessing objects are saved using `joblib`:

- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `rf_classifier.pkl`
- `rf_regressor.pkl`

These files are used directly in the web application.

---

## Web Interface Explanation
The web application is implemented in the `app.py` file using Streamlit.

★ This script loads the pre-trained models (`tfidf_vectorizer.pkl`, `label_encoder.pkl`, 
`rf_classifier.pkl`, `rf_regressor.pkl`), accepts problem text as input,
computes features, and outputs the predicted difficulty class and score in real time.

The Streamlit-based web interface allows users to predict the difficulty of
programming problems in real time.

The user provides:
- Problem Description
- Input Description
- Output Description

Workflow:
1. User enters the problem text.
2. Text is cleaned and vectorized using the saved TF-IDF model.
3. Handcrafted features are computed.
4. The classification model predicts difficulty class.
5. The regression model predicts a numeric difficulty score.
6. Results are displayed instantly on the UI.

The application runs locally and does not require hosting.

---
## Opening of Web browser
How to open web browser :
- source ~/myenv/bin/activate
- python -m streamlit run app.py
---
## Demo Video
A 2–3 minute demo explaining the project, model approach, and working web interface.

Demo Video Link: https://drive.google.com/file/d/1VyvprgoCO5fnSlspG8KxV3p9fUgVlvwJ/view?usp=drive_link

## Author
**Name:** Payili Meenakshi  
**Enrollment Number:** 23323028  

**GitHub Repository:**  
https://github.com/payilimeenakshi2/AutoJudge-Difficulty-Predictor

