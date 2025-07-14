# Autism Detection using Machine Learning

## Overview
This project implements multiple machine learning models to detect Autism Spectrum Disorder (ASD) traits based on the dataset provided by Dr. Fadi Tabtah. The dataset contains responses to screening questions, demographic details, and medical history, allowing predictive modeling for ASD detection.

## Dataset
**Total Entries:** 1,054  
**Features:** 19  
**Target Variable:** `ClassASD_Traits` (Yes/No)  

### Key Features:
- **A1 - A10:** Binary responses to ASD screening questions.
- **Qchat-10-Score:** Score from the Q-CHAT-10 autism screening test.
- **Age_Mons:** Age in months.
- **Sex, Ethnicity, Jaundice, Family_mem_with_ASD:** Demographic and medical history.
- **Who_completed_the_test:** Indicates the person who completed the screening.

## Machine Learning Models Implemented
- **Random Forest Classifier**
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**
- **Gradient Boosting Classifier**

## Feature Selection
Two different feature sets were explored:
1. **With Qchat-10-Score:** `['Qchat-10-Score', 'A9', 'A5', 'A6', 'A7', 'A4']`
2. **Without Qchat-10-Score:** `['A9', 'A5', 'A6', 'A7', 'A4']`

## Data Preprocessing
- Target variable `ClassASD_Traits` mapped to binary (Yes → 1, No → 0).
- Handled class imbalance using techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** to ensure balanced training data.
- Standardization using `StandardScaler` to normalize feature values.
- Train-test split performed for model training and validation.

## Model Training and Evaluation
### Steps Implemented:
1. **Data Cleaning:**
   - Removed any missing or inconsistent values.
   - Converted categorical variables into numerical representations.
2. **Feature Engineering:**
   - Selected two sets of features (with and without `Qchat-10-Score`).
   - Applied feature scaling where necessary.
3. **Handling Class Imbalance:**
   - Used **SMOTE** to oversample the minority class, improving model performance.
4. **Model Training:**
   - Trained six different models using both feature sets.
   - Optimized hyperparameters using `GridSearchCV`.
5. **Model Evaluation:**
   - Used `accuracy_score`, `classification_report`, and confusion matrices to compare model performance.
   - Identified best-performing models based on precision, recall, and F1-score.
6. **Results Analysis:**
   - Compared models to determine the most accurate for ASD detection.
   - Found that **Random Forest** and **Gradient Boosting** had the best classification accuracy.
   - Conducted feature importance analysis, highlighting `Qchat-10-Score` as a significant predictor.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/autism-detection.git
   cd autism-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook multi_model_analysis.ipynb
   ```

## Results & Conclusion
- The models were trained and evaluated on selected features.
- Addressed class imbalance using SMOTE to improve prediction performance.
- Random Forest and Gradient Boosting performed better in classification accuracy.
- Feature importance analysis highlighted `Qchat-10-Score` as a key predictor.

## Acknowledgment
Dataset provided by Dr. Fadi Tabtah for research on Autism Spectrum Disorder detection.
