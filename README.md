# Task 4: Classification with Logistic Regression

## Objective
The goal of this task is to build a binary classification model using **Logistic Regression** to predict whether a tumor is **benign (non-cancerous)** or **malignant (cancerous)** based on various diagnostic features.

## Dataset
We used the **Breast Cancer Wisconsin Dataset** from the UCI Machine Learning Repository, available via Kaggle:  
[Dataset Link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

The dataset contains:
- **30 numeric features** describing tumor characteristics (mean radius, texture, smoothness, etc.).
- **Target variable**:
  - `0` → Malignant
  - `1` → Benign

## Steps Performed
1. **Data Loading**
   - Loaded the dataset directly from Scikit-learn or downloaded from Kaggle.
   
2. **Data Preprocessing**
   - Dropped `id` and other irrelevant columns.
   - Encoded target labels (`M` and `B`) into numeric form.
   - Split into **training** (80%) and **testing** (20%) sets.
   - Standardized numeric features using `StandardScaler`.

3. **Model Training**
   - Implemented **Logistic Regression** from `scikit-learn`.
   - Used training data to fit the model.

4. **Model Evaluation**
   - **Confusion Matrix**
   - **Accuracy, Precision, Recall, and F1-score**
   - **ROC Curve** and **AUC score**

5. **Threshold Tuning**
   - Adjusted decision threshold to optimize for different metrics.
   - Demonstrated trade-offs between precision and recall.
   
6. **Sigmoid Function Explanation**
   - Logistic regression predicts probabilities using:
     \[
     \sigma(z) = \frac{1}{1 + e^{-z}}
     \]
     - This maps any real number into the range (0, 1).
     - Thresholding converts probabilities into class labels.

## Results
| Metric      | Value   |
|-------------|---------|
| Accuracy    | 100.00% |
| Precision   | 100.00% |
| Recall      | 100.00% |
| F1-Score    | 100.00% |
| AUC         | 1.00    |

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- scikit-learn

Install with:
```bash
pip install pandas numpy matplotlib scikit-learn
