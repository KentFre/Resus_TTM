# Resus_TTM
Repository for the data science pipeline for identifying patient subgroups benefiting from targeted temperature management (TTM) after resuscitation.

## Identifying Subpopulations Benefiting from Targeted Temperature Management in Cardiac Arrest: An Artificial Intelligence Approach

### Authors
Kent Fredriksdotter, Muna Shati, Valentina Ortiz-Milosevic, Nan Jiang, Aurelia Maria Ozora

---

## Introduction

Cardiac arrest (CA) remains a significant global health challenge, with low survival rates despite advancements in post-resuscitation care. Targeted Temperature Management (TTM), a critical intervention aimed at improving neurological outcomes, has been the subject of intense debate in recent years. Evidence from clinical trials suggests varying benefits of TTM protocols (e.g., 33°C, 36°C, or no TTM), indicating that patient outcomes may depend on specific subpopulation characteristics.

This repository documents the application of Artificial Intelligence (AI) methods, including Machine Learning (ML) and Deep Learning (DL), to explore counterfactual scenarios and identify CA subpopulations that might benefit from specific TTM protocols. By analyzing data from the International Cardiac Arrest Research Consortium (I-CARE) database, we aim to uncover insights that could inform personalized TTM strategies.

---

## Research Question

**Which cardiac arrest subpopulations benefit from specific TTM protocols (33°C, 36°C, or no TTM) as evidenced by neurological outcomes?**

---

## Methodology

This project follows the **CRISP-DM framework**, with each step implemented in a structured and modular manner across Jupyter Notebooks.

---

### 1. Data Download and Preparation  
**File**: `00_PatientData_Download.ipynb`  

- Patient-related data is downloaded from the [I-CARE repository](https://physionet.org/content/i-care/2.1/).  
- Raw ECG files are handled separately and processed using the **`aws_ecg_download_scripts`** on an **AWS EC2 instance**.

---

### 2. Exploratory Data Analysis (EDA) and Preprocessing  
**File**: `01_Exploratory_Data_Analysis.ipynb`  

- Conducted descriptive statistics and data visualization to analyze the overall dataset.  
- Preprocessing included:  
  - Handling missing values and imputation (e.g., ROSC time, sex).  
  - Transformation of key features:  
    - Converted **TTM** to categorical groups: `33°C`, `36°C`, and `No TTM`.  
    - Standardized numerical features for downstream analysis.  
- Outputs **clean, ready-to-use datasets** for subsequent analysis.

---

### 3. Feature Importance Analysis  
**File**: `02_Feature_Importance.ipynb`  

- Identified critical predictors of neurological outcomes using three methods:  
  1. Mutual Information  
  2. Permutation Importance  
  3. Tree-based Feature Importance (Random Forest)  
- Results guided the selection of features for **modeling and clustering**.

---

### 4. Subgroup Identification via Clustering  
**File**: `03_Clustering.ipynb`  

- Applied **K-Means clustering** to uncover subgroups of patients with shared characteristics.  
- Evaluated clustering performance using:  
  - **Silhouette Score**  
  - **Calinski-Harabasz Index**  
- Analyzed clusters to identify patterns in **TTM protocols** and outcomes, such as worse outcomes in non-shockable rhythm clusters.

---

### 5. Machine Learning Model Training  
**File**: `04_Prediction_Model.ipynb`  

- Built and trained conventional machine learning models to predict neurological outcomes:  
  - **Decision Tree**, **Random Forest**, **AdaBoost**, **Extra Trees**, **XGBoost**, **SVM**, **k-NN**, and **Logistic Regression**.  
- Applied hyperparameter tuning using **GridSearchCV** to optimize each model.  
- Compared models using evaluation metrics: **Accuracy, Precision, Recall, F1 Score, and AUC-ROC**.

---

### 6. Neural Network Training with ECG Data  
**File**: `05_Neural_Network_ECG.ipynb`  

- Extracted ECG-based features (e.g., HR, HRV, LF/HF ratio) for **5-minute intervals over 24 hours**.  
- Combined ECG time-series data with clinical features to train a **hybrid neural network**:  
  - **LSTM layers** for time-series data.  
  - **Dense layers** for tabular clinical data.  
- Optimized the network using **Keras Tuner** and evaluated performance using **AUC-ROC**.

---

### 7. Counterfactual Analysis and Consensus Voting  
**File**: `06_Counterfactual_Analysis.ipynb`  

- Performed **counterfactual analysis** using ANN and XGBoost models under **ceteris paribus** conditions.  
- Implemented **consensus-based decision voting** to identify patients whose outcomes would change with adjusted TTM protocols.  
- Visualized changes using a **Sankey Diagram** and analyzed group characteristics using:  
  - Descriptive statistics  
  - **SHAP analysis**  
  - Decision tree-based rule extraction.

---

### 8. ECG Data Combination and Processing  
**File**: `98_combine_ecg_data.ipynb`  

- Combined ECG-derived features with patient data.  
- Applied signal filtering techniques to clean and preprocess time-series data:  
  - **High-pass filter** (baseline drift removal).  
  - **Notch filter** (utility noise).  
  - **Median filter** (artifact spikes removal).  
- Fixed timeseries inconsistencies and validated feature extraction accuracy.

---

## Data and Models  

- **Raw and Processed Data**: Stored in the **`data`** folder.  
- **Trained Models**: Best-performing models (e.g., ANN, XGBoost) are saved in the **`models`** folder for reproducibility.
