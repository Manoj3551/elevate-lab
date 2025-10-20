📁 Task: Data Cleaning & Preprocessing
🧰 Tools Used

Python

Pandas

NumPy

Matplotlib

Seaborn

🧑‍💻 Python Code (main.py)
# Task 1: Data Cleaning & Preprocessing using Titanic Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("✅ Dataset Loaded Successfully")
print(df.head())
print("\n---Basic Info---")
print(df.info())

# Step 2: Check Missing Values
print("\n---Missing Values---")
print(df.isnull().sum())

# Step 3: Handle Missing Values
# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

print("\n---Missing Values After Imputation---")
print(df.isnull().sum())

# Step 4: Encode Categorical Features
# Convert 'Sex' and 'Embarked' to numeric
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("\n---Data After Encoding---")
print(df.head())

# Step 5: Feature Scaling (Normalization / Standardization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scale numerical columns
num_cols = ['Age', 'Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])
print("\n---Data After Scaling---")
print(df.head())

# Step 6: Detect & Remove Outliers using Boxplots
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title('Outlier Detection')
plt.show()

# Remove outliers using IQR method
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
print("\n✅ Outliers removed. Final Shape:", df.shape)

# Step 7: Final Cleaned Data
print("\n---Final Cleaned Data---")
print(df.head())

# Save cleaned data
df.to_csv("cleaned_titanic.csv", index=False)
print("\n💾 Cleaned data saved as 'cleaned_titanic.csv'")

# 🧼 Task 1: Data Cleaning & Preprocessing

## 📌 Objective
The goal of this task is to learn how to **clean and preprocess raw data** to make it ready for machine learning models.  
We use the [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) for this exercise.

---

## 🧰 Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🪜 Steps Followed

### 1. Load the Dataset
We used the Titanic dataset and loaded it using Pandas `read_csv()`.

### 2. Explore Basic Info
Checked for:
- Null values
- Data types
- Shape and sample records

### 3. Handle Missing Values
- **Age** → filled with **median**  
- **Embarked** → filled with **mode**  
- **Cabin** → dropped (too many missing values)

### 4. Encode Categorical Variables
- Converted `Sex` and `Embarked` columns to numeric using **One-Hot Encoding**.

### 5. Feature Scaling
- Applied **Standardization** using `StandardScaler` for `Age` and `Fare`.

### 6. Outlier Detection & Removal
- Used **Boxplot** visualization to detect outliers.
- Removed outliers using **IQR (Interquartile Range)** method.

### 7. Save Final Cleaned Data
- Exported the final cleaned dataset as `cleaned_titanic.csv`.

---

## 📊 Output
- Null values handled ✅
- Categorical variables encoded ✅
- Data scaled ✅
- Outliers removed ✅
- Final dataset ready for ML models ✅

---

## ❓ Interview Questions Covered
1. Types of missing data
2. Handling categorical variables
3. Normalization vs Standardization
4. Outlier detection methods
5. Importance of preprocessing
6. One-hot vs Label encoding
7. Data imbalance handling
8. Effect of preprocessing on accuracy

---

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/your-username/data-cleaning-task.git

# Navigate
cd data-cleaning-task

# Run
python main.py
🏁 Final Notes

Data cleaning is a critical step in ML pipelines.
Clean data = Better accuracy ✅, better model performance ✅, fewer errors ✅.

✍️ Author: MANOJ KUMAR
