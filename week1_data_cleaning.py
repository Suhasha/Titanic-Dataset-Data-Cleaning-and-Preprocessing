"""
Week 1 - Data Acquisition, Cleaning and Preprocessing
Dataset: Titanic - Machine Learning from Disaster
Source: Kaggle - https://www.kaggle.com/competitions/titanic/data

Instructions:
1. Download train.csv from the Kaggle Titanic dataset.
2. Put train.csv in the same folder as this script.
3. Install packages:
   pip install pandas numpy matplotlib seaborn scikit-learn
4. Run:
   python week1_data_cleaning.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "train.csv"
OUTPUT_FILE = "titanic_cleaned.csv"
MODEL_FILE = "titanic_model_ready.csv"

# ---------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------
# The dataset is publicly available from Kaggle.
# Download train.csv manually from:
# https://www.kaggle.com/competitions/titanic/data

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "train.csv was not found. Download it from Kaggle and place it "
        "in the same folder as this script."
    )

df = pd.read_csv(INPUT_FILE)

print("\n========== INITIAL DATA ==========")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------
# 2. INITIAL EXPLORATION
# ---------------------------------------------------------
print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe(include="all").T)

print("\n========== MISSING VALUES ==========")
print(df.isna().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------
clean = df.copy()

# 3.1 Remove duplicate rows
clean = clean.drop_duplicates().copy()

# 3.2 Standardize text fields
for col in ["Sex", "Embarked", "Cabin"]:
    if col in clean.columns:
        clean[col] = clean[col].astype("string").str.strip()

# Convert blank strings to missing values
clean = clean.replace(r"^\s*$", np.nan, regex=True)

# 3.3 Validate categorical values
valid_sex = {"male", "female"}
valid_embarked = {"S", "C", "Q"}

invalid_sex = sorted(set(clean["Sex"].dropna()) - valid_sex)
invalid_embarked = sorted(set(clean["Embarked"].dropna()) - valid_embarked)

print("\nInvalid Sex values:", invalid_sex)
print("Invalid Embarked values:", invalid_embarked)

# 3.4 Validate numeric ranges
invalid_age = ((clean["Age"] < 0) | (clean["Age"] > 100)).sum()
invalid_fare = (clean["Fare"] < 0).sum()
invalid_pclass = (~clean["Pclass"].isin([1, 2, 3])).sum()

print("Invalid Age values:", int(invalid_age))
print("Negative Fare values:", int(invalid_fare))
print("Invalid Pclass values:", int(invalid_pclass))

# ---------------------------------------------------------
# 4. MISSING VALUE HANDLING
# ---------------------------------------------------------
print("\nMissing values before treatment:")
print(clean.isna().sum())

# Cabin has very high missingness, so it is removed.
# Keeping it would introduce a large amount of artificial imputation.
clean = clean.drop(columns=["Cabin"])

# Age: use median within Pclass + Sex groups.
# This preserves differences in age distributions between groups.
group_medians = clean.groupby(["Pclass", "Sex"])["Age"].transform("median")
clean["Age"] = clean["Age"].fillna(group_medians)

# Fallback if any Age value is still missing.
clean["Age"] = clean["Age"].fillna(clean["Age"].median())

# Embarked: only a very small number of values are missing,
# so use the mode.
clean["Embarked"] = clean["Embarked"].fillna(clean["Embarked"].mode()[0])

print("\nMissing values after treatment:")
print(clean.isna().sum())

# ---------------------------------------------------------
# 5. OUTLIER DETECTION USING IQR
# ---------------------------------------------------------
Q1 = clean["Fare"].quantile(0.25)
Q3 = clean["Fare"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

fare_outliers = clean[
    (clean["Fare"] < lower_bound) |
    (clean["Fare"] > upper_bound)
]

print("\n========== FARE OUTLIERS ==========")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Number of Fare outliers:", len(fare_outliers))

# Do NOT delete these passengers because high fares can be genuine.
# Instead, cap extreme values at the IQR boundaries.
clean["Fare"] = clean["Fare"].clip(lower=lower_bound, upper=upper_bound)

# Age = 80 is plausible, so it is retained rather than treated as an error.

# ---------------------------------------------------------
# 6. FEATURE ENGINEERING / PREPROCESSING
# ---------------------------------------------------------
clean["FamilySize"] = clean["SibSp"] + clean["Parch"] + 1
clean["IsAlone"] = (clean["FamilySize"] == 1).astype(int)

# One-hot encode categorical variables for later machine learning.
model_ready = pd.get_dummies(
    clean,
    columns=["Sex", "Embarked"],
    drop_first=True,
    dtype=int
)

# ---------------------------------------------------------
# 7. SAVE CLEANED DATA
# ---------------------------------------------------------
clean.to_csv(OUTPUT_FILE, index=False)
model_ready.to_csv(MODEL_FILE, index=False)

print("\n========== FINAL DATA ==========")
print("Cleaned shape:", clean.shape)
print("Model-ready shape:", model_ready.shape)
print("\nSaved:", OUTPUT_FILE)
print("Saved:", MODEL_FILE)

# ---------------------------------------------------------
# 8. VISUALIZATION
# ---------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution Before Cleaning")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution_before.png", dpi=160)
plt.show()

plt.figure(figsize=(7, 4))
plt.boxplot(df["Fare"].dropna())
plt.title("Fare Before Outlier Treatment")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("fare_before.png", dpi=160)
plt.show()

plt.figure(figsize=(7, 4))
plt.boxplot(clean["Fare"].dropna())
plt.title("Fare After IQR Capping")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("fare_after.png", dpi=160)
plt.show()
