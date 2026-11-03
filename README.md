# Week 1 – Data Acquisition, Cleaning and Preprocessing

## Objective

Acquire a public dataset, inspect its quality, clean missing/inconsistent data, detect and treat outliers, and prepare the dataset for further analysis.

## Dataset

**Titanic – Machine Learning from Disaster**

Source: Kaggle  
https://www.kaggle.com/competitions/titanic/data

## Folder Contents

- `code/` – Python preprocessing code
- `data/` – cleaned/model-ready data and generated plots
- `report/` – final Week 1 report
- `screenshots/` – place execution screenshots here if required by the evaluator

## Key Cleaning Decisions

1. Duplicate rows were checked and removed.
2. Text values were standardized.
3. `Cabin` was removed because of very high missingness.
4. Missing `Age` values were filled using the median within `Pclass` and `Sex` groups.
5. Missing `Embarked` values were filled using the mode.
6. `Fare` outliers were identified using the IQR method.
7. Extreme `Fare` values were capped instead of deleted because high fares may be genuine observations.
8. `FamilySize` and `IsAlone` were created as derived features.
9. `Sex` and `Embarked` were one-hot encoded in the model-ready dataset.

## Reproducibility

```bash
pip install pandas numpy matplotlib scikit-learn
python code/week1_data_cleaning.py
```

## Deliverables

The report explains the methodology, rationale, challenges, preprocessing impact and results.
