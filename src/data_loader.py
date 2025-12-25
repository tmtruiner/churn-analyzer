import pandas as pd
import numpy as np
from pathlib import Path


def load_data(file_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df: pd.DataFrame, target_col: str = 'Churn') -> tuple:
    data = df.copy()

    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)

    numeric_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if col == 'TotalCharges':
                data[col] = data[col].fillna(0)
            else:
                data[col] = data[col].fillna(0)

    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if target_col is None or col != target_col:
            data[col] = data[col].apply(lambda x: str(x) if pd.notna(x) else '')
            data[col] = data[col].replace(['nan', 'None', 'NaN', 'nan', '<NA>'], '')

    if target_col is not None and target_col in data.columns:
        y = data[target_col].map({'Yes': 1, 'No': 0}) if target_col == 'Churn' else data[target_col]
        X = data.drop(target_col, axis=1)
    else:
        X = data
        y = None

    return X, y


def get_categorical_features(X: pd.DataFrame) -> list:
    numeric_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    categorical_cols = [col for col in categorical_cols if col not in numeric_columns]

    return categorical_cols


if __name__ == "__main__":
    df = load_data()
    print(f"Размер датасета: {df.shape}")
    print(f"\nПервые строки:")
    print(df.head())
    print(f"\nИнформация о данных:")
    print(df.info())
