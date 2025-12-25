import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data_loader import load_data, preprocess_data, get_categorical_features
from src.metrics import calculate_all_metrics, plot_all_metrics, generate_metrics_report
from src.hyperparameter_tuning import optimize_hyperparameters, train_with_best_params


def train_model():
    print("Загрузка данных...")
    df = load_data()
    print(f"Загружено строк: {df.shape[0]}")

    print("Предобработка данных...")
    X, y = preprocess_data(df)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    categorical_features = get_categorical_features(X_train)
    print(f"Категориальные признаки: {categorical_features}")

    print("Оптимизация гиперпараметров...")
    best_params = optimize_hyperparameters(
        X_train, y_train, X_val, y_val, categorical_features, n_trials=20
    )

    print("Обучение модели с оптимизированными гиперпараметрами...")
    model = train_with_best_params(
        X_train, y_train, X_val, y_val, categorical_features, best_params
    )

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)

    print("\n=== Результаты модели ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")

    print("\nСоздание графиков метрик...")
    plot_all_metrics(y_test, y_pred, y_pred_proba)

    print("Генерация отчета по метрикам...")
    generate_metrics_report(y_test, y_pred, y_pred_proba)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "churn_model.pkl"
    cat_features_path = model_dir / "categorical_features.pkl"
    feature_order_path = model_dir / "feature_order.pkl"

    feature_order = X_train.columns.tolist()

    joblib.dump(model, model_path)
    joblib.dump(categorical_features, cat_features_path)
    joblib.dump(feature_order, feature_order_path)

    print(f"\nМодель сохранена в: {model_path}")
    print(f"Категориальные признаки сохранены в: {cat_features_path}")
    print(f"Порядок признаков сохранен в: {feature_order_path}")

    return model, categorical_features


if __name__ == "__main__":
    train_model()
