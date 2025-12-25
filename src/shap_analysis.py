"""
Модуль для SHAP анализа важности признаков
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import joblib


def analyze_feature_importance(model, X: pd.DataFrame, output_dir: str = "plots"):
    """
    Выполняет SHAP анализ важности признаков
    
    Args:
        model: Обученная модель CatBoost
        X: Данные для анализа
        output_dir: Директория для сохранения графиков
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Создаем SHAP explainer
    # Для CatBoost используем TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Вычисляем SHAP значения (используем выборку для ускорения)
    sample_size = min(100, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # График важности признаков (summary plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # График важности признаков (bar plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Важность признаков (средние абсолютные значения)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    # Сохраняем важность признаков
    feature_importance.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    print(f"SHAP анализ завершен. Графики сохранены в: {output_dir}/")
    print("\nТоп-10 важных признаков:")
    print(feature_importance.head(10).to_string(index=False))
    
    return feature_importance


def plot_feature_importance_bar(feature_importance: pd.DataFrame, top_n: int = 15,
                                output_path: str = "plots/feature_importance_bar.png"):
    """
    Создает столбчатую диаграмму важности признаков
    """
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Важность признака')
    plt.title(f'Топ-{top_n} важных признаков')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

