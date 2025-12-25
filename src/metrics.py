"""
Модуль для расчета расширенных метрик модели
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, 
    precision_recall_curve, confusion_matrix, classification_report
)
from pathlib import Path
import seaborn as sns


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Вычисляет все метрики модели
    
    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения
        y_pred_proba: Вероятности предсказаний
        
    Returns:
        dict: Словарь с метриками
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba)
    }
    
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str = "plots/roc_curve.png"):
    """
    Строит ROC-кривую
    """
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный классификатор')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return fpr, tpr, roc_auc


def plot_pr_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str = "plots/pr_curve.png"):
    """
    Строит Precision-Recall кривую
    """
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR кривая (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall кривая')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return precision, recall, pr_auc


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str = "plots/confusion_matrix.png"):
    """
    Строит матрицу ошибок
    """
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return cm


def generate_metrics_report(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray,
                           output_file: str = "metrics_report.txt"):
    """
    Генерирует отчет с метриками
    """
    metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("METRICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Основные метрики:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-score:  {metrics['f1_score']:.4f}\n")
        f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")
        f.write(f"  PR-AUC:    {metrics['pr_auc']:.4f}\n\n")
        
        f.write("Матрица ошибок:\n")
        f.write(f"  True Negative (TN):  {cm[0, 0]}\n")
        f.write(f"  False Positive (FP): {cm[0, 1]}\n")
        f.write(f"  False Negative (FN): {cm[1, 0]}\n")
        f.write(f"  True Positive (TP):  {cm[1, 1]}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))


def plot_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray,
                     output_dir: str = "plots"):
    """
    Создает все графики метрик
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    plot_roc_curve(y_true, y_pred_proba, f"{output_dir}/roc_curve.png")
    plot_pr_curve(y_true, y_pred_proba, f"{output_dir}/pr_curve.png")
    plot_confusion_matrix(y_true, y_pred, f"{output_dir}/confusion_matrix.png")

    print(f"Графики метрик сохранены в: {output_dir}/")


def calculate_kpis(predictions_df: pd.DataFrame, original_df: pd.DataFrame = None,
                   high_risk_threshold: float = 0.7) -> dict:
    """
    Вычисляет KPI метрики на основе результатов предсказаний

    Args:
        predictions_df: DataFrame с результатами предсказаний (churn_prediction, churn_probability)
        original_df: Исходный DataFrame с данными клиентов (опционально, для revenue метрик)
        high_risk_threshold: Порог для определения высокого риска

    Returns:
        dict: Словарь с KPI метриками
    """
    kpis = {}

    # Основные метрики
    total_customers = len(predictions_df)
    churn_count = predictions_df['churn_prediction'].sum()
    churn_rate = (churn_count / total_customers) * 100 if total_customers > 0 else 0
    avg_probability = predictions_df['churn_probability'].mean()
    high_risk_count = (predictions_df['churn_probability'] > high_risk_threshold).sum()

    kpis['total_customers'] = total_customers
    kpis['churn_count'] = int(churn_count)
    kpis['churn_rate'] = churn_rate
    kpis['avg_probability'] = avg_probability
    kpis['high_risk_count'] = int(high_risk_count)

    # Revenue метрики (если доступны)
    if original_df is not None and len(original_df) == len(predictions_df):
        # Ежемесячные платежи клиентов с риском оттока
        monthly_revenue_at_risk = original_df.loc[predictions_df['churn_prediction'] == 1, 'MonthlyCharges'].sum()
        # Общие платежи клиентов с риском оттока
        total_revenue_at_risk = original_df.loc[predictions_df['churn_prediction'] == 1, 'TotalCharges'].sum()
        # Ежемесячные платежи клиентов с высоким риском
        high_risk_monthly_revenue = original_df.loc[predictions_df['churn_probability'] > high_risk_threshold, 'MonthlyCharges'].sum()

        kpis['monthly_revenue_at_risk'] = monthly_revenue_at_risk
        kpis['total_revenue_at_risk'] = total_revenue_at_risk
        kpis['high_risk_monthly_revenue'] = high_risk_monthly_revenue

        # Общая месячная выручка
        total_monthly_revenue = original_df['MonthlyCharges'].sum()
        kpis['total_monthly_revenue'] = total_monthly_revenue

        # Процент выручки под риском
        revenue_risk_percentage = (monthly_revenue_at_risk / total_monthly_revenue) * 100 if total_monthly_revenue > 0 else 0
        kpis['revenue_risk_percentage'] = revenue_risk_percentage

    return kpis

