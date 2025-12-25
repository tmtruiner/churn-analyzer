"""
Модуль для настройки гиперпараметров через Optuna
"""
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np


def objective(trial, X_train, y_train, X_val, y_val, categorical_features):
    """
    Целевая функция для Optuna
    """
    params = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 1),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_state': 42,
        'verbose': False,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC'
    }
    
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=categorical_features,
        eval_set=(X_val, y_val),
        verbose=False,
        use_best_model=True
    )
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred_proba)
    
    return score


def optimize_hyperparameters(X_train, y_train, X_val, y_val, categorical_features, 
                            n_trials: int = 50):
    """
    Оптимизирует гиперпараметры модели через Optuna
    
    Args:
        X_train: Обучающие данные
        y_train: Обучающие метки
        X_val: Валидационные данные
        y_val: Валидационные метки
        categorical_features: Список категориальных признаков
        n_trials: Количество итераций оптимизации
        
    Returns:
        dict: Лучшие параметры
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, categorical_features),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\nЛучшие параметры:")
    print(f"  ROC-AUC: {study.best_value:.4f}")
    print(f"  Параметры: {study.best_params}")
    
    return study.best_params


def train_with_best_params(X_train, y_train, X_test, y_test, categorical_features, 
                          best_params: dict):
    """
    Обучает модель с лучшими параметрами
    
    Args:
        X_train: Обучающие данные
        y_train: Обучающие метки
        X_test: Тестовые данные
        y_test: Тестовые метки
        categorical_features: Список категориальных признаков
        best_params: Лучшие параметры
        
    Returns:
        Trained model
    """
    model = CatBoostClassifier(
        **best_params,
        random_state=42,
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=50
    )
    
    model.fit(
        X_train, y_train,
        cat_features=categorical_features,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    return model

