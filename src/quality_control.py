"""
Модуль контроля качества модели
Включает детекцию дрейфа, мониторинг задержек и мониторинг запросов
"""
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Метрики запроса"""
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class QualityMetrics:
    """Общие метрики качества"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    requests_per_minute: float = 0.0
    success_rate: float = 0.0
    drift_detected: bool = False
    drift_score: float = 0.0
    last_updated: Optional[datetime] = None


@dataclass
class DriftDetector:
    """Детектор дрейфа данных"""
    reference_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    current_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    drift_threshold: float = 0.05  # p-value threshold for statistical tests
    min_samples: int = 100  # minimum samples for drift detection

    def update_reference_stats(self, data: pd.DataFrame, categorical_features: List[str] = None):
        """Обновить референтные статистики для детекции дрейфа"""
        self.reference_stats = {}

        for col in data.columns:
            if col in (categorical_features or []):
                # Для категориальных признаков - распределение частот
                value_counts = data[col].value_counts(normalize=True)
                self.reference_stats[col] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict()
                }
            else:
                # Для числовых признаков - среднее и стандартное отклонение
                if pd.api.types.is_numeric_dtype(data[col]):
                    self.reference_stats[col] = {
                        'type': 'numeric',
                        'mean': data[col].mean(),
                        'std': data[col].std(),
                        'min': data[col].min(),
                        'max': data[col].max()
                    }

        logger.info(f"Reference statistics updated for {len(self.reference_stats)} features")

    def detect_drift(self, data: pd.DataFrame, categorical_features: List[str] = None) -> Dict[str, Any]:
        """Обнаружить дрейф в данных"""
        if not self.reference_stats:
            return {'drift_detected': False, 'drift_score': 0.0, 'details': {}}

        drift_details = {}
        total_drift_score = 0.0
        features_with_drift = 0

        for col in data.columns:
            if col not in self.reference_stats:
                continue

            ref_stats = self.reference_stats[col]

            if ref_stats['type'] == 'categorical':
                # Тест хи-квадрат для категориальных признаков
                ref_dist = ref_stats['distribution']
                current_dist = data[col].value_counts(normalize=True).to_dict()

                # Создаем contingency table
                all_categories = set(ref_dist.keys()) | set(current_dist.keys())
                ref_counts = [ref_dist.get(cat, 0) * len(data) for cat in all_categories]
                current_counts = [current_dist.get(cat, 0) * len(data) for cat in all_categories]

                if len(all_categories) > 1 and sum(ref_counts) > 0 and sum(current_counts) > 0:
                    try:
                        chi2, p_value = stats.chisquare(current_counts, ref_counts)
                        drift_score = 1 - p_value  # Convert p-value to drift score
                        drift_details[col] = {
                            'drift_score': drift_score,
                            'p_value': p_value,
                            'significant': p_value < self.drift_threshold
                        }
                        if p_value < self.drift_threshold:
                            features_with_drift += 1
                        total_drift_score += drift_score
                    except:
                        drift_details[col] = {'drift_score': 0.0, 'error': 'chisquare test failed'}

            else:
                # Kolmogorov-Smirnov test для числовых признаков
                if pd.api.types.is_numeric_dtype(data[col]):
                    ref_mean = ref_stats['mean']
                    ref_std = ref_stats['std']

                    try:
                        # Стандартизируем данные
                        current_data = data[col].dropna()
                        if len(current_data) > 10:
                            # KS test against normal distribution with reference parameters
                            ks_stat, p_value = stats.kstest(current_data, 'norm', args=(ref_mean, ref_std))
                            drift_score = 1 - p_value
                            drift_details[col] = {
                                'drift_score': drift_score,
                                'p_value': p_value,
                                'significant': p_value < self.drift_threshold,
                                'current_mean': current_data.mean(),
                                'ref_mean': ref_mean
                            }
                            if p_value < self.drift_threshold:
                                features_with_drift += 1
                            total_drift_score += drift_score
                    except:
                        drift_details[col] = {'drift_score': 0.0, 'error': 'ks test failed'}

        avg_drift_score = total_drift_score / max(len(drift_details), 1)
        drift_detected = features_with_drift > 0 or avg_drift_score > 0.5

        return {
            'drift_detected': drift_detected,
            'drift_score': avg_drift_score,
            'features_with_drift': features_with_drift,
            'details': drift_details
        }


class LatencyMonitor:
    """Монитор задержек запросов"""

    def __init__(self, warning_threshold_ms: float = 1000.0, critical_threshold_ms: float = 5000.0):
        self.warning_threshold_ms = warning_threshold_ms
        self.critical_threshold_ms = critical_threshold_ms
        self.latencies: List[float] = []
        self.lock = threading.Lock()

    def record_latency(self, latency_ms: float):
        """Записать задержку"""
        with self.lock:
            self.latencies.append(latency_ms)
            # Ограничиваем размер списка последними 1000 измерениями
            if len(self.latencies) > 1000:
                self.latencies = self.latencies[-1000:]

    def get_latency_stats(self) -> Dict[str, float]:
        """Получить статистику задержек"""
        with self.lock:
            if not self.latencies:
                return {
                    'avg_latency_ms': 0.0,
                    'median_latency_ms': 0.0,
                    'p95_latency_ms': 0.0,
                    'p99_latency_ms': 0.0,
                    'max_latency_ms': 0.0,
                    'min_latency_ms': 0.0
                }

            latencies = np.array(self.latencies)
            return {
                'avg_latency_ms': np.mean(latencies),
                'median_latency_ms': np.median(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'max_latency_ms': np.max(latencies),
                'min_latency_ms': np.min(latencies)
            }

    def check_thresholds(self) -> Dict[str, bool]:
        """Проверить превышение порогов"""
        stats = self.get_latency_stats()
        p95_latency = stats['p95_latency_ms']

        return {
            'warning': p95_latency > self.warning_threshold_ms,
            'critical': p95_latency > self.critical_threshold_ms
        }


class RequestMonitor:
    """Монитор запросов"""

    def __init__(self):
        self.requests: List[RequestMetrics] = []
        self.lock = threading.Lock()
        self.max_history_hours = 24

    def record_request(self, endpoint: str, method: str, status_code: int,
                      latency_ms: float, error_message: Optional[str] = None):
        """Записать запрос"""
        request = RequestMetrics(
            timestamp=datetime.now(),
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            success=status_code < 400,
            error_message=error_message
        )

        with self.lock:
            self.requests.append(request)
            # Очищаем старые записи
            cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
            self.requests = [r for r in self.requests if r.timestamp > cutoff_time]

    def get_request_stats(self) -> Dict[str, Any]:
        """Получить статистику запросов"""
        with self.lock:
            if not self.requests:
                return {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'success_rate': 0.0,
                    'requests_per_minute': 0.0,
                    'error_distribution': {}
                }

            total = len(self.requests)
            successful = sum(1 for r in self.requests if r.success)
            failed = total - successful

            # Запросы за последнюю минуту
            one_minute_ago = datetime.now() - timedelta(minutes=1)
            recent_requests = [r for r in self.requests if r.timestamp > one_minute_ago]
            requests_per_minute = len(recent_requests)

            # Распределение ошибок
            error_distribution = {}
            for r in self.requests:
                if not r.success and r.error_message:
                    error_distribution[r.error_message] = error_distribution.get(r.error_message, 0) + 1

            return {
                'total_requests': total,
                'successful_requests': successful,
                'failed_requests': failed,
                'success_rate': successful / total if total > 0 else 0.0,
                'requests_per_minute': requests_per_minute,
                'error_distribution': error_distribution
            }


class ModelQualityController:
    """Контроллер качества модели"""

    def __init__(self):
        self.drift_detector = DriftDetector()
        self.latency_monitor = LatencyMonitor()
        self.request_monitor = RequestMonitor()
        self.reference_data_path = Path("models/reference_data.pkl")
        self.categorical_features_path = Path("models/categorical_features.pkl")
        self.lock = threading.Lock()

    def initialize_reference_data(self, reference_data: pd.DataFrame, categorical_features: List[str]):
        """Инициализировать референтные данные для детекции дрейфа"""
        self.drift_detector.update_reference_stats(reference_data, categorical_features)

        # Сохранить референтные данные
        reference_info = {
            'data': reference_data,
            'categorical_features': categorical_features,
            'created_at': datetime.now()
        }
        joblib.dump(reference_info, self.reference_data_path)
        logger.info("Reference data initialized and saved")

    def load_reference_data(self):
        """Загрузить референтные данные"""
        if self.reference_data_path.exists():
            try:
                reference_info = joblib.load(self.reference_data_path)
                self.drift_detector.update_reference_stats(
                    reference_info['data'],
                    reference_info['categorical_features']
                )
                logger.info("Reference data loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load reference data: {e}")
                return False
        return False

    def check_data_drift(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """Проверить дрейф данных"""
        with self.lock:
            return self.drift_detector.detect_drift(input_data)

    def record_request(self, endpoint: str, method: str, status_code: int,
                      latency_ms: float, error_message: Optional[str] = None):
        """Записать метрики запроса"""
        self.latency_monitor.record_latency(latency_ms)
        self.request_monitor.record_request(endpoint, method, status_code, latency_ms, error_message)

    def get_quality_metrics(self) -> QualityMetrics:
        """Получить общие метрики качества"""
        latency_stats = self.latency_monitor.get_latency_stats()
        request_stats = self.request_monitor.get_request_stats()

        # Проверяем пороги задержек
        threshold_checks = self.latency_monitor.check_thresholds()

        metrics = QualityMetrics(
            total_requests=request_stats['total_requests'],
            successful_requests=request_stats['successful_requests'],
            failed_requests=request_stats['failed_requests'],
            avg_latency_ms=latency_stats['avg_latency_ms'],
            max_latency_ms=latency_stats['max_latency_ms'],
            min_latency_ms=latency_stats['min_latency_ms'],
            requests_per_minute=request_stats['requests_per_minute'],
            success_rate=request_stats['success_rate'],
            last_updated=datetime.now()
        )

        # Добавляем предупреждения
        if threshold_checks['critical']:
            logger.warning(f"Critical latency threshold exceeded: {latency_stats['p95_latency_ms']:.2f}ms")
        elif threshold_checks['warning']:
            logger.warning(f"Warning latency threshold exceeded: {latency_stats['p95_latency_ms']:.2f}ms")

        return metrics

    def get_detailed_report(self) -> Dict[str, Any]:
        """Получить детальный отчет о качестве"""
        metrics = self.get_quality_metrics()
        latency_stats = self.latency_monitor.get_latency_stats()
        request_stats = self.request_monitor.get_request_stats()

        return {
            'quality_metrics': {
                'total_requests': metrics.total_requests,
                'success_rate': metrics.success_rate,
                'avg_latency_ms': metrics.avg_latency_ms,
                'requests_per_minute': metrics.requests_per_minute,
                'drift_detected': metrics.drift_detected
            },
            'latency_stats': latency_stats,
            'request_stats': request_stats,
            'threshold_checks': self.latency_monitor.check_thresholds(),
            'last_updated': metrics.last_updated.isoformat() if metrics.last_updated else None
        }


# Глобальный экземпляр контроллера качества
quality_controller = ModelQualityController()
