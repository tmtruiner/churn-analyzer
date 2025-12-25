import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import preprocess_data
from src.quality_control import quality_controller
from src.auth import authenticate_user, create_access_token, get_current_active_user, User, Token
from src.clustering import prepare_for_clustering, perform_kmeans_clustering, add_cluster_features, visualize_clusters_2d
from sklearn.decomposition import PCA
import io
import base64

app = FastAPI(title="Churn Prediction API", version="1.0.0")


@app.middleware("http")
async def quality_control_middleware(request: Request, call_next):
    start_time = time.time()

    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000

        quality_controller.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms
        )

        return response

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        quality_controller.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=500,
            latency_ms=latency_ms,
            error_message=str(e)
        )

        raise

MODEL_PATH = Path("models/churn_model.pkl")
CAT_FEATURES_PATH = Path("models/categorical_features.pkl")
FEATURE_ORDER_PATH = Path("models/feature_order.pkl")

model = None
categorical_features = []
feature_order = None


def load_model():
    global model, categorical_features, feature_order

    if not MODEL_PATH.exists():
        print("Модель не найдена. Сначала обучите модель: python -m src.train")
        return

    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Модель загружена")

        if CAT_FEATURES_PATH.exists():
            categorical_features = joblib.load(CAT_FEATURES_PATH)
            print(f"✅ Категориальные признаки загружены: {categorical_features}")

        if FEATURE_ORDER_PATH.exists():
            feature_order = joblib.load(FEATURE_ORDER_PATH)
            print(f"✅ Порядок признаков загружен: {len(feature_order)} признаков")

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")


load_model()


class PredictionRequest(BaseModel):
    data: List[Dict]
    clustering_enabled: bool = False


class SinglePredictionRequest(BaseModel):
    data: Dict


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username})

    return Token(access_token=access_token, token_type="bearer", role=user.role)


@app.post("/predict")
def predict(request: PredictionRequest, current_user: User = Depends(get_current_active_user)):
    if current_user.role == "analyst":
        raise HTTPException(status_code=403, detail="Analyst role does not have access to prediction features")

    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        df = pd.DataFrame(request.data)

        drift_info = quality_controller.check_data_drift(df)

        X, _ = preprocess_data(df, target_col=None)

        if categorical_features:
            for col in categorical_features:
                if col in X.columns:
                    X[col] = X[col].apply(lambda x: str(x) if pd.notna(x) and x != '' else '')
                    X[col] = X[col].replace(['nan', 'None', 'NaN', 'nan', '<NA>', '0.0', '1.0'], '')
                    X[col] = X[col].astype(str)

        cluster_labels = None
        cluster_chart = None
        if request.clustering_enabled:
            try:
                print(f"Starting clustering for {len(df)} customers")
                X_clustering, scaler, _ = prepare_for_clustering(df)
                print(f"Data prepared for clustering, shape: {X_clustering.shape}")
                cluster_labels, _ = perform_kmeans_clustering(X_clustering, n_clusters=4)
                print(f"Clustering completed, labels: {cluster_labels}")
                X = add_cluster_features(X, cluster_labels, method='one_hot')
                print(f"Cluster features added to X, new shape: {X.shape}")

                try:
                    fig = visualize_clusters_2d(X_clustering, cluster_labels, title="Распределение клиентов по кластерам")

                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                    buf.seek(0)
                    cluster_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
                    buf.close()
                    fig.clf()

                except Exception as chart_e:
                    print(f"Ошибка создания основного графика кластеризации: {chart_e}")
                    try:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                        ax.bar(unique_labels, counts, color='skyblue', edgecolor='black')
                        ax.set_xlabel('Кластер')
                        ax.set_ylabel('Количество клиентов')
                        ax.set_title('Распределение клиентов по кластерам')
                        ax.grid(True, alpha=0.3)

                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                        buf.seek(0)
                        cluster_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
                        buf.close()
                        fig.clf()
                        print("✅ Создан резервный график кластеризации (распределение по кластерам)")
                    except Exception as fallback_e:
                        print(f"Ошибка создания резервного графика кластеризации: {fallback_e}")
                        cluster_chart = None

            except Exception as e:
                print(f"Ошибка кластеризации: {e}")
                cluster_labels = None
                cluster_chart = None

        if feature_order is not None:
            for col in feature_order:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_order]

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                "customer_index": i,
                "churn_prediction": int(pred),
                "churn_probability": float(prob)
            }
            if cluster_labels is not None:
                result["cluster"] = int(cluster_labels[i])
            results.append(result)

        response = {
            "predictions": results,
            "total_customers": len(results),
            "drift_detected": drift_info['drift_detected'],
            "drift_score": drift_info['drift_score'],
            "clustering_enabled": request.clustering_enabled
        }

        if request.clustering_enabled:
            if cluster_chart is not None:
                response["cluster_chart"] = cluster_chart
                print(f"✅ Cluster chart added to response, length: {len(cluster_chart)}")
            else:
                response["cluster_chart"] = None
                print("❌ Cluster chart is None")

        if drift_info['drift_detected']:
            response["warning"] = "Обнаружен дрейф данных. Рекомендуется переобучение модели."

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {str(e)}")


@app.post("/predict_single")
def predict_single(request: SinglePredictionRequest, current_user: User = Depends(get_current_active_user)):
    if current_user.role == "analyst":
        raise HTTPException(status_code=403, detail="Analyst role does not have access to prediction features")

    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        df = pd.DataFrame([request.data])

        X, _ = preprocess_data(df, target_col=None)

        if categorical_features:
            for col in categorical_features:
                if col in X.columns:
                    X[col] = X[col].apply(lambda x: str(x) if pd.notna(x) and x != '' else '')
                    X[col] = X[col].replace(['nan', 'None', 'NaN', 'nan', '<NA>', '0.0', '1.0'], '')
                    X[col] = X[col].astype(str)

        if feature_order is not None:
            for col in feature_order:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_order]

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]

        result = {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
