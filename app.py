import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide"
)

if 'high_risk_threshold' not in st.session_state:
    st.session_state.high_risk_threshold = 0.7
if 'enable_clustering' not in st.session_state:
    st.session_state.enable_clustering = True
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'last_predictions' not in st.session_state:
    st.session_state.last_predictions = None
if 'last_input_data' not in st.session_state:
    st.session_state.last_input_data = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

st.title("Модуль прогноза оттока клиентов")

API_URL = "https://churn-analyzer-7ky6.onrender.com"

if not st.session_state.authenticated:
    st.header("Вход в систему")

    with st.form("login_form"):
        username = st.text_input("Имя пользователя", help="Введите ваше имя пользователя")
        password = st.text_input("Пароль", type="password", help="Введите ваш пароль")

        submitted = st.form_submit_button("Войти", type="primary")

    if submitted:
        if username and password:
            try:
                response = requests.post(
                    f"{API_URL}/token",
                    data={"username": username, "password": password},
                    timeout=10
                )

                if response.status_code == 200:
                    token_data = response.json()
                    st.session_state.auth_token = token_data["access_token"]
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = token_data["role"]
                    st.success("✅ Успешный вход в систему!")
                    st.rerun()
                else:
                    st.error("❌ Неверное имя пользователя или пароль")

            except Exception as e:
                st.error(f"❌ Ошибка подключения к API: {str(e)}")
        else:
            st.warning("Пожалуйста, введите имя пользователя и пароль")

    st.stop()

else:
    with st.sidebar:
        st.markdown(f"**Пользователь:** {st.session_state.username}")
        if st.button("Выйти", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.auth_token = None
            st.session_state.username = None
            st.rerun()

with st.sidebar:
    st.header("📚 Навигация")

    if st.session_state.user_role == "admin":
        available_pages = ["Прогноз", "Единичный прогноз", "Аналитика", "О проекте"]
    elif st.session_state.user_role == "analyst":
        available_pages = ["Аналитика", "О проекте"]
    else:
        available_pages = ["О проекте"]

    page = st.radio(
        "Выберите страницу",
        available_pages
    )

    st.header("⚙️ Настройки")

    st.session_state.high_risk_threshold = st.slider(
        "Порог высокого риска",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.high_risk_threshold,
        step=0.05,
        help="Вероятность оттока выше этого значения считается высоким риском"
    )

    st.session_state.enable_clustering = st.checkbox(
        "Включить кластеризацию",
        value=st.session_state.enable_clustering,
        help="Включить анализ по кластерам клиентов"
    )

    st.session_state.theme = st.selectbox(
        "Тема интерфейса",
        options=['light', 'dark'],
        index=0 if st.session_state.theme == 'light' else 1,
        help="Выберите тему интерфейса (пока не реализовано)"
    )

@st.cache_data(ttl=60)
def check_api_health(api_url):
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        return response.json()
    except:
        return {"status": "error", "model_loaded": False}

api_status = check_api_health(API_URL)

if page == "Прогноз":
    if api_status.get("model_loaded"):
        st.success("✅ API подключен, модель загружена")
    elif api_status.get("status") == "error":
        st.error("Не удалось подключиться к API. Убедитесь, что API запущен: `uvicorn src.api:app --reload`")
    else:
        st.warning("API подключен, модель не загружена")

    st.subheader("Загрузка данных")
    uploaded_file = st.file_uploader(
        "Загрузите CSV файл с данными клиентов",
        type=['csv'],
        help="Файл должен содержать те же колонки, что и обучающий датасет"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Загружено {len(df)} строк, {len(df.columns)} столбцов")

        with st.expander("Просмотр данных", expanded=False):
            st.dataframe(df, use_container_width=True)
            st.write(f"**Размер:** {df.shape[0]} строк × {df.shape[1]} столбцов")

        if st.button("Спрогнозировать отток", type="primary", use_container_width=True):
            if not api_status.get("model_loaded"):
                st.error("Модель не загружена в API")
            else:
                with st.spinner("Выполняется прогноз..."):
                    try:
                        data = df.to_dict('records')

                        headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"data": data, "clustering_enabled": st.session_state.enable_clustering},
                            headers=headers,
                            timeout=30
                        )

                        if response.status_code == 200:
                            result = response.json()

                            st.success(f"✅ Прогноз выполнен для {result['total_customers']} клиентов")

                            st.session_state.last_predictions = result['predictions']
                            st.session_state.last_input_data = df.copy()

                            clustering_enabled = result.get('clustering_enabled', False)
                            cluster_chart = result.get('cluster_chart', None)

                            results_df = pd.DataFrame(result['predictions'])

                            if 'cluster' in results_df.columns:
                                st.info(f"Кластеризация включена. Найдено кластеров: {results_df['cluster'].nunique()}")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                churn_count = results_df['churn_prediction'].sum()
                                st.metric("Клиентов с риском оттока", churn_count)
                            with col2:
                                churn_rate = (churn_count / len(results_df)) * 100
                                st.metric("Процент оттока", f"{churn_rate:.1f}%")
                            with col3:
                                avg_prob = results_df['churn_probability'].mean()
                                st.metric("Средняя вероятность", f"{avg_prob:.2%}")
                            with col4:
                                high_risk = (results_df['churn_probability'] > st.session_state.high_risk_threshold).sum()
                                st.metric(f"Высокий риск (>{st.session_state.high_risk_threshold:.0%})", high_risk)

                            st.subheader("Детальные результаты")
                            st.dataframe(results_df, use_container_width=True)

                            st.subheader("Визуализация результатов")

                            col1, col2 = st.columns(2)

                            with col1:
                                fig = px.histogram(
                                    results_df,
                                    x='churn_probability',
                                    nbins=30,
                                    title='Распределение вероятностей оттока',
                                    labels={'churn_probability': 'Вероятность оттока', 'count': 'Количество клиентов'}
                                )
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                churn_counts = results_df['churn_prediction'].value_counts()
                                fig = px.pie(
                                    values=churn_counts.values,
                                    names=['Без риска', 'Риск оттока'],
                                    title='Распределение предсказаний'
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            if 'cluster' in results_df.columns:
                                st.subheader("Анализ по кластерам")

                                if cluster_chart is not None:
                                    st.subheader("График распределения клиентов по кластерам")
                                    import base64
                                    from io import BytesIO
                                    image_data = base64.b64decode(cluster_chart)
                                    st.image(image_data, caption="Распределение клиентов по кластерам (PCA)", use_container_width=False)

                                col1, col2 = st.columns(2)

                                with col1:
                                    cluster_counts = results_df['cluster'].value_counts().sort_index()
                                    fig = px.bar(
                                        x=cluster_counts.index,
                                        y=cluster_counts.values,
                                        title='Распределение клиентов по кластерам',
                                        labels={'x': 'Кластер', 'y': 'Количество клиентов'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                with col2:
                                    cluster_churn = results_df.groupby('cluster')['churn_probability'].mean().sort_index()
                                    fig = px.bar(
                                        x=cluster_churn.index,
                                        y=cluster_churn.values,
                                        title='Средняя вероятность оттока по кластерам',
                                        labels={'x': 'Кластер', 'y': 'Средняя вероятность оттока'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                cluster_stats = results_df.groupby('cluster').agg({
                                    'churn_prediction': ['count', 'sum', 'mean'],
                                    'churn_probability': 'mean'
                                }).round(4)
                                cluster_stats.columns = ['Всего клиентов', 'С оттоком', 'Доля оттока', 'Средняя вероятность']
                                st.dataframe(cluster_stats, use_container_width=True)

                            high_risk_df = results_df[results_df['churn_probability'] > st.session_state.high_risk_threshold].sort_values(
                                'churn_probability', ascending=False
                            )
                            if len(high_risk_df) > 0:
                                st.subheader(f"Клиенты с высоким риском оттока (>{st.session_state.high_risk_threshold:.0%})")
                                st.dataframe(high_risk_df, use_container_width=True)

                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Скачать результаты (CSV)",
                                data=csv,
                                file_name="churn_predictions.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"Ошибка API: {response.text}")

                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")
                        st.exception(e)

    else:
        st.info("Загрузите CSV файл для начала работы")
        st.markdown("""
        ### Пример структуры данных:
        - `customerID` - ID клиента (опционально)
        - `gender` - Пол
        - `SeniorCitizen` - Пожилой клиент (0/1)
        - `Partner` - Есть партнер (Yes/No)
        - `Dependents` - Есть иждивенцы (Yes/No)
        - `tenure` - Стаж клиента (месяцы)
        - `PhoneService` - Телефонная служба (Yes/No)
        - `MultipleLines` - Несколько линий (Yes/No/No phone service)
        - `InternetService` - Интернет-сервис
        - `OnlineSecurity` - Онлайн безопасность
        - `OnlineBackup` - Онлайн резервное копирование
        - `DeviceProtection` - Защита устройства
        - `TechSupport` - Техническая поддержка
        - `StreamingTV` - Потоковое ТВ
        - `StreamingMovies` - Потоковые фильмы
        - `Contract` - Тип контракта
        - `PaperlessBilling` - Безбумажный биллинг (Yes/No)
        - `PaymentMethod` - Способ оплаты
        - `MonthlyCharges` - Ежемесячные платежи
        - `TotalCharges` - Общие платежи
        """)

elif page == "Единичный прогноз":
    if api_status.get("model_loaded"):
        st.success("✅ API подключен, модель загружена")
    elif api_status.get("status") == "error":
        st.error("Не удалось подключиться к API. Убедитесь, что API запущен: `uvicorn src.api:app --reload`")
    else:
        st.warning("API подключен, модель не загружена")

    st.header("Единичный прогноз оттока клиента")

    st.subheader("Введите данные клиента")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Пол", ["Male", "Female"], help="Пол клиента")
            senior_citizen = st.selectbox("Пожилой клиент", [0, 1], help="1 - да, 0 - нет")
            partner = st.selectbox("Есть партнер", ["Yes", "No"], help="Есть ли у клиента партнер")
            dependents = st.selectbox("Есть иждивенцы", ["Yes", "No"], help="Есть ли у клиента иждивенцы")

        with col2:
            tenure = st.number_input("Стаж клиента (месяцы)", min_value=0, max_value=100, value=12, help="Количество месяцев обслуживания")
            phone_service = st.selectbox("Телефонная служба", ["Yes", "No"], help="Есть ли телефонная служба")
            multiple_lines = st.selectbox("Несколько линий", ["Yes", "No", "No phone service"], help="Несколько телефонных линий")
            internet_service = st.selectbox("Интернет-сервис", ["DSL", "Fiber optic", "No"], help="Тип интернет-сервиса")

        with col3:
            online_security = st.selectbox("Онлайн безопасность", ["Yes", "No", "No internet service"], help="Есть ли онлайн безопасность")
            online_backup = st.selectbox("Онлайн резервное копирование", ["Yes", "No", "No internet service"], help="Есть ли онлайн резервное копирование")
            device_protection = st.selectbox("Защита устройства", ["Yes", "No", "No internet service"], help="Есть ли защита устройства")
            tech_support = st.selectbox("Техническая поддержка", ["Yes", "No", "No internet service"], help="Есть ли техническая поддержка")

        col4, col5, col6 = st.columns(3)

        with col4:
            streaming_tv = st.selectbox("Потоковое ТВ", ["Yes", "No", "No internet service"], help="Есть ли потоковое ТВ")
            streaming_movies = st.selectbox("Потоковые фильмы", ["Yes", "No", "No internet service"], help="Есть ли потоковые фильмы")
            contract = st.selectbox("Тип контракта", ["Month-to-month", "One year", "Two year"], help="Тип контракта")

        with col5:
            paperless_billing = st.selectbox("Безбумажный биллинг", ["Yes", "No"], help="Безбумажный биллинг")
            payment_method = st.selectbox("Способ оплаты",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                help="Способ оплаты")

        with col6:
            monthly_charges = st.number_input("Ежемесячные платежи", min_value=0.0, max_value=200.0, value=50.0, step=0.01, help="Ежемесячные платежи в долларах")
            total_charges = st.number_input("Общие платежи", min_value=0.0, max_value=10000.0, value=500.0, step=0.01, help="Общие платежи в долларах")

        submitted = st.form_submit_button("Спрогнозировать отток", type="primary", use_container_width=True)

    if submitted:
        if not api_status.get("model_loaded"):
            st.error("Модель не загружена в API")
        else:
            with st.spinner("Выполняется прогноз..."):
                try:
                    customer_data = {
                        "gender": gender,
                        "SeniorCitizen": senior_citizen,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless_billing,
                        "PaymentMethod": payment_method,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges
                    }

                    headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}
                    response = requests.post(
                        f"{API_URL}/predict_single",
                        json={"data": customer_data},
                        headers=headers,
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()

                        st.success("✅ Прогноз выполнен!")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            prediction = result['churn_prediction']
                            prediction_text = "Риск оттока" if prediction == 1 else "Без риска"
                            color = "🔴" if prediction == 1 else "🟢"
                            st.metric("Прогноз", f"{color} {prediction_text}")

                        with col2:
                            probability = result['churn_probability']
                            st.metric("Вероятность оттока", f"{probability:.1%}")

                        with col3:
                            risk_level = "Высокий риск" if probability > st.session_state.high_risk_threshold else "Низкий риск"
                            risk_color = "🔴" if probability > st.session_state.high_risk_threshold else "🟢"
                            st.metric("Уровень риска", f"{risk_color} {risk_level}")

                        st.subheader("Визуализация результата")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Вероятность оттока")
                            st.progress(probability)
                            if probability > st.session_state.high_risk_threshold:
                                st.warning(f"⚠️ Вероятность оттока выше порога высокого риска ({st.session_state.high_risk_threshold:.0%})")
                            else:
                                st.info(f"✅ Вероятность оттока ниже порога высокого риска ({st.session_state.high_risk_threshold:.0%})")

                        with col2:
                            fig = px.pie(
                                values=[1-probability, probability],
                                names=['Без риска', 'Риск оттока'],
                                title='Распределение предсказаний',
                                color_discrete_sequence=['#00CC96', '#EF553B']
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Детальная информация", expanded=False):
                            st.json(result)
                            st.write("**Введенные данные клиента:**")
                            st.json(customer_data)

                    else:
                        st.error(f"Ошибка API: {response.text}")

                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
                    st.exception(e)

elif page == "Аналитика":
    st.header("Аналитика и метрики модели")

    st.subheader("KPI Dashboard")

    if st.session_state.last_predictions is not None and st.session_state.last_input_data is not None:
        try:
            from src.metrics import calculate_kpis

            predictions_df = pd.DataFrame(st.session_state.last_predictions)
            input_data_df = st.session_state.last_input_data

            kpis = calculate_kpis(predictions_df, input_data_df, st.session_state.high_risk_threshold)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Всего клиентов",
                    f"{kpis['total_customers']:,}",
                    help="Общее количество клиентов в анализе"
                )

            with col2:
                st.metric(
                    "Клиентов с риском оттока",
                    f"{kpis['churn_count']:,}",
                    help="Количество клиентов, предсказанных к оттоку"
                )

            with col3:
                st.metric(
                    "Процент оттока",
                    f"{kpis['churn_rate']:.1f}%",
                    help="Процент клиентов с риском оттока"
                )

            with col4:
                st.metric(
                    "Средняя вероятность",
                    f"{kpis['avg_probability']:.1%}",
                    help="Средняя вероятность оттока по всем клиентам"
                )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    f"Высокий риск (>{st.session_state.high_risk_threshold:.0%})",
                    f"{kpis['high_risk_count']:,}",
                    help=f"Клиенты с вероятностью оттока выше {st.session_state.high_risk_threshold:.0%}"
                )

            if 'monthly_revenue_at_risk' in kpis:
                with col2:
                    st.metric(
                        "Месячная выручка под риском",
                        f"${kpis['monthly_revenue_at_risk']:,.0f}",
                        help="Ежемесячные платежи клиентов с риском оттока"
                    )

                with col3:
                    st.metric(
                        "Процент выручки под риском",
                        f"{kpis['revenue_risk_percentage']:.1f}%",
                        help="Доля месячной выручки от клиентов с риском оттока"
                    )

                with col4:
                    st.metric(
                        "Общая месячная выручка",
                        f"${kpis['total_monthly_revenue']:,.0f}",
                        help="Общая сумма ежемесячных платежей всех клиентов"
                    )

            st.markdown("---")

        except Exception as e:
            st.warning("Не удалось загрузить данные для KPI. Используйте страницу 'Предсказание' для расчета актуальных метрик.")
            st.info("KPI метрики будут рассчитаны на основе загруженных данных клиентов")

    plots_dir = Path("plots")

    if plots_dir.exists():
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                if (plots_dir / "roc_curve.png").exists():
                    with st.container():
                        st.subheader("ROC-кривая")
                        st.image(str(plots_dir / "roc_curve.png"), use_container_width=True)
            with col2:
                if (plots_dir / "pr_curve.png").exists():
                    with st.container():
                        st.subheader("Precision-Recall кривая")
                        st.image(str(plots_dir / "pr_curve.png"), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if (plots_dir / "confusion_matrix.png").exists():
                    with st.container():
                        st.subheader("Матрица ошибок")
                        st.image(str(plots_dir / "confusion_matrix.png"), use_container_width=True)
            with col2:
                if (plots_dir / "shap_summary.png").exists():
                    with st.container():
                        st.subheader("SHAP анализ важности признаков")
                        st.image(str(plots_dir / "shap_summary.png"), use_container_width=True)

            if (plots_dir / "feature_importance.csv").exists():
                with st.container():
                    st.subheader("Важность признаков (график)")
                    importance_df = pd.read_csv(plots_dir / "feature_importance.csv")
                    fig = px.bar(
                        importance_df.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Топ-15 важных признаков',
                        labels={'importance': 'Важность', 'feature': 'Признак'}
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

        if (plots_dir / "feature_importance.csv").exists():
            st.subheader("Важность признаков (таблица)")
            importance_df = pd.read_csv(plots_dir / "feature_importance.csv")
            st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Графики метрик будут доступны после обучения модели с расширенными метриками")

    if Path("metrics_report.txt").exists():
        with st.expander("Отчет по метрикам", expanded=False):
            with open("metrics_report.txt", "r", encoding="utf-8") as f:
                st.text(f.read())

elif page == "О проекте":
    st.header("О проекте")

    st.markdown("""
    ### Описание
    Это приложение использует машинное обучение для предсказания вероятности оттока клиентов
    телеком-оператора на основе их характеристик и истории использования услуг.

    ### Функциональность
    - Загрузка и анализ данных клиентов
    - Предсказание вероятности оттока
    - Визуализация результатов
    - Анализ метрик модели
    - SHAP анализ важности признаков

    ### Использование
    1. Загрузите CSV файл с данными клиентов
    2. Нажмите кнопку "Спрогнозировать отток"
    3. Просмотрите результаты и вероятности
    4. Экспортируйте результаты при необходимости

    ### API Endpoints
    - `GET /health` - Проверка здоровья сервиса
    - `POST /predict` - Предсказание оттока
    """)
