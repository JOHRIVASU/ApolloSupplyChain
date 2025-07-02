import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ───────────────────────────────
# Streamlit Page Configuration
# ───────────────────────────────
st.set_page_config(layout="wide", page_title="📊 Apollo Forecasting", page_icon="📈")
st.title("📦 Apollo Sales Forecasting System")
st.markdown("🧠 *AI-driven forecasting using ARIMA, SARIMA, and LSTM models.*")

forecast_horizon = 12
uploaded_file = st.sidebar.file_uploader("📤 Upload your Excel File", type=["xlsx"])
model_export_choice = st.sidebar.selectbox("📌 Choose Model for Bulk Export", ["LSTM", "ARIMA", "SARIMA"])

# ───────────────────────────────
# Helper Functions
# ───────────────────────────────
def prepare_series(sales_row):
    time_cols = [col for col in sales_row.index if "'" in col and col[:3].isalpha()]
    time_cols.sort(key=lambda x: datetime.strptime(x.replace("'", ""), "%b%y"))
    sales = sales_row[time_cols].astype(float)
    dates = pd.date_range(start=datetime.strptime(time_cols[0].replace("'", ""), "%b%y"), periods=len(time_cols), freq='M')
    return pd.Series(data=sales.values, index=dates)

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": np.sqrt(mse)}

def summarize_forecast(series, forecast, model_name):
    rows = []
    for h in [3, 6, 12]:
        trimmed_forecast = forecast[:h]
        actual = series[-h:] if len(series) >= h else series
        score = evaluate(actual, trimmed_forecast[:len(actual)])
        rows.append({
            "Model": model_name,
            "Horizon": f"{h} Months",
            "MAE": round(score["MAE"], 2),
            "RMSE": round(score["RMSE"], 2),
            "Forecast Total": round(trimmed_forecast.sum(), 2),
            "Forecast Values": ', '.join(str(round(x, 2)) for x in trimmed_forecast.values)
        })
    return rows

def arima_forecast(series):
    model = ARIMA(series, order=(1, 1, 1)).fit()
    forecast = model.forecast(steps=forecast_horizon)
    return pd.Series(forecast.values, index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=forecast_horizon, freq='M'))

def sarima_forecast(series):
    if len(series) < 24 or series.std() == 0:
        return pd.Series([series.mean()] * forecast_horizon,
                         index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=forecast_horizon, freq='M'))
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
        forecast = result.forecast(steps=forecast_horizon)
        return pd.Series(forecast.values, index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=forecast_horizon, freq='M'))
    except Exception as e:
        print(f"SARIMA fallback: {e}")
        return pd.Series([series.mean()] * forecast_horizon,
                         index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=forecast_horizon, freq='M'))

def lstm_forecast(series):
    data = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - 12):
        X.append(scaled[i:i+12])
        y.append(scaled[i+12])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(12, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0)

    last_seq = scaled[-12:].reshape(1, 12, 1)
    predictions = []
    for _ in range(forecast_horizon):
        pred = model.predict(last_seq, verbose=0)[0]
        predictions.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[pred]], axis=1)
    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return pd.Series(forecast, index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=forecast_horizon, freq='M'))

def check_stock_status(total, forecasted):
    status = []
    for f in forecasted:
        buffer_forecast = f * 1.15
        buffer_total = total + 0.5 * total
        if buffer_total - buffer_forecast < f:
            status.append("Deficit")
        else:
            status.append("Sufficient")
    return status

def plot_forecast(original, forecast_dict):
    if st.get_option("theme.base") == "light":
        plt.style.use("default")
    else:
        plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(original.index, original.values, label="Actual", marker='o', linewidth=2, color='#76b5c5')
    colors = {"ARIMA": "#00cc96", "SARIMA": "#ffa15a", "LSTM": "#ab63fa"}
    markers = {"ARIMA": 'x', "SARIMA": 'd', "LSTM": 's'}

    for label, forecast in forecast_dict.items():
        ax.plot(forecast.index, forecast.values,
                label=label,
                linestyle='--',
                marker=markers.get(label, 'x'),
                linewidth=2,
                color=colors.get(label, None))

    ax.set_title("📊 Forecasted Sales", fontsize=18, weight='bold')
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Sales", fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    st.pyplot(fig)

# ───────────────────────────────
# Main App Logic
# ───────────────────────────────
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.dropna(subset=["ITEM NAME", "ITEM CODE"])

    search_input = st.sidebar.text_input("🔍 Search by ITEM NAME or CODE").lower()
    filtered_df = df[df["ITEM NAME"].str.lower().str.contains(search_input) | df["ITEM CODE"].astype(str).str.contains(search_input)]

    if filtered_df.empty:
        st.sidebar.warning("No products match your search.")
        st.stop()

    selected = st.sidebar.selectbox("🛒 Select Product", options=filtered_df["ITEM NAME"].unique())
    selected_row = df[df["ITEM NAME"] == selected].iloc[0]
    series = prepare_series(selected_row)

    st.subheader(f"📊 Forecasting for: `{selected}`")
    st.caption(f"🗓️ Forecasting starts from: **{series.index[-1].strftime('%b %Y')}** → +12 months")

    arima_pred = arima_forecast(series)
    sarima_pred = sarima_forecast(series)
    lstm_pred = lstm_forecast(series)

    forecasts = {"ARIMA": arima_pred, "SARIMA": sarima_pred, "LSTM": lstm_pred}
    plot_forecast(series, forecasts)

    results = []
    for model_name, forecast in forecasts.items():
        results.extend(summarize_forecast(series, forecast, model_name))

    st.subheader("📈 Model Performance Summary")
    st.dataframe(pd.DataFrame(results))

    # ─────── Bulk Forecasting ───────
    if st.button("📦 Generate Forecasts for All SKUs"):
        st.info("Running full dataset forecast, please wait...")

        export_all = []
        for _, row in df.iterrows():
            item_code = row["ITEM CODE"]
            item_name = row["ITEM NAME"]
            total = row.get("TOTAL", np.nan)
            product_series = prepare_series(row)

            if model_export_choice == "LSTM":
                pred = lstm_forecast(product_series)
            elif model_export_choice == "ARIMA":
                pred = arima_forecast(product_series)
            else:
                pred = sarima_forecast(product_series)

            status = check_stock_status(total, pred) if not np.isnan(total) else ["N/A"] * len(pred)
            export_df = pd.DataFrame({
                "ITEM CODE": item_code,
                "ITEM NAME": item_name,
                "Month": pred.index.strftime("%b-%Y"),
                "Predicted Sales": pred.values.round(2),
                "Stock Status": status
            })
            export_all.append(export_df)

        combined = pd.concat(export_all)
        csv_data = combined.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download All Product Forecasts CSV", csv_data, "Apollo_All_SKU_Forecast.csv", "text/csv")
else:
    st.info("📥 Please upload your Excel file to begin.")
