import pandas as pd
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests

# ─────────────────────────────────────────────────────────────
# Tavily API Setup
# ─────────────────────────────────────────────────────────────
TAVILY_API_KEY = "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"



def get_apollo_product_description(product_name):
    query = f"{product_name} site:apollopharmacy.in"
    tavily_url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False
    }
    try:
        res = requests.post(tavily_url, json=payload, timeout=15)
        if res.status_code == 200:
            data = res.json()
            return data.get("answer") or "🔍 No description found via Tavily."
        else:
            return "⚠️ Failed to fetch product description from Tavily."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ─────────────────────────────────────────────────────────────
# CNN + LSTM MODEL
# ─────────────────────────────────────────────────────────────
class CNNLSTMModel(nn.Module):
    def __init__(self, input_len, forecast_len):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, forecast_len)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────
def scale_series(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    return scaled, scaler

def next_n_months(start_str, n):
    try:
        month = datetime.strptime(start_str[:3] + " 01 " + start_str[-2:], "%b %d %y")
    except ValueError:
        return [f"Month {i+1}" for i in range(n)]
    return [(month + relativedelta(months=i+1)).strftime("%b'%y").upper() for i in range(n)]

# ─────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Forecast Dashboard", page_icon="📊")

st.title("📦 Apollo Pharmacy Supply Forecasting")
st.markdown("🔮 Predict future sales using historical monthly data powered by a CNN + LSTM neural network.")

st.sidebar.header("📂 Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
forecast_len = st.sidebar.select_slider("Forecast Duration (Months)", options=[3, 6, 12])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
    except:
        st.error("⚠️ Couldn't read the Excel file. Make sure it's a valid `.xlsx` file.")
        st.stop()

    df = df.dropna(subset=["ITEM NAME"])
    time_cols = [col for col in df.columns if "'" in col and col[:3].isalpha()]
    time_cols.sort(key=lambda x: datetime.strptime(x.replace("'", ""), "%b%y"))

    if len(time_cols) < 4:
        st.error("❌ Not enough time-series columns (minimum 4 months required).")
        st.stop()

    latest_month = time_cols[-1]
    st.sidebar.success(f"🕒 Detected latest month: {latest_month}")

    product_names = df["ITEM NAME"].unique()
    selected_product = st.sidebar.selectbox("🛒 Select Product", product_names)

    st.subheader(f"📘 Product Description for: **{selected_product}**")
    description = get_apollo_product_description(selected_product)
    st.markdown(description)

    row = df[df["ITEM NAME"] == selected_product][time_cols].dropna(axis=1)
    if row.empty or row.shape[1] < 4:
        st.warning(f"⚠️ Not enough data for **{selected_product}**. Needs ≥ 4 months of sales.")
        st.stop()

    full_series = row.values.flatten()
    input_len = len(full_series)
    scaled_series, scaler = scale_series(full_series)

    X = torch.tensor(scaled_series.reshape(1, 1, -1), dtype=torch.float32)
    model = CNNLSTMModel(input_len=input_len, forecast_len=forecast_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    y_dummy = torch.tensor(scaled_series[-forecast_len:], dtype=torch.float32).reshape(1, -1)

    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y_dummy)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        forecast_scaled = model(X).numpy().flatten()
        forecast_actual = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    forecast_months = next_n_months(latest_month, forecast_len)

    st.markdown(f"### 📊 {forecast_len}-Month Forecast for **{selected_product}**")
    df_out = pd.DataFrame({
        "Forecast Month": forecast_months,
        "Predicted Sales": np.round(forecast_actual, 2)
    })
    st.dataframe(df_out, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast_months, forecast_actual, marker='o', linestyle='--', color='mediumblue')
    ax.set_title(f"{selected_product} – {forecast_len} Month Sales Forecast", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales Units")
    ax.grid(True)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # Check for deficit or surplus based on last month's actual vs forecast first month
    last_month_actual = full_series[-1]
    next_month_forecast = forecast_actual[0]
    if next_month_forecast > last_month_actual:
        st.warning(f"📉 Stock Deficit Alert: Forecasted demand ({next_month_forecast:.0f}) exceeds last known stock ({last_month_actual:.0f})")
    else:
        st.success(f"📦 Stock OK: Forecasted demand ({next_month_forecast:.0f}) is within available stock ({last_month_actual:.0f})")

    st.success("✅ Forecast generated successfully.")
else:
    st.info("📅 Please upload your Excel file using the sidebar to begin.")
