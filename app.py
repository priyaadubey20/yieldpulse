import streamlit as st
import pandas as pd
import numpy as np
import os
from google import genai

# Page Config
st.set_page_config(page_title="YieldPulse | AdTech L2R Diagnostic Engine", layout="wide")

st.title("⚡ YieldPulse")
st.subheader("AdTech Yield & Loss-to-Revenue (L2R) Automated Diagnostic Engine")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("sample_ad_logs.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Sidebar Controls
st.sidebar.header("Filter Telemetry")
selected_domain = st.sidebar.selectbox("Select Publisher Domain", df['domain'].unique())

filtered_df = df[df['domain'] == selected_domain].sort_values('date')

# Metric Cards
st.markdown("### 📊 Performance Summary")
latest_row = filtered_df.iloc[-1]
prev_row = filtered_df.iloc[-2]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current RPM ($)", f"${latest_row['rpm']:.2f}", f"{latest_row['rpm'] - prev_row['rpm']:.2f}")
col2.metric("Fill Rate (%)", f"{latest_row['fill_rate']:.1f}%", f"{latest_row['fill_rate'] - prev_row['fill_rate']:.1f}%")
col3.metric("CTR (%)", f"{latest_row['ctr']:.2f}%", f"{latest_row['ctr'] - prev_row['ctr']:.2f}%")
col4.metric("L2R Leakage (%)", f"{latest_row['l2r_leakage_pct']:.1f}%", f"{latest_row['l2r_leakage_pct'] - prev_row['l2r_leakage_pct']:.1f}%", delta_color="inverse")

# Data Table & Chart
st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 📈 Revenue & Leakage Trends")
    st.line_chart(filtered_df.set_index('date')[['rpm', 'l2r_leakage_pct']])

with col_right:
    st.markdown("### 🔍 Anomalies Detected")
    anomalies = filtered_df[filtered_df['l2r_leakage_pct'] > 5.0]
    if not anomalies.empty:
        for idx, row in anomalies.iterrows():
            st.error(f"**{row['date'].strftime('%Y-%m-%d')}**: High L2R ({row['l2r_leakage_pct']}%) on `{row['ad_placement']}` - Status: `{row['status']}`")
    else:
        st.success("No critical yield anomalies detected.")

# AI Root Cause Analysis Section
st.markdown("---")
st.markdown("### 🤖 LLM Yield Triage Assistant")

raw_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
api_key = raw_key.strip().strip('"').strip("'") if raw_key else None

if st.button("Run AI Root Cause Analysis"):
    if not api_key:
        st.warning("Please configure your GEMINI_API_KEY to generate AI insights.")
    else:
        with st.spinner("Analyzing telemetry logs & running diagnostic models..."):
            client = genai.Client(api_key=api_key)
            
            prompt = f"""
            You are an expert AdTech Yield Analytics Lead. Analyze the following telemetry log data for publisher domain '{selected_domain}':

            {filtered_df.to_string()}

            Provide a concise, executive-level diagnostic breakdown containing:
            1. **Root Cause Analysis**: What caused the drop in RPM and rise in Loss-to-Revenue (L2R)?
            2. **Technical Diagnosis**: Identify specific issues (e.g., misfiring ad tags, low bid density, schema mismatch).
            3. **Actionable Remediation**: Provide 3 step-by-step actions for product operations and ad ops teams to resolve the issue immediately.
            """

            # List of candidate models in order of priority
            models_to_try = [
                'gemini-3.8-flash',
                'gemini-2.5-flash',
                'gemini-1.5-flash',
                'gemini-2.0-flash'
            ]

            success = False
            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    st.markdown(response.text)
                    success = True
                    break
                except Exception:
                    continue

            if not success:
                st.error("All Gemini model endpoints are currently experiencing high demand. Please wait a few seconds and click again.")
