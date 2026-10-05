import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import google.generativeai as genai

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

@st.cache_data(show_spinner=False)
def generate_triage_report(api_key, domain, telemetry_text):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.8-flash')

    prompt = f"""
    You are an expert AdTech Yield Analytics Lead. Analyze the following telemetry log data for publisher domain '{domain}':

    {telemetry_text}

    Provide a concise, executive-level diagnostic breakdown containing:
    1. **Root Cause Analysis**: What caused the drop in RPM and rise in Loss-to-Revenue (L2R)?
    2. **Technical Diagnosis**: Identify specific issues (e.g., misfiring ad tags, low bid density, schema mismatch).
    3. **Actionable Remediation**: Provide 3 step-by-step actions for product operations and ad ops teams to resolve the issue immediately.
    """

    # Retry loop for 429 rate limit backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(12)  # Wait 12 seconds before auto-retrying
            else:
                raise e

if st.button("Run AI Root Cause Analysis"):
    if not api_key:
        st.warning("Please configure your GEMINI_API_KEY in Streamlit Secrets.")
    else:
        with st.spinner("Analyzing telemetry logs & running diagnostic models..."):
            try:
                report = generate_triage_report(api_key, selected_domain, filtered_df.to_string())
                st.markdown(report)
            except Exception as e:
                if "429" in str(e):
                    st.error("Google AI Studio rate limit is still cooling down. Please wait 1 minute before clicking again.")
                else:
                    st.error(f"Failed to generate analysis: {e}")
