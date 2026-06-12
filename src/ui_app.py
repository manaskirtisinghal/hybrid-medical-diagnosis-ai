import streamlit as st
import pandas as pd
from diagnosis_system import DiagnosisSystem

# -----------------------------
# Load model + dataset
# -----------------------------
DATA_PATH = "data/processed/final_dataset.csv"

system = DiagnosisSystem(DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Get symptoms dynamically (NO HARDCODING 🔥)
SYMPTOMS = [col for col in df.columns if col != "diseases"]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Medical Diagnosis AI", layout="wide")

st.title("🧠 Explainable Medical Diagnosis System")
st.write("Select symptoms and get diagnosis with reasoning.")

# -----------------------------
# Symptom Selection (2 columns UI)
# -----------------------------
col1, col2 = st.columns(2)

patient = {}

half = len(SYMPTOMS) // 2

for i, sym in enumerate(SYMPTOMS):
    if i < half:
        patient[sym] = 1 if col1.checkbox(sym) else 0
    else:
        patient[sym] = 1 if col2.checkbox(sym) else 0

# -----------------------------
# Diagnose Button
# -----------------------------
if st.button("🔍 Diagnose"):

    diagnosis, explanation = system.diagnose(patient)

    st.success(f"✅ Final Diagnosis: {diagnosis}")

    st.subheader("🧠 Explanation Trace")
    for line in explanation:
        st.write(f"- {line}")