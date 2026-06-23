import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from optimizer import optimize_resources

st.set_page_config(page_title="Astram Command Control", layout="wide", page_icon="🚨")

@st.cache_resource
def load_production_pipeline():
    with open("congestion_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    pipeline = load_production_pipeline()
    model = pipeline["model"]
    encoders = pipeline["encoders"]
    unique_causes = pipeline["unique_causes"]
    unique_corridors = pipeline["unique_corridors"]
    unique_veh_types = pipeline["unique_veh_types"]
except FileNotFoundError:
    st.error("❌ Execute 'python train_model.py' to generate artifacts.")
    st.stop()

@st.cache_data
def load_geospatial_telemetry():
    df = pd.read_csv("Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df.dropna(subset=["latitude", "longitude"])

df_raw = load_geospatial_telemetry()

st.title(" Astram Tactical Command & Control Interface")
st.markdown("### Predictive Congestion Mitigation via High-Dimensional Telemetry")
st.markdown("---")

tab1, tab2 = st.tabs([" Predictive Resource Planner", " Historical Geospatial Intel"])

with tab1:
    st.subheader("Compute Operational Contingency Requirements")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### **Incoming Event Characteristics**")
        in_type = st.selectbox("Operational Classification", ["unplanned", "planned"])
        in_cause = st.selectbox("Event Primary Cause", unique_causes)
        in_corridor = st.selectbox("Target Transport Corridor", unique_corridors)
        
        # New Golden Features
        in_veh = st.selectbox("Vehicle Type Involved (If Applicable)", unique_veh_types, index=unique_veh_types.index("not_applicable") if "not_applicable" in unique_veh_types else 0)
        
        c_time1, c_time2 = st.columns(2)
        with c_time1:
            in_hour = st.slider("Hour of Day", 0, 23, 17)
        with c_time2:
            day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
            in_day_str = st.selectbox("Day of Week", list(day_map.keys()))
            in_day = day_map[in_day_str]
            
        in_closure = st.checkbox("Enforce Absolute Corridor Road Closure")
        
        # Transforms
        enc_type = encoders["event_type"].transform([in_type.strip().lower()])[0]
        enc_cause = encoders["event_cause"].transform([in_cause.strip().lower()])[0]
        enc_corridor = encoders["corridor"].transform([in_corridor.strip()])[0]
        enc_veh = encoders["veh_type"].transform([in_veh.strip().lower()])[0]
        enc_closure = 1 if in_closure else 0
        
        # Predict
        feature_vector = [[enc_type, enc_cause, enc_corridor, enc_veh, enc_closure, in_hour, in_day]]
        prediction_probability = model.predict_proba(feature_vector)[0][1]
        
        results = optimize_resources(prediction_probability, in_type, enc_closure, in_cause.strip().lower())

    with col2:
        st.markdown("#### **Optimized Tactical Resource Metrics**")
        score = results["impact_score"]
        st.metric("Algorithmic Congestion Probability Score", f"{score} %")
        
        if score >= 65:
            st.error("🚨 CRITICAL CONGESTION RISK PROFILE DETECTED")
        elif score >= 40:
            st.warning("⚠️ ELEVATED BOTTLE-NECK WARNING GENERATED")
        else:
            st.success("✅ SYSTEM STATUS NOMINAL: ROUTINE OUTAGE MATRIX")
            
        st.progress(score / 100)
        st.markdown("---")
        st.info(f"👮 **Personnel Dispatched:** {results['manpower_required']} Traffic Personnel")
        st.warning(f"🚧 **Targeted Physical Barriers:** {results['barricades_required']} Heavy Barricades")
        st.error(f"🔀 **Active Routing Directives:** {results['diversion_strategy']}")

with tab2:
    st.subheader("Historical Corridor Diagnostic Analysis")
    selected_corridors = st.multiselect("Isolate Metrics by Key Corridor", unique_corridors, default=unique_corridors[:2])
    
    filtered_df = df_raw[df_raw["corridor"].isin(selected_corridors)]
    if not filtered_df.empty:
        st.map(filtered_df, latitude="latitude", longitude="longitude", zoom=11)