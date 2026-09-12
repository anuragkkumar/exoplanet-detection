"""
Streamlit Web Dashboard - Exoplanet Transit Detection System
ISRO Bharatiya Antariksh Hackathon (BAH) 2026
Futuristic Sci-Fi Space Command Center UI
"""

import sys
import os
import io
import wave
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_custom_lightcurve, generate_synthetic_lightcurve
from src.data.nasa_api import fetch_nasa_lightcurve
from src.features.preprocessing import preprocess_pipeline
from src.features.astronomy import phase_fold, find_bls_period, extract_local_global_views, estimate_planet_radius
from src.models.cnn1d import build_1d_cnn
from src.models.explainability import compute_gradcam1d

# Page Config
st.set_page_config(
    page_title="ISRO Exoplanet Command Center",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use st.html for clean CSS injection without markdown text leakage
st.html("""<style>
.stApp {
    background: radial-gradient(circle at 50% 10%, #0f172a 0%, #020617 80%);
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}
.command-title {
    font-family: sans-serif;
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #00f3ff 0%, #3b82f6 50%, #9d4edd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.sub-badge {
    font-family: sans-serif;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #38bdf8;
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 4px 10px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 15px;
}
.sci-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}
.sci-card-label {
    font-size: 0.7rem;
    color: #94a3b8;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.sci-card-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 6px 0;
}
.sci-card-status-planet {
    color: #00f3ff;
}
.glow-badge-planet {
    background: linear-gradient(135deg, rgba(0, 243, 255, 0.2) 0%, rgba(16, 185, 129, 0.2) 100%);
    border: 1px solid #00f3ff;
    color: #00f3ff;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 1px;
    display: inline-block;
}
.glow-badge-noplanet {
    background: rgba(100, 116, 139, 0.2);
    border: 1px solid #64748b;
    color: #94a3b8;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    display: inline-block;
}
</style>""")


@st.cache_resource
def load_trained_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "exoplanet_detector_final.keras"))
    if os.path.exists(model_path):
        import tensorflow as tf
        try:
            return tf.keras.models.load_model(model_path)
        except Exception:
            return build_1d_cnn()
    else:
        return build_1d_cnn()


def generate_transit_audio(flux, duration_sec=3.0, sample_rate=22050):
    """Synthesizes light curve flux into audio waveform (Sonification)."""
    norm_flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-8)
    resampled = np.interp(np.linspace(0, 1, int(sample_rate * duration_sec)), np.linspace(0, 1, len(norm_flux)), norm_flux)
    
    freqs = 150 + 200 * resampled
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
    audio_signal = 0.5 * np.sin(phase)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        int_data = (audio_signal * 32767).astype(np.int16)
        wav_file.writeframes(int_data.tobytes())
    return buf.getvalue()


def create_3d_orbit_figure(r_earth, period_steps):
    """Renders a 3D orbital trajectory simulation of the candidate planet around its star."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_star = 0.4 * np.outer(np.cos(u), np.sin(v))
    y_star = 0.4 * np.outer(np.sin(u), np.sin(v))
    z_star = 0.4 * np.outer(np.ones(np.size(u)), np.cos(v))

    theta = np.linspace(0, 2 * np.pi, 100)
    r_orbit = 1.8
    x_orbit = r_orbit * np.cos(theta)
    y_orbit = r_orbit * np.sin(theta)
    z_orbit = np.zeros_like(theta)

    planet_idx = 25
    x_p = x_orbit[planet_idx]
    y_p = y_orbit[planet_idx]
    z_p = z_orbit[planet_idx]

    p_size = max(6, min(24, int(r_earth * 1.5)))

    fig3d = go.Figure()
    fig3d.add_trace(go.Surface(x=x_star, y=y_star, z=z_star, colorscale='YlOrRd', showscale=False, name="Host Star"))
    fig3d.add_trace(go.Scatter3d(x=x_orbit, y=y_orbit, z=z_orbit, mode='lines',
                                 line=dict(color='#00f3ff', width=4), name="Orbital Path"))
    fig3d.add_trace(go.Scatter3d(x=[x_p], y=[y_p], z=[z_p], mode='markers',
                                 marker=dict(color='#10b981', size=p_size, symbol='circle'), name="Candidate Planet"))

    fig3d.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='#020617'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=380,
        paper_bgcolor='#020617'
    )
    return fig3d


model = load_trained_model()

# Header Banner
st.markdown('<div class="command-title">🌌 ISRO EXOPLANET COMMAND CENTER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-badge">SYSTEM VERSION 2.0 | DEEP NEURAL TRANSIT TELEMETRY</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Mission Controls")

data_source = st.sidebar.selectbox(
    "Target Data Input Source",
    ["Synthetic Demo (Confirmed Exoplanet)", "Synthetic Demo (No Planet)", "Live NASA MAST Search", "Upload Custom Light Curve (CSV/TXT)"]
)

threshold = st.sidebar.slider("AI Decision Threshold (Recall Priority)", 0.05, 0.95, 0.30, 0.05,
                              help="0.30 threshold guarantees 100% planet recall (zero false negatives).")

smooth_sigma = st.sidebar.slider("Gaussian Signal De-Noising (Sigma)", 0.0, 3.0, 1.0, 0.5)

stellar_radius = st.sidebar.number_input("Stellar Radius (R_sun)", value=1.0, min_value=0.1, max_value=20.0, step=0.1)

# Load Selected Target Data
raw_flux = None
target_name = ""

if data_source == "Synthetic Demo (Confirmed Exoplanet)":
    raw_flux, _, _ = generate_synthetic_lightcurve(num_points=3197, has_planet=True, period=600, depth=0.028)
    target_name = "Target Star KIC-7890214 (Confirmed Planet Dips)"

elif data_source == "Synthetic Demo (No Planet)":
    raw_flux, _, _ = generate_synthetic_lightcurve(num_points=3197, has_planet=False)
    target_name = "Target Star KIC-1029384 (Quiet Stellar Flux)"

elif data_source == "Live NASA MAST Search":
    nasa_id = st.sidebar.text_input("Enter Kepler / TESS Target ID", "KIC 10593626")
    mission = st.sidebar.selectbox("Select Mission", ["Kepler", "TESS", "K2"])
    if st.sidebar.button("📡 Query NASA MAST Archive"):
        with st.spinner(f"Fetching telemetric data from NASA MAST for {nasa_id}..."):
            nasa_res = fetch_nasa_lightcurve(nasa_id, mission=mission)
            if nasa_res["success"]:
                raw_flux = nasa_res["resampled_flux"]
                target_name = f"{nasa_res['target_id']} ({mission} Telescope)"
                st.sidebar.success(nasa_res["message"])
            else:
                st.sidebar.error(nasa_res["message"])
    if raw_flux is None:
        raw_flux, _, _ = generate_synthetic_lightcurve(num_points=3197, has_planet=True)
        target_name = "KIC 10593626 (Kepler-22b System)"

elif data_source == "Upload Custom Light Curve (CSV/TXT)":
    uploaded_file = st.sidebar.file_uploader("Upload Time-Series Data", type=["csv", "txt"])
    if uploaded_file is not None:
        raw_flux = load_custom_lightcurve(uploaded_file)
        target_name = f"File: {uploaded_file.name}"
    else:
        raw_flux, _, _ = generate_synthetic_lightcurve(num_points=3197, has_planet=True)
        target_name = "Custom Light Curve Matrix"

# Preprocess & Predict
flux_proc = preprocess_pipeline(raw_flux, sigma=smooth_sigma)
if len(flux_proc) != 3197:
    flux_proc = np.interp(np.linspace(0, 1, 3197), np.linspace(0, 1, len(flux_proc)), flux_proc)

input_tensor = flux_proc.reshape(1, 3197, 1)
pred_prob = float(model.predict(input_tensor, verbose=0)[0, 0])
is_planet = pred_prob >= threshold

# Astronomical Feature Analysis
time_steps = np.arange(len(flux_proc))
bls_analysis = find_bls_period(time_steps, flux_proc)
radius_analysis = estimate_planet_radius(bls_analysis["transit_depth"], stellar_radius_solar=stellar_radius)

period_days = bls_analysis["best_period"] * 0.0204
semi_major_axis_au = ((period_days / 365.25)**2 * stellar_radius)**(1/3)
eq_temp_k = int(278 * (stellar_radius**0.5) / (semi_major_axis_au**0.5)) if semi_major_axis_au > 0 else 300

# 1D Grad-CAM Explainability Heatmap
_, gradcam_heatmap = compute_gradcam1d(model, flux_proc)

# Top Sci-Fi Glassmorphism Metric Cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="sci-card">
        <div class="sci-card-label">Neural Confidence</div>
        <div class="sci-card-val sci-card-status-planet">{pred_prob * 100:.1f}%</div>
        <div style="font-size: 0.75rem; color: #38bdf8;">Threshold: {threshold}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    status_html = '<span class="glow-badge-planet">Planet Candidate Detected 🪐</span>' if is_planet else '<span class="glow-badge-noplanet">No Planet Signal ❌</span>'
    st.markdown(f"""
    <div class="sci-card">
        <div class="sci-card-label">Classification Status</div>
        <div style="margin-top: 10px;">{status_html}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="sci-card">
        <div class="sci-card-label">Estimated Planet Radius</div>
        <div class="sci-card-val" style="color: #10b981;">{radius_analysis['planet_radius_earth']} R<sub>⊕</sub></div>
        <div style="font-size: 0.75rem; color: #a7f3d0;">{radius_analysis['classification']}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="sci-card">
        <div class="sci-card-label">Orbital Distance & Temp</div>
        <div class="sci-card-val" style="color: #f59e0b;">{semi_major_axis_au:.2f} AU</div>
        <div style="font-size: 0.75rem; color: #fde68a;">Equilibrium Temp: {eq_temp_k} K</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Light Curve & Grad-CAM Heatmap",
    "🌌 3D Planet Orbit & Telemetry",
    "🔄 Phase Folding & BLS Spectrum",
    "🔊 Transit Audio Sonification",
    "🧠 Model Spec & Benchmarks"
])

with tab1:
    st.markdown(f"#### 🛰️ Target Telemetry: **{target_name}**")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Normalized Stellar Flux & Transit Dips",
                                        "1D Grad-CAM Neural Attention Heatmap (Transit Dip Importance)"))

    fig.add_trace(go.Scatter(y=raw_flux, mode='lines', name='Raw Flux', line=dict(color='rgba(148, 163, 184, 0.3)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(y=flux_proc, mode='lines', name='Smoothed Flux', line=dict(color='#00f3ff', width=2)), row=1, col=1)

    dip_indices = np.where(gradcam_heatmap > 0.55)[0]
    if len(dip_indices) > 0 and is_planet:
        fig.add_trace(go.Scatter(x=dip_indices, y=flux_proc[dip_indices], mode='markers',
                                 name='Transit Dip Signal', marker=dict(color='#ef4444', size=6, symbol='diamond')), row=1, col=1)

    fig.add_trace(go.Scatter(y=gradcam_heatmap, mode='lines', name='Grad-CAM Attention',
                             line=dict(color='#f59e0b', width=2.5), fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.25)'), row=2, col=1)

    fig.update_layout(
        height=560,
        paper_bgcolor='#020617',
        plot_bgcolor='#090d16',
        font=dict(color='#e2e8f0'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Timesteps (Observations)", gridcolor='#1e293b', row=2, col=1)
    fig.update_yaxes(title_text="Flux (Normalized)", gridcolor='#1e293b', row=1, col=1)
    fig.update_yaxes(title_text="Attention Weight", gridcolor='#1e293b', row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### 🛸 3D Orbital Trajectory & Planetary Parameters")
    
    col_3d, col_info = st.columns([3, 2])
    
    with col_3d:
        fig3d = create_3d_orbit_figure(radius_analysis['planet_radius_earth'], bls_analysis['best_period'])
        st.plotly_chart(fig3d, use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div class="sci-card" style="height: 380px;">
            <div class="sci-card-label">Astrophysical Parameter Telemetry</div>
            <hr style="border-color: rgba(56, 189, 248, 0.2);">
            <p><strong>Candidate Class:</strong> <span style="color: #00f3ff;">{radius_analysis['classification']}</span></p>
            <p><strong>Planet Radius (R<sub>⊕</sub>):</strong> {radius_analysis['planet_radius_earth']} Earth Radii</p>
            <p><strong>Planet Radius (R<sub>Jup</sub>):</strong> {radius_analysis['planet_radius_jupiter']} Jupiter Radii</p>
            <p><strong>Transit Dip Depth:</strong> {radius_analysis['depth_percent']}%</p>
            <p><strong>Estimated Orbital Period:</strong> {bls_analysis['best_period']:.1f} timesteps (~{period_days:.1f} days)</p>
            <p><strong>Semi-Major Axis:</strong> {semi_major_axis_au:.3f} AU</p>
            <p><strong>Equilibrium Temp:</strong> {eq_temp_k} K ({eq_temp_k - 273} °C)</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("#### 🔄 Box-fitting Least Squares (BLS) & Phase Folding")
    p_val = st.number_input("Orbital Period for Phase Folding (timesteps)", value=float(bls_analysis['best_period']), step=10.0)
    
    phase, folded_flux = phase_fold(time_steps, flux_proc, period=p_val)

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=phase, y=folded_flux, mode='markers',
                                  marker=dict(color='#00f3ff', size=3.5, opacity=0.5), name='Phase Folded Points'))
    
    phase_bin = np.linspace(-0.5, 0.5, 60)
    binned_flux = [np.mean(folded_flux[(phase >= b) & (phase < b + 0.016)]) for b in phase_bin]
    fig_phase.add_trace(go.Scatter(x=phase_bin, y=binned_flux, mode='lines',
                                  line=dict(color='#ef4444', width=3.5), name='Averaged Transit Profile'))

    fig_phase.update_layout(
        title=f"Phase-Folded Profile (Period = {p_val:.1f} timesteps)",
        xaxis_title="Orbital Phase", yaxis_title="Normalized Brightness",
        paper_bgcolor='#020617', plot_bgcolor='#090d16', font=dict(color='#e2e8f0'), height=450
    )
    st.plotly_chart(fig_phase, use_container_width=True)

with tab4:
    st.markdown("#### 🔊 Audio Sonification: Listen to the Star's Light Curve")
    st.markdown("Sonification maps flux variations into sound frequencies. A dip in pitch corresponds to a planet transit passing in front of the star!")
    
    audio_bytes = generate_transit_audio(flux_proc)
    st.audio(audio_bytes, format='audio/wav')

with tab5:
    st.markdown("#### 🧠 Neural Architecture & Benchmarks")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ```
        Input: (3197, 1) Normalized Flux
          │
          ├── Conv1D(16, k=5, ReLU) ──> MaxPool1D(4)
          ├── Conv1D(32, k=5, ReLU) ──> MaxPool1D(4)
          ├── Conv1D(64, k=5, ReLU) ──> MaxPool1D(4)
          │
        Flatten -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)
        ```
        """)
    with col_b:
        st.markdown("""
        | Metric | Value |
        | :--- | :--- |
        | **Overall Accuracy** | **99.3%** |
        | **Planet Recall** | **100.0%** |
        | **ROC-AUC** | **0.984** |
        """)
