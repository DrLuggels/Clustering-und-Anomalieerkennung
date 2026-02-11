import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from utils.viz import scatter_2d
from utils.metrics import pairwise_distances_for_points
from utils.explanations import SCALING_INTRO, SCALER_DESCRIPTIONS


SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


def render():
    st.header("Skalierungs-Labor")
    st.markdown(SCALING_INTRO)

    if "data" not in st.session_state:
        st.warning("Bitte zuerst Daten in der **Daten-Werkstatt** generieren.")
        return

    df = st.session_state["data"]
    y = st.session_state["labels"]

    # Scaler selection
    scaler_name = st.radio(
        "Skalierungsmethode",
        list(SCALERS.keys()),
        horizontal=True,
    )
    st.caption(SCALER_DESCRIPTIONS[scaler_name])

    # Apply scaling
    scaler = SCALERS[scaler_name]()
    X_scaled = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(X_scaled, columns=df.columns)

    # --- Side by Side Scatter ---
    st.subheader("Vorher / Nachher")
    col_left, col_right = st.columns(2)

    with col_left:
        fig = scatter_2d(df, y, title="Original (unskaliert)", show_legend=False)
        st.plotly_chart(fig, width="stretch")

    with col_right:
        fig = scatter_2d(df_scaled, y, title=f"Skaliert ({scaler_name})", show_legend=False)
        st.plotly_chart(fig, width="stretch")

    # --- Statistics Table ---
    st.subheader("Feature-Statistiken")

    stats_before = df.describe().loc[["mean", "std", "min", "max"]].T
    stats_before.columns = ["Mean (vorher)", "Std (vorher)", "Min (vorher)", "Max (vorher)"]

    stats_after = df_scaled.describe().loc[["mean", "std", "min", "max"]].T
    stats_after.columns = ["Mean (nachher)", "Std (nachher)", "Min (nachher)", "Max (nachher)"]

    stats_combined = pd.concat([stats_before, stats_after], axis=1)
    stats_combined = stats_combined.round(4)
    st.dataframe(stats_combined, width="stretch")

    # --- Distance Impact ---
    st.subheader("Auswirkung auf Distanzen")
    st.caption("Vergleich der Distanzen zwischen Punkt 0 und Punkt 1 — vorher vs. nachher:")

    if len(df) >= 2:
        dists_before = pairwise_distances_for_points(df.values, 0, 1)
        dists_after = pairwise_distances_for_points(X_scaled, 0, 1)

        col1, col2, col3 = st.columns(3)
        for col, metric in zip([col1, col2, col3], ["Euklidisch", "Manhattan", "Cosinus-Ähnlichkeit"]):
            before = dists_before[metric]
            after = dists_after[metric]
            if abs(before) > 1e-10:
                change = ((after - before) / abs(before)) * 100
                col.metric(
                    metric,
                    f"{after:.4f}",
                    f"{change:+.1f}% (vorher: {before:.4f})",
                )
            else:
                col.metric(metric, f"{after:.4f}", f"vorher: {before:.4f}")

    # --- Use scaled data toggle ---
    st.divider()
    use_scaled = st.toggle("Skalierte Daten für weitere Tabs verwenden", value=True)
    if use_scaled:
        st.session_state["data_processed"] = df_scaled
        st.session_state["scaling_active"] = scaler_name
        st.success(f"Skalierte Daten ({scaler_name}) werden in den folgenden Tabs verwendet.")
    else:
        st.session_state["data_processed"] = df
        st.session_state["scaling_active"] = None
        st.info("Originaldaten werden in den folgenden Tabs verwendet.")
