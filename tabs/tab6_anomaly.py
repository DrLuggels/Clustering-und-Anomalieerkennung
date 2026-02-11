import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.viz import anomaly_scatter, scatter_2d, ANOMALY_COLOR, NORMAL_COLOR
from utils.explanations import (
    ANOMALY_CLUSTER_INTRO, ANOMALY_IFOREST_INTRO,
    ANOMALY_LOF_INTRO, ANOMALY_DBSCAN_INTRO,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    st.header("Anomalie-Detektor")

    if "data" not in st.session_state:
        st.warning("Bitte zuerst Daten in der **Daten-Werkstatt** generieren.")
        return

    df = st.session_state.get("data_processed", st.session_state["data"])
    y_true = st.session_state["labels"]
    outlier_mask_true = st.session_state.get("outlier_mask", np.zeros(len(df), dtype=bool))
    X = df.values

    # Use reduced data for visualization
    df_viz = st.session_state.get("data_pca", df)
    if df_viz is None:
        df_viz = df
    viz_cols = df_viz.columns.tolist()

    has_ground_truth = outlier_mask_true.any()
    if has_ground_truth:
        st.success(f"Ground Truth vorhanden: {int(outlier_mask_true.sum())} injizierte Outlier.")
    else:
        st.info("Keine Outlier injiziert. Metriken wie Precision/Recall sind nicht verfügbar.")

    # ========================
    # PART A: Cluster-basiert
    # ========================
    st.subheader("A) Cluster-basierte Anomalieerkennung")
    st.markdown(ANOMALY_CLUSTER_INTRO)

    if "km_model" in st.session_state:
        km = st.session_state["km_model"]
        distances = np.min(
            np.linalg.norm(X[:, np.newaxis] - km.cluster_centers_[np.newaxis, :], axis=2),
            axis=1,
        )
        threshold_pct = st.slider("Threshold (Perzentil)", 80, 99, 95, key="cluster_thresh")
        threshold = np.percentile(distances, threshold_pct)
        anomaly_cluster = distances > threshold

        fig = anomaly_scatter(df_viz, distances, anomaly_cluster,
                              title=f"Cluster-basiert (Threshold: {threshold_pct}. Perzentil)",
                              col_x=viz_cols[0], col_y=viz_cols[1])
        st.plotly_chart(fig, use_container_width=True)

        if has_ground_truth:
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{precision_score(outlier_mask_true, anomaly_cluster, zero_division=0):.3f}")
            col2.metric("Recall", f"{recall_score(outlier_mask_true, anomaly_cluster, zero_division=0):.3f}")
            col3.metric("F1-Score", f"{f1_score(outlier_mask_true, anomaly_cluster, zero_division=0):.3f}")
    else:
        st.warning("Bitte zuerst **Clustering** in Tab 5 durchführen, um K-Means-Zentroide zu erhalten.")
        anomaly_cluster = np.zeros(len(X), dtype=bool)

    st.divider()

    # ========================
    # PART B: Spezialisierte Algorithmen
    # ========================
    st.subheader("B) Spezialisierte Algorithmen")

    col_if, col_lof, col_dbscan = st.columns(3)

    # --- Isolation Forest ---
    with col_if:
        st.markdown(ANOMALY_IFOREST_INTRO)
        contamination = st.slider("Contamination", 0.01, 0.20, 0.05, step=0.01, key="if_cont")
        n_estimators = st.slider("n_estimators", 50, 300, 100, step=50, key="if_est")

    iforest = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                               random_state=42)
    iforest_labels = iforest.fit_predict(X)
    anomaly_iforest = iforest_labels == -1
    iforest_scores = -iforest.decision_function(X)  # Higher = more anomalous

    # --- LOF ---
    with col_lof:
        st.markdown(ANOMALY_LOF_INTRO)
        lof_neighbors = st.slider("n_neighbors", 5, 50, 20, key="lof_nn")

    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=contamination)
    lof_labels = lof.fit_predict(X)
    anomaly_lof = lof_labels == -1
    lof_scores = -lof.negative_outlier_factor_

    # --- DBSCAN ---
    with col_dbscan:
        st.markdown(ANOMALY_DBSCAN_INTRO)
        eps = st.slider("eps", 0.1, 5.0, 0.5, step=0.1, key="db_eps")
        min_samples = st.slider("min_samples", 2, 20, 5, key="db_ms")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X)
    anomaly_dbscan = dbscan_labels == -1

    st.divider()

    # ========================
    # PART C: 4er-Grid Vergleich
    # ========================
    st.subheader("C) Methoden-Vergleich")

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Cluster-basiert", "Isolation Forest",
                                        "Local Outlier Factor", "DBSCAN"])

    methods = [
        (anomaly_cluster, "Cluster-basiert"),
        (anomaly_iforest, "Isolation Forest"),
        (anomaly_lof, "LOF"),
        (anomaly_dbscan, "DBSCAN"),
    ]

    for idx, (mask, name) in enumerate(methods):
        row = idx // 2 + 1
        col = idx % 2 + 1
        normal = ~mask

        fig.add_trace(go.Scatter(
            x=df_viz[viz_cols[0]][normal], y=df_viz[viz_cols[1]][normal],
            mode="markers",
            marker=dict(size=5, color=NORMAL_COLOR, opacity=0.5),
            name="Normal",
            showlegend=(idx == 0),
            legendgroup="normal",
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=df_viz[viz_cols[0]][mask], y=df_viz[viz_cols[1]][mask],
            mode="markers",
            marker=dict(size=8, color=ANOMALY_COLOR, symbol="x"),
            name="Anomalie",
            showlegend=(idx == 0),
            legendgroup="anomaly",
        ), row=row, col=col)

    fig.update_layout(template="plotly_white", height=800, margin=dict(t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- Overlap Analysis ---
    st.subheader("Overlap-Analyse")
    st.caption("Welche Methoden stimmen überein? Zelle (i,j) = Anteil gemeinsam erkannter Anomalien.")

    method_names = ["Cluster", "IForest", "LOF", "DBSCAN"]
    method_masks = [anomaly_cluster, anomaly_iforest, anomaly_lof, anomaly_dbscan]

    overlap_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if method_masks[i].sum() > 0 and method_masks[j].sum() > 0:
                intersection = (method_masks[i] & method_masks[j]).sum()
                union = (method_masks[i] | method_masks[j]).sum()
                overlap_matrix[i, j] = intersection / union if union > 0 else 0
            elif i == j:
                overlap_matrix[i, j] = 1.0

    fig_overlap = go.Figure(data=go.Heatmap(
        z=overlap_matrix,
        x=method_names,
        y=method_names,
        colorscale="YlOrRd",
        zmin=0, zmax=1,
        text=np.round(overlap_matrix, 3),
        texttemplate="%{text:.3f}",
        hovertemplate="%{y} ∩ %{x}<br>Jaccard: %{z:.3f}<extra></extra>",
    ))
    fig_overlap.update_layout(
        title="Jaccard-Übereinstimmung zwischen Methoden",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_overlap, use_container_width=True)

    # --- Metrics Table ---
    if has_ground_truth:
        st.subheader("Metriken vs. Ground Truth")

        metrics_rows = []
        for name, mask in zip(method_names, method_masks):
            p = precision_score(outlier_mask_true, mask, zero_division=0)
            r = recall_score(outlier_mask_true, mask, zero_division=0)
            f = f1_score(outlier_mask_true, mask, zero_division=0)
            n_detected = int(mask.sum())
            metrics_rows.append({
                "Methode": name,
                "Erkannte Anomalien": n_detected,
                "Precision": f"{p:.3f}",
                "Recall": f"{r:.3f}",
                "F1-Score": f"{f:.3f}",
            })

        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        best_f1_idx = np.argmax([f1_score(outlier_mask_true, m, zero_division=0) for m in method_masks])
        st.success(
            f"**Beste Methode (F1):** {method_names[best_f1_idx]} — "
            f"Kein einzelner Algorithmus ist immer der Beste. "
            f"In der Praxis kombiniert man oft mehrere Methoden (Ensemble)."
        )
    else:
        st.subheader("Erkennungs-Übersicht")
        for name, mask in zip(method_names, method_masks):
            st.write(f"**{name}:** {int(mask.sum())} Anomalien erkannt")
