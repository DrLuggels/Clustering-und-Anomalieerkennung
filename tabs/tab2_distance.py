import streamlit as st
import numpy as np
from utils.metrics import compute_distance_matrices, pairwise_distances_for_points, distance_comparison_stats
from utils.viz import heatmap, scatter_2d
from utils.explanations import DISTANCE_EUCLIDEAN, DISTANCE_MANHATTAN, DISTANCE_COSINE


def render():
    st.header("Distanz-Explorer")

    if "data" not in st.session_state:
        st.warning("Bitte zuerst Daten in der **Daten-Werkstatt** generieren.")
        return

    df = st.session_state["data"]
    y = st.session_state["labels"]
    X = df.values

    # Theory section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(DISTANCE_EUCLIDEAN)
    with col2:
        st.markdown(DISTANCE_MANHATTAN)
    with col3:
        st.markdown(DISTANCE_COSINE)

    st.divider()

    # --- Point comparison ---
    st.subheader("Punkt-Vergleich")
    st.caption("Wähle zwei Punkte und vergleiche die Distanzen mit allen drei Metriken.")

    n_points = len(df)
    col_a, col_b = st.columns(2)
    with col_a:
        idx_a = st.number_input("Punkt A (Index)", 0, n_points - 1, 0)
    with col_b:
        idx_b = st.number_input("Punkt B (Index)", 0, n_points - 1, min(1, n_points - 1))

    if idx_a != idx_b:
        dists = pairwise_distances_for_points(X, idx_a, idx_b)
        col1, col2, col3 = st.columns(3)
        col1.metric("Euklidisch", f"{dists['Euklidisch']:.4f}")
        col2.metric("Manhattan", f"{dists['Manhattan']:.4f}")
        col3.metric("Cosinus-Ähnlichkeit", f"{dists['Cosinus-Ähnlichkeit']:.4f}")

        # Show the two points on a scatter
        fig = scatter_2d(df, y, title="Ausgewählte Punkte", show_legend=False)
        import plotly.graph_objects as go
        cols = df.columns.tolist()
        fig.add_trace(go.Scatter(
            x=[df.iloc[idx_a][cols[0]], df.iloc[idx_b][cols[0]]],
            y=[df.iloc[idx_a][cols[1]], df.iloc[idx_b][cols[1]]],
            mode="markers+lines",
            marker=dict(size=15, color="red", symbol="diamond"),
            line=dict(color="red", width=2, dash="dash"),
            name=f"A({idx_a}) → B({idx_b})",
        ))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Bitte zwei **verschiedene** Punkte wählen.")

    st.divider()

    # --- Distance Matrix Heatmaps ---
    st.subheader("Distanzmatrix-Heatmaps")

    # Subsample for performance
    max_display = st.slider("Punkte für Heatmap (Subsample)", 10, min(100, n_points), min(50, n_points))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_points, size=max_display, replace=False)
    sample_idx.sort()
    X_sub = X[sample_idx]

    matrices = compute_distance_matrices(X_sub)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = heatmap(matrices["Euklidisch"], title="Euklidisch", colorscale="Blues")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = heatmap(matrices["Manhattan"], title="Manhattan", colorscale="Greens")
        st.plotly_chart(fig, width="stretch")
    with col3:
        fig = heatmap(matrices["Cosinus-Ähnlichkeit"], title="Cosinus-Ähnlichkeit",
                      colorscale="RdYlBu", zmin=-1, zmax=1)
        st.plotly_chart(fig, width="stretch")

    # --- Statistics ---
    st.subheader("Vergleichs-Statistiken")
    stats = distance_comparison_stats(X_sub)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Euklidisch (Ø)", f"{stats['euc_mean']:.3f}", f"σ = {stats['euc_std']:.3f}")
    with col2:
        st.metric("Manhattan (Ø)", f"{stats['man_mean']:.3f}", f"σ = {stats['man_std']:.3f}")
    with col3:
        st.metric("Cosinus (Ø)", f"{stats['cos_mean']:.3f}", f"σ = {stats['cos_std']:.3f}")

    st.info(
        f"Manhattan-Distanzen sind im Schnitt **{stats['man_euc_ratio']:.2f}x** "
        f"größer als Euklidische. In höheren Dimensionen wächst dieser Faktor — "
        f"bekannt als *Curse of Dimensionality*."
    )
