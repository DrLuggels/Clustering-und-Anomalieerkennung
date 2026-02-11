import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import plotly.graph_objects as go
import plotly.express as px
from utils.viz import scatter_2d, CLUSTER_COLORS
from utils.explanations import PCA_INTRO, UMAP_INTRO


def render():
    st.header("Dimensionsreduktion")

    if "data" not in st.session_state:
        st.warning("Bitte zuerst Daten in der **Daten-Werkstatt** generieren.")
        return

    df = st.session_state.get("data_processed", st.session_state["data"])
    y = st.session_state["labels"]
    X = df.values
    n_features = X.shape[1]

    if n_features < 3:
        st.info(
            "Die Daten haben nur 2 Features — Dimensionsreduktion ist hier nicht nötig, "
            "aber du kannst trotzdem sehen, wie PCA und UMAP die Daten transformieren."
        )

    # Controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        st.markdown(PCA_INTRO)
        max_pca_comp = min(3, n_features)
        if max_pca_comp > 2:
            n_components_pca = st.slider("PCA: Komponenten", 2, max_pca_comp, 2, key="pca_comp")
        else:
            n_components_pca = 2
            st.caption("PCA: 2 Komponenten (nur 2 Features vorhanden)")

    with col_ctrl2:
        st.markdown(UMAP_INTRO)
        umap_neighbors = st.slider("UMAP: n_neighbors", 5, 50, 15, key="umap_nn")
        umap_min_dist = st.slider("UMAP: min_dist", 0.0, 1.0, 0.1, step=0.05, key="umap_md")
        umap_metric = st.selectbox("UMAP: metric", ["euclidean", "manhattan", "cosine"],
                                   key="umap_metric")

    st.divider()

    # --- PCA ---
    pca = PCA(n_components=n_components_pca)
    X_pca = pca.fit_transform(X)
    pca_cols = [f"PC{i+1}" for i in range(n_components_pca)]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)

    # --- UMAP ---
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2, n_neighbors=umap_neighbors,
            min_dist=umap_min_dist, metric=umap_metric,
            random_state=42,
        )
        X_umap = reducer.fit_transform(X)
        df_umap = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
        umap_available = True
    except ImportError:
        st.error("UMAP nicht installiert. Bitte `pip install umap-learn` ausführen.")
        umap_available = False
        df_umap = None

    # --- Side by Side ---
    st.subheader("PCA vs. UMAP")
    col_pca, col_umap = st.columns(2)

    with col_pca:
        fig_pca = scatter_2d(df_pca, y, title="PCA", col_x=pca_cols[0], col_y=pca_cols[1])
        st.plotly_chart(fig_pca, width="stretch")

    with col_umap:
        if umap_available:
            fig_umap = scatter_2d(df_umap, y, title="UMAP", col_x="UMAP1", col_y="UMAP2")
            st.plotly_chart(fig_umap, width="stretch")
        else:
            st.warning("UMAP nicht verfügbar.")

    st.divider()

    # --- PCA Details ---
    st.subheader("PCA Details")
    col_var, col_biplot = st.columns(2)

    with col_var:
        # Explained Variance
        pca_full = PCA().fit(X)
        explained = pca_full.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(explained))],
            y=explained,
            name="Einzeln",
            marker_color="#2196F3",
        ))
        fig_var.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(cumulative))],
            y=cumulative,
            mode="lines+markers",
            name="Kumulativ",
            line=dict(color="#FF9800", width=3),
        ))
        fig_var.update_layout(
            title="Explained Variance Ratio",
            yaxis_title="Varianzanteil",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_var, width="stretch")
        st.caption(f"Die ersten {n_components_pca} Komponenten erklären "
                   f"**{sum(explained[:n_components_pca])*100:.1f}%** der Gesamtvarianz.")

    with col_biplot:
        # Biplot (loadings as arrows)
        if n_components_pca >= 2:
            loadings = pca.components_.T  # (n_features, n_components)
            fig_bi = scatter_2d(df_pca, y, title="PCA Biplot (Ladungsvektoren)",
                                col_x=pca_cols[0], col_y=pca_cols[1], show_legend=False)

            # Scale loadings for visibility
            max_range = max(np.abs(X_pca[:, :2]).max(), 1)
            scale = max_range * 0.8

            for i in range(min(n_features, 10)):  # max 10 arrows
                fig_bi.add_annotation(
                    x=loadings[i, 0] * scale,
                    y=loadings[i, 1] * scale,
                    ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3, arrowsize=1.5, arrowwidth=2,
                    arrowcolor="#FF5722",
                )
                fig_bi.add_annotation(
                    x=loadings[i, 0] * scale * 1.15,
                    y=loadings[i, 1] * scale * 1.15,
                    text=df.columns[i],
                    showarrow=False,
                    font=dict(size=11, color="#FF5722"),
                )
            st.plotly_chart(fig_bi, width="stretch")

    # --- Trustworthiness ---
    st.subheader("Qualitätsvergleich")
    col_t1, col_t2 = st.columns(2)

    trust_pca = trustworthiness(X, X_pca, n_neighbors=5)
    col_t1.metric("Trustworthiness PCA", f"{trust_pca:.4f}")

    if umap_available:
        trust_umap = trustworthiness(X, X_umap, n_neighbors=5)
        col_t2.metric("Trustworthiness UMAP", f"{trust_umap:.4f}")

    st.caption(
        "**Trustworthiness** misst, wie gut die lokale Nachbarschaftsstruktur "
        "in der Reduktion erhalten bleibt (1.0 = perfekt)."
    )

    # Store reduced data
    st.session_state["data_pca"] = df_pca
    st.session_state["data_umap"] = df_umap if umap_available else None
