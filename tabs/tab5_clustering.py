import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage
from kneed import KneeLocator
from utils.viz import (
    elbow_chart, silhouette_chart, silhouette_diagram,
    comparison_scatter_side_by_side, dendrogram_plot, scatter_2d, CLUSTER_COLORS,
)
from utils.explanations import ELBOW_INTRO, SILHOUETTE_INTRO, KMEANS_INTRO, AGGLOM_INTRO
import plotly.graph_objects as go


def render():
    st.header("Clustering-Arena")

    if "data" not in st.session_state:
        st.warning("Bitte zuerst Daten in der **Daten-Werkstatt** generieren.")
        return

    df = st.session_state.get("data_processed", st.session_state["data"])
    y_true = st.session_state["labels"]
    X = df.values

    # Use reduced data for visualization if available
    df_viz = st.session_state.get("data_pca", df)
    if df_viz is None:
        df_viz = df

    # ========================
    # PART A: Optimale k finden
    # ========================
    st.subheader("A) Optimale Cluster-Anzahl finden")

    col_elbow, col_sil = st.columns(2)

    k_range = range(2, 11)

    # Elbow
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    # Find knee
    try:
        kl = KneeLocator(list(k_range), inertias, curve="convex", direction="decreasing")
        knee_k = kl.knee
    except Exception:
        knee_k = None

    best_sil_k = list(k_range)[np.argmax(sil_scores)]

    with col_elbow:
        st.markdown(ELBOW_INTRO)
        fig_elbow = elbow_chart(k_range, inertias, knee_k)
        st.plotly_chart(fig_elbow, width="stretch")

    with col_sil:
        st.markdown(SILHOUETTE_INTRO)
        fig_sil = silhouette_chart(k_range, sil_scores, best_sil_k)
        st.plotly_chart(fig_sil, width="stretch")

    # Recommendation
    rec_k = knee_k if knee_k else best_sil_k
    st.info(
        f"**Empfehlung:** Elbow → k={knee_k or '?'} | "
        f"Silhouette → k={best_sil_k} | "
        f"Ground Truth → k={st.session_state.get('n_clusters_true', '?')}"
    )

    st.divider()

    # ========================
    # PART B: Algorithmus-Vergleich
    # ========================
    st.subheader("B) K-Means vs. Agglomeratives Clustering")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        st.markdown(KMEANS_INTRO)
        k_chosen = st.slider("Cluster-Anzahl (k)", 2, 10, rec_k or 3, key="k_cluster")
        n_init = st.slider("K-Means: n_init (Neustarts)", 1, 30, 10)

    with col_ctrl2:
        st.markdown(AGGLOM_INTRO)
        linkage_method = st.selectbox("Linkage-Methode",
                                       ["ward", "complete", "average", "single"])

    # Run K-Means
    t0 = time.time()
    km = KMeans(n_clusters=k_chosen, n_init=n_init, random_state=42)
    km_labels = km.fit_predict(X)
    km_time = time.time() - t0

    # Run Agglomerative
    t0 = time.time()
    agg = AgglomerativeClustering(n_clusters=k_chosen, linkage=linkage_method)
    agg_labels = agg.fit_predict(X)
    agg_time = time.time() - t0

    # Side by side visualization
    fig_compare = comparison_scatter_side_by_side(
        df_viz, km_labels, agg_labels,
        title_left=f"K-Means (k={k_chosen})",
        title_right=f"Agglomerativ ({linkage_method}, k={k_chosen})",
    )
    st.plotly_chart(fig_compare, width="stretch")

    # K-Means: show centroids
    st.subheader("K-Means: Zentroide")
    if df_viz.shape[1] >= 2:
        # Transform centroids to viz space if PCA was used
        if "data_pca" in st.session_state and st.session_state["data_pca"] is not None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2).fit(X)
            centroids_viz = pca.transform(km.cluster_centers_)
        else:
            centroids_viz = km.cluster_centers_[:, :2]

        viz_cols = df_viz.columns.tolist()
        fig_km = scatter_2d(df_viz, km_labels, title="K-Means mit Zentroiden", show_legend=False)
        fig_km.add_trace(go.Scatter(
            x=centroids_viz[:, 0], y=centroids_viz[:, 1],
            mode="markers",
            marker=dict(size=20, color="black", symbol="cross", line=dict(width=3, color="white")),
            name="Zentroide",
        ))
        st.plotly_chart(fig_km, width="stretch")

    # Dendrogram
    st.subheader("Agglomerativ: Dendrogram")
    # Subsample for dendrogram if too many points
    n_dendro = min(200, len(X))
    if n_dendro < len(X):
        rng = np.random.RandomState(42)
        dendro_idx = rng.choice(len(X), n_dendro, replace=False)
        X_dendro = X[dendro_idx]
        st.caption(f"Dendrogram auf Subsample von {n_dendro} Punkten (Performance).")
    else:
        X_dendro = X

    Z = linkage(X_dendro, method=linkage_method)
    fig_dendro = dendrogram_plot(Z, n_clusters=k_chosen)
    st.plotly_chart(fig_dendro, width="stretch")

    # Silhouette diagram for chosen k
    st.subheader("Silhouette-Diagramm")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_sil_km = silhouette_diagram(X, km_labels)
        fig_sil_km.update_layout(title="K-Means Silhouette")
        st.plotly_chart(fig_sil_km, width="stretch")
    with col_s2:
        fig_sil_agg = silhouette_diagram(X, agg_labels)
        fig_sil_agg.update_layout(title="Agglomerativ Silhouette")
        st.plotly_chart(fig_sil_agg, width="stretch")

    # === Metrics comparison ===
    st.subheader("Metriken-Vergleich")

    km_sil = silhouette_score(X, km_labels)
    agg_sil = silhouette_score(X, agg_labels)

    # Only compute ARI/NMI if ground truth has more than one unique non-outlier label
    y_true_clean = y_true.copy()
    has_gt = len(np.unique(y_true_clean[y_true_clean >= 0])) > 1

    metrics_data = {
        "Metrik": ["Silhouette Score", "Laufzeit (s)"],
        "K-Means": [f"{km_sil:.4f}", f"{km_time:.4f}"],
        "Agglomerativ": [f"{agg_sil:.4f}", f"{agg_time:.4f}"],
    }

    if has_gt:
        # Exclude outliers for ARI/NMI
        mask = y_true >= 0
        km_ari = adjusted_rand_score(y_true[mask], km_labels[mask])
        agg_ari = adjusted_rand_score(y_true[mask], agg_labels[mask])
        km_nmi = normalized_mutual_info_score(y_true[mask], km_labels[mask])
        agg_nmi = normalized_mutual_info_score(y_true[mask], agg_labels[mask])

        metrics_data["Metrik"].extend(["Adjusted Rand Index", "Normalized Mutual Info"])
        metrics_data["K-Means"].extend([f"{km_ari:.4f}", f"{km_nmi:.4f}"])
        metrics_data["Agglomerativ"].extend([f"{agg_ari:.4f}", f"{agg_nmi:.4f}"])

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    # Winner announcement
    if km_sil > agg_sil:
        st.success(f"K-Means hat den besseren Silhouette Score ({km_sil:.4f} vs. {agg_sil:.4f}).")
    elif agg_sil > km_sil:
        st.success(f"Agglomerativ hat den besseren Silhouette Score ({agg_sil:.4f} vs. {km_sil:.4f}).")
    else:
        st.info("Beide Algorithmen liefern den gleichen Silhouette Score.")

    # Store for anomaly tab
    st.session_state["km_model"] = km
    st.session_state["km_labels"] = km_labels
    st.session_state["agg_labels"] = agg_labels
