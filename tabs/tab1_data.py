import streamlit as st
import numpy as np
from utils.data_gen import generate_data, inject_outliers
from utils.viz import scatter_2d, scatter_3d
from utils.explanations import DATA_INTRO


def render():
    st.header("Daten-Werkstatt")
    st.markdown(DATA_INTRO)

    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Parameter")
        shape = st.selectbox("Datenform", ["Blobs", "Moons", "Circles", "Anisotropic"],
                             help="Bestimmt die geometrische Struktur der Cluster")

        n_features_max = 10
        if shape in ("Moons", "Circles"):
            st.caption("Moons/Circles basieren auf 2D — extra Features werden als Rauschen hinzugefügt.")

        n_clusters = st.slider("Anzahl Cluster", 2, 8, 3,
                               disabled=(shape in ("Moons", "Circles")))
        if shape in ("Moons", "Circles"):
            n_clusters = 2

        n_samples = st.slider("Datenpunkte", 100, 2000, 500, step=50)
        n_features = st.slider("Features (Dimensionen)", 2, n_features_max, 2)
        noise = st.slider("Rauschen / Std", 0.1, 3.0, 0.8, step=0.1)
        random_state = st.number_input("Random Seed", 0, 999, 42)

        st.divider()
        add_outliers = st.checkbox("Outlier injizieren", value=False)
        n_outliers = 0
        if add_outliers:
            n_outliers = st.slider("Anzahl Outlier", 1, 50, 10)

    # Generate data
    df, y = generate_data(shape, n_samples, n_clusters, n_features, noise, random_state)

    outlier_mask = np.zeros(len(df), dtype=bool)
    if add_outliers and n_outliers > 0:
        df, y, outlier_mask = inject_outliers(df, y, n_outliers, random_state)

    # Store in session state
    st.session_state["data"] = df
    st.session_state["labels"] = y
    st.session_state["outlier_mask"] = outlier_mask
    st.session_state["n_clusters_true"] = n_clusters
    st.session_state["n_features"] = n_features

    with col_viz:
        st.subheader("Visualisierung")
        if n_features >= 3:
            view_mode = st.radio("Ansicht", ["2D (erste 2 Features)", "3D (erste 3 Features)"],
                                 horizontal=True)
        else:
            view_mode = "2D (erste 2 Features)"

        if view_mode.startswith("3D"):
            fig = scatter_3d(df[df.columns[:3]], y, title="Synthetische Daten (3D)",
                             outlier_mask=outlier_mask)
        else:
            fig = scatter_2d(df, y, title="Synthetische Daten",
                             outlier_mask=outlier_mask)
        st.plotly_chart(fig, use_container_width=True)

        # Data summary
        st.subheader("Datensatz-Übersicht")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Datenpunkte", len(df))
        col2.metric("Features", n_features)
        col3.metric("Cluster", n_clusters)
        col4.metric("Outlier", int(outlier_mask.sum()))

        with st.expander("Rohdaten anzeigen"):
            st.dataframe(df.head(20), use_container_width=True)
