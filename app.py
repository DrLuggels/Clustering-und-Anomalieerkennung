import streamlit as st

st.set_page_config(
    page_title="Clustering & Anomalieerkennung",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.title("Clustering &\nAnomalieerkennung")
    st.caption("Interaktive Demo — Vom Datenpunkt zur Erkenntnis")
    st.divider()

    # Data info
    if "data" in st.session_state:
        df = st.session_state["data"]
        st.markdown("**Aktuelle Daten:**")
        col1, col2 = st.columns(2)
        col1.metric("Punkte", len(df), label_visibility="visible")
        col2.metric("Features", df.shape[1], label_visibility="visible")

        n_clusters = st.session_state.get("n_clusters_true", "?")
        outlier_count = int(st.session_state.get("outlier_mask", []).sum()) if "outlier_mask" in st.session_state else 0
        col3, col4 = st.columns(2)
        col3.metric("Cluster", n_clusters)
        col4.metric("Outlier", outlier_count)

        if st.session_state.get("scaling_active"):
            st.success(f"Skalierung: {st.session_state['scaling_active']}")
        else:
            st.info("Skalierung: Keine")

        st.divider()

    st.markdown(
        "**Navigation:** Arbeite die Tabs von links nach rechts durch. "
        "Jeder Tab baut auf dem vorherigen auf."
    )
    st.divider()

    # CSV Download
    if "data" in st.session_state:
        csv = st.session_state["data"].to_csv(index=False)
        st.download_button(
            label="Daten als CSV exportieren",
            data=csv,
            file_name="clustering_data.csv",
            mime="text/csv",
        )

# --- Main Content: Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Daten-Werkstatt",
    "2. Distanz-Explorer",
    "3. Skalierung",
    "4. Dimensionsreduktion",
    "5. Clustering-Arena",
    "6. Anomalie-Detektor",
])

from tabs import tab1_data, tab2_distance, tab3_scaling, tab4_dimred, tab5_clustering, tab6_anomaly

with tab1:
    tab1_data.render()

with tab2:
    tab2_distance.render()

with tab3:
    tab3_scaling.render()

with tab4:
    tab4_dimred.render()

with tab5:
    tab5_clustering.render()

with tab6:
    tab6_anomaly.render()
