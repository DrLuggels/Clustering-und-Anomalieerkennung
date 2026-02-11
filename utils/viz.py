import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Consistent color palette across all tabs
CLUSTER_COLORS = px.colors.qualitative.Set2
OUTLIER_COLOR = "#FF0000"
ANOMALY_COLOR = "#FF4444"
NORMAL_COLOR = "#4488FF"


def scatter_2d(df: pd.DataFrame, labels: np.ndarray, title: str = "",
               outlier_mask: np.ndarray | None = None,
               col_x: str | None = None, col_y: str | None = None,
               show_legend: bool = True) -> go.Figure:
    """Create a 2D scatter plot with cluster coloring."""
    cols = df.columns.tolist()
    cx = col_x or cols[0]
    cy = col_y or cols[1]

    color_labels = [str(l) for l in labels]
    fig = px.scatter(
        df, x=cx, y=cy, color=color_labels,
        color_discrete_sequence=CLUSTER_COLORS,
        title=title,
        labels={"color": "Cluster"},
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))

    if outlier_mask is not None and outlier_mask.any():
        outlier_df = df[outlier_mask]
        fig.add_trace(go.Scatter(
            x=outlier_df[cx], y=outlier_df[cy],
            mode="markers",
            marker=dict(size=12, color=OUTLIER_COLOR, symbol="x", line=dict(width=2)),
            name="Outlier",
            showlegend=True,
        ))

    fig.update_layout(
        template="plotly_white",
        showlegend=show_legend,
        height=500,
        margin=dict(t=40, b=20),
    )
    return fig


def scatter_3d(df: pd.DataFrame, labels: np.ndarray, title: str = "",
               outlier_mask: np.ndarray | None = None) -> go.Figure:
    """Create a 3D scatter plot."""
    cols = df.columns.tolist()
    color_labels = [str(l) for l in labels]
    fig = px.scatter_3d(
        df, x=cols[0], y=cols[1], z=cols[2],
        color=color_labels,
        color_discrete_sequence=CLUSTER_COLORS,
        title=title,
    )
    fig.update_traces(marker=dict(size=3, opacity=0.7))

    if outlier_mask is not None and outlier_mask.any():
        outlier_df = df[outlier_mask]
        fig.add_trace(go.Scatter3d(
            x=outlier_df[cols[0]], y=outlier_df[cols[1]], z=outlier_df[cols[2]],
            mode="markers",
            marker=dict(size=6, color=OUTLIER_COLOR, symbol="x"),
            name="Outlier",
        ))

    fig.update_layout(template="plotly_white", height=600, margin=dict(t=40, b=20))
    return fig


def heatmap(matrix: np.ndarray, title: str = "", labels: list[str] | None = None,
            colorscale: str = "Viridis", zmin: float | None = None,
            zmax: float | None = None) -> go.Figure:
    """Create an interactive heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        hovertemplate="Punkt %{x} → Punkt %{y}<br>Wert: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=450,
        margin=dict(t=40, b=20),
        xaxis_title="Datenpunkt",
        yaxis_title="Datenpunkt",
    )
    return fig


def elbow_chart(k_range: range, inertias: list[float],
                knee_k: int | None = None) -> go.Figure:
    """Create an elbow chart with optional knee point marker."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range), y=inertias,
        mode="lines+markers",
        name="Inertia",
        line=dict(color="#2196F3", width=3),
        marker=dict(size=8),
    ))
    if knee_k is not None:
        knee_idx = knee_k - k_range.start
        fig.add_trace(go.Scatter(
            x=[knee_k], y=[inertias[knee_idx]],
            mode="markers",
            marker=dict(size=16, color="#FF5722", symbol="star"),
            name=f"Ellbogen (k={knee_k})",
        ))
        fig.add_vline(x=knee_k, line_dash="dash", line_color="#FF5722", opacity=0.5)

    fig.update_layout(
        title="Elbow-Methode",
        xaxis_title="Anzahl Cluster (k)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        template="plotly_white",
        height=400,
    )
    return fig


def silhouette_chart(k_range: range, scores: list[float],
                     best_k: int | None = None) -> go.Figure:
    """Create a silhouette score line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range), y=scores,
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#4CAF50", width=3),
        marker=dict(size=8),
    ))
    if best_k is not None:
        best_idx = best_k - k_range.start
        fig.add_trace(go.Scatter(
            x=[best_k], y=[scores[best_idx]],
            mode="markers",
            marker=dict(size=16, color="#FF9800", symbol="star"),
            name=f"Bestes k={best_k}",
        ))

    fig.update_layout(
        title="Silhouette Score",
        xaxis_title="Anzahl Cluster (k)",
        yaxis_title="Silhouette Score",
        template="plotly_white",
        height=400,
    )
    return fig


def silhouette_diagram(X: np.ndarray, cluster_labels: np.ndarray) -> go.Figure:
    """Create a silhouette diagram showing per-sample silhouette values."""
    from sklearn.metrics import silhouette_samples

    sample_scores = silhouette_samples(X, cluster_labels)
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)

    fig = go.Figure()
    y_lower = 0

    for i, label in enumerate(sorted(unique_labels)):
        cluster_scores = sample_scores[cluster_labels == label]
        cluster_scores.sort()
        size = len(cluster_scores)
        y_upper = y_lower + size

        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        fig.add_trace(go.Bar(
            x=cluster_scores,
            y=list(range(y_lower, y_upper)),
            orientation="h",
            marker_color=color,
            name=f"Cluster {label}",
            showlegend=True,
        ))
        y_lower = y_upper + 2

    avg_score = np.mean(sample_scores)
    fig.add_vline(x=avg_score, line_dash="dash", line_color="red",
                  annotation_text=f"Durchschnitt: {avg_score:.3f}")

    fig.update_layout(
        title="Silhouette-Diagramm",
        xaxis_title="Silhouette-Wert",
        yaxis_title="Datenpunkte (nach Cluster sortiert)",
        template="plotly_white",
        height=500,
        showlegend=True,
        yaxis=dict(showticklabels=False),
        bargap=0,
    )
    return fig


def dendrogram_plot(linkage_matrix: np.ndarray, n_clusters: int | None = None) -> go.Figure:
    """Create a dendrogram from a linkage matrix using plotly."""
    from scipy.cluster.hierarchy import dendrogram as scipy_dendro
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_mpl, ax = plt.subplots(1, 1, figsize=(10, 5))
    dendro = scipy_dendro(linkage_matrix, ax=ax, truncate_mode="lastp", p=30,
                          color_threshold=0)
    plt.close(fig_mpl)

    fig = go.Figure()

    # Draw dendrogram lines
    for xs, ys, color in zip(dendro["icoord"], dendro["dcoord"], dendro["color_list"]):
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color="#2196F3", width=1.5),
            showlegend=False,
        ))

    # Add cut line if n_clusters specified
    if n_clusters is not None and n_clusters > 1:
        distances = linkage_matrix[:, 2]
        if len(distances) >= n_clusters:
            cut_distance = (distances[-n_clusters + 1] + distances[-n_clusters]) / 2
            fig.add_hline(y=cut_distance, line_dash="dash", line_color="#FF5722",
                         annotation_text=f"Schnitt bei k={n_clusters}")

    fig.update_layout(
        title="Dendrogram",
        xaxis_title="Datenpunkte",
        yaxis_title="Distanz",
        template="plotly_white",
        height=400,
    )
    return fig


def comparison_scatter_side_by_side(
    df: pd.DataFrame, labels_left: np.ndarray, labels_right: np.ndarray,
    title_left: str = "Links", title_right: str = "Rechts",
    col_x: str | None = None, col_y: str | None = None,
) -> go.Figure:
    """Create side-by-side scatter plots for algorithm comparison."""
    cols = df.columns.tolist()
    cx = col_x or cols[0]
    cy = col_y or cols[1]

    fig = make_subplots(rows=1, cols=2, subplot_titles=[title_left, title_right])

    for label in np.unique(labels_left):
        mask = labels_left == label
        fig.add_trace(go.Scatter(
            x=df[cx][mask], y=df[cy][mask],
            mode="markers",
            marker=dict(size=6, color=CLUSTER_COLORS[int(label) % len(CLUSTER_COLORS)], opacity=0.7),
            name=f"C{label}",
            legendgroup=f"left_{label}",
            showlegend=True,
        ), row=1, col=1)

    for label in np.unique(labels_right):
        mask = labels_right == label
        fig.add_trace(go.Scatter(
            x=df[cx][mask], y=df[cy][mask],
            mode="markers",
            marker=dict(size=6, color=CLUSTER_COLORS[int(label) % len(CLUSTER_COLORS)], opacity=0.7),
            name=f"C{label}",
            legendgroup=f"right_{label}",
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(template="plotly_white", height=500, margin=dict(t=50, b=20))
    return fig


def anomaly_scatter(df: pd.DataFrame, scores: np.ndarray, anomaly_mask: np.ndarray,
                    title: str = "", col_x: str | None = None,
                    col_y: str | None = None) -> go.Figure:
    """Scatter plot with anomaly highlighting and score coloring."""
    cols = df.columns.tolist()
    cx = col_x or cols[0]
    cy = col_y or cols[1]

    normal = ~anomaly_mask
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[cx][normal], y=df[cy][normal],
        mode="markers",
        marker=dict(size=6, color=scores[normal], colorscale="Blues",
                    opacity=0.6, showscale=True, colorbar=dict(title="Score")),
        name="Normal",
    ))
    fig.add_trace(go.Scatter(
        x=df[cx][anomaly_mask], y=df[cy][anomaly_mask],
        mode="markers",
        marker=dict(size=10, color=ANOMALY_COLOR, symbol="x", line=dict(width=2)),
        name="Anomalie",
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        margin=dict(t=40, b=20),
    )
    return fig
