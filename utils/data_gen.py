import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
import pandas as pd


def generate_data(shape: str, n_samples: int, n_clusters: int, n_features: int,
                  noise: float, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic data based on chosen shape."""
    if shape == "Blobs":
        X, y = make_blobs(
            n_samples=n_samples, centers=n_clusters,
            n_features=n_features, cluster_std=noise,
            random_state=random_state
        )
    elif shape == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        if n_features > 2:
            extra = np.random.RandomState(random_state).randn(n_samples, n_features - 2) * noise
            X = np.hstack([X, extra])
    elif shape == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        if n_features > 2:
            extra = np.random.RandomState(random_state).randn(n_samples, n_features - 2) * noise
            X = np.hstack([X, extra])
    elif shape == "Anisotropic":
        X, y = make_blobs(
            n_samples=n_samples, centers=n_clusters,
            n_features=n_features, cluster_std=noise,
            random_state=random_state
        )
        rng = np.random.RandomState(random_state)
        transformation = rng.randn(n_features, n_features) * 0.6
        np.fill_diagonal(transformation, [1.5] + [0.5] * (n_features - 1))
        X = X @ transformation
    else:
        X, y = make_blobs(
            n_samples=n_samples, centers=n_clusters,
            n_features=n_features, cluster_std=noise,
            random_state=random_state
        )

    columns = [f"Feature_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    return df, y


def inject_outliers(df: pd.DataFrame, y: np.ndarray, n_outliers: int,
                    random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Inject outliers far from the data distribution. Returns df, labels, outlier_mask."""
    rng = np.random.RandomState(random_state)
    n_features = df.shape[1]

    # Place outliers at 3-5x the data range away from center
    data_min = df.values.min(axis=0)
    data_max = df.values.max(axis=0)
    data_range = data_max - data_min
    center = (data_max + data_min) / 2

    outliers = []
    for _ in range(n_outliers):
        direction = rng.randn(n_features)
        direction = direction / np.linalg.norm(direction)
        distance = rng.uniform(1.5, 3.0)
        point = center + direction * data_range * distance
        outliers.append(point)

    outlier_array = np.array(outliers)
    columns = df.columns.tolist()
    outlier_df = pd.DataFrame(outlier_array, columns=columns)

    df_combined = pd.concat([df, outlier_df], ignore_index=True)
    y_combined = np.concatenate([y, np.full(n_outliers, -1)])
    outlier_mask = np.concatenate([np.zeros(len(df), dtype=bool),
                                   np.ones(n_outliers, dtype=bool)])

    return df_combined, y_combined, outlier_mask
