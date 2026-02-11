"""Kurze Theorie-Texte für jeden Tab (Mix: Fachbegriffe EN, Erklärungen DE)."""

DATA_INTRO = """
**Synthetische Daten** ermöglichen volle Kontrolle über Cluster-Struktur,
Rauschen und Dimensionalität. So können wir Algorithmen unter kontrollierten
Bedingungen vergleichen — die Ground Truth ist bekannt.
"""

DISTANCE_EUCLIDEAN = r"""
**Euklidische Distanz** — die "Luftlinie" zwischen zwei Punkten:
$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$
"""

DISTANCE_MANHATTAN = r"""
**Manhattan-Distanz** — die "Taxifahrer-Distanz", nur entlang der Achsen:
$$d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n}|x_i - y_i|$$
"""

DISTANCE_COSINE = r"""
**Cosinus-Ähnlichkeit** — misst den Winkel, nicht den Abstand:
$$\text{sim}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{||\mathbf{x}|| \cdot ||\mathbf{y}||}$$
Werte nahe 1 = ähnliche Richtung, 0 = orthogonal, -1 = entgegengesetzt.
"""

SCALING_INTRO = """
**Feature-Skalierung** bringt alle Features auf einen vergleichbaren Wertebereich.
Ohne Skalierung dominieren Features mit großen Wertebereichen die Distanzberechnung.
"""

SCALER_DESCRIPTIONS = {
    "StandardScaler": "Transformiert auf Mittelwert=0, Standardabweichung=1. Gut für normalverteilte Daten.",
    "MinMaxScaler": "Skaliert auf den Bereich [0, 1]. Gut wenn die Grenzen bekannt sind.",
    "RobustScaler": "Nutzt Median und IQR statt Mean/Std. Robust gegenüber Ausreißern.",
}

PCA_INTRO = """
**PCA (Principal Component Analysis)** findet die Achsen maximaler Varianz in den Daten.
Reduziert Dimensionen, indem wenig-informative Richtungen verworfen werden.
Erhält **globale Struktur**, kann aber nicht-lineare Muster übersehen.
"""

UMAP_INTRO = """
**UMAP (Uniform Manifold Approximation and Projection)** ist ein nicht-lineares
Verfahren, das **lokale Nachbarschaften** erhält. Besonders gut für die
Visualisierung von hochdimensionalen Cluster-Strukturen.
"""

ELBOW_INTRO = """
Die **Elbow-Methode** plottet die Inertia (Within-Cluster Sum of Squares) gegen
die Cluster-Anzahl k. Am "Ellbogen" bringt ein weiteres Cluster kaum noch Verbesserung.
"""

SILHOUETTE_INTRO = """
Der **Silhouette Score** misst, wie gut jeder Punkt zu seinem Cluster passt
(vs. zum nächsten fremden Cluster). Werte nahe +1 = perfekte Trennung,
nahe 0 = Grenzfall, negativ = falsch zugeordnet.
"""

KMEANS_INTRO = """
**K-Means** partitioniert Daten in k Cluster durch iteratives Verschieben
von Zentroiden. Schnell und skalierbar, aber erwartet kugelförmige Cluster.
"""

AGGLOM_INTRO = """
**Agglomeratives Clustering** baut Cluster von unten auf: startet mit jedem
Punkt als eigenem Cluster und verschmilzt schrittweise die nächstgelegenen.
Die Linkage-Methode bestimmt, wie "Nähe" zwischen Clustern gemessen wird.
"""

ANOMALY_CLUSTER_INTRO = """
**Cluster-basierte Anomalieerkennung:** Punkte, die weit vom nächsten
Cluster-Zentroid entfernt sind, gelten als verdächtig. Einfach und interpretierbar.
"""

ANOMALY_IFOREST_INTRO = """
**Isolation Forest** isoliert Anomalien durch zufällige Splits. Anomalien
brauchen weniger Splits → kürzere Pfade im Baum = höherer Anomaly-Score.
"""

ANOMALY_LOF_INTRO = """
**Local Outlier Factor (LOF)** vergleicht die lokale Dichte eines Punktes
mit der seiner Nachbarn. Punkte in dünn besiedelten Regionen = Anomalien.
"""

ANOMALY_DBSCAN_INTRO = """
**DBSCAN** findet dichte Regionen und markiert Punkte in dünnen Regionen
als Noise. Diese Noise-Punkte können als Anomalien interpretiert werden.
"""
