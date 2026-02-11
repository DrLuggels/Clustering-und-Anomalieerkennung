# Clustering & Anomalieerkennung — Streamlit Demo App

## Konzept: "Vom Datenpunkt zur Erkenntnis"

Eine durchgängige, interaktive Story in 6 Tabs. Die Daten, die im ersten Tab
erzeugt werden, fließen durch die gesamte App. Jeder Tab baut auf dem vorherigen
auf. Der User sieht **live**, wie sich Entscheidungen auswirken.

---

## Architektur

```
app.py                    # Hauptdatei mit Tab-Steuerung
├── tabs/
│   ├── tab1_data.py      # Daten-Werkstatt
│   ├── tab2_distance.py  # Distanz-Explorer
│   ├── tab3_scaling.py   # Skalierungs-Labor
│   ├── tab4_dimred.py    # Dimensionsreduktion
│   ├── tab5_clustering.py# Clustering-Arena
│   └── tab6_anomaly.py   # Anomalie-Detektor
├── utils/
│   ├── data_gen.py       # Datengenerierung
│   ├── metrics.py        # Distanzberechnungen
│   ├── viz.py            # Plotly-Visualisierungen
│   └── explanations.py   # Theorie-Texte (kurz)
├── requirements.txt
└── README.md             # Nur wenn gewünscht
```

---

## Tab 1: Daten-Werkstatt 🔬

**Ziel:** Studierende verstehen, wie Daten aussehen und was "Cluster" bedeutet.

### Features:
- **Sidebar-Controls:**
  - Dropdown: Datenform (`make_blobs`, `make_moons`, `make_circles`, `anisotropic`)
  - Slider: Anzahl Cluster (2-8)
  - Slider: Anzahl Datenpunkte (100-2000)
  - Slider: Rauschen / Standardabweichung
  - Slider: Anzahl Features (2-10) — für höherdimensionale Demos
  - Checkbox: "Outlier injizieren" + Slider für Anzahl (1-50)
- **Visualisierung:**
  - Plotly Scatter (2D wenn 2 Features, 3D-Rotation wenn 3+)
  - Farbcodiert nach echten Labels (Ground Truth)
  - Outlier als rote X-Marker hervorgehoben
- **Erklärtext:** 2-3 Sätze was synthetische Daten sind und warum wir sie nutzen

### Daten werden in `st.session_state` gespeichert → verfügbar in allen Tabs

---

## Tab 2: Distanz-Explorer 📏

**Ziel:** Intuitives Verständnis für Distanzmetriken entwickeln.

### Features:
- **Interaktiv: Punkt-Picker**
  - User klickt 2-3 Punkte im Scatter-Plot
  - Distanzen werden live berechnet und als Linien eingezeichnet
  - Alle 3 Metriken (Euklidisch, Manhattan, Cosinus) parallel angezeigt
- **Distanzmatrix-Heatmaps:**
  - 3 nebeneinander: Euklidisch | Manhattan | Cosinus-Ähnlichkeit
  - Plotly Heatmap mit Hover-Werten
  - Auf Subsample (z.B. 50 Punkte) für Performance
- **Mini-Erklärung mit Formel:**
  - Euklidisch: `d = √(Σ(xi-yi)²)` — "Luftlinie"
  - Manhattan: `d = Σ|xi-yi|` — "Taxifahrer-Distanz"
  - Cosinus: `sim = (A·B)/(|A|·|B|)` — "Winkel zwischen Vektoren"
- **Insight-Box:** Automatischer Vergleich: "Für diese Daten unterscheiden sich
  Euklidisch und Manhattan um durchschnittlich X%"

---

## Tab 3: Skalierungs-Labor ⚖️

**Ziel:** Zeigen warum Skalierung kritisch ist.

### Features:
- **Vorher/Nachher Split-View:**
  - Links: Unskalierte Daten (Scatter)
  - Rechts: Skalierte Daten (Scatter)
  - Gleicher Plot-Scale für visuellen Effekt
- **Scaler-Auswahl:**
  - Radio: `StandardScaler` | `MinMaxScaler` | `RobustScaler`
  - Kurze Erklärung pro Scaler (1 Satz)
- **Statistik-Tabelle:**
  - Mean, Std, Min, Max pro Feature — vorher vs. nachher
- **Distanz-Impact:**
  - Gleiche 2 Punkte, Distanzen vorher vs. nachher
  - "Skalierung hat die Euklidische Distanz um X% verändert"
- **Entscheidung:** Toggle "Skalierte Daten für weitere Tabs verwenden?" → wird in
  session_state gespeichert

---

## Tab 4: Dimensionsreduktion 🌀

**Ziel:** PCA und UMAP verstehen und vergleichen.

### Features:
- **Side-by-Side: PCA vs. UMAP**
  - 2 Scatter-Plots nebeneinander
  - Gleiche Farbcodierung (Ground Truth)
- **PCA-Controls:**
  - Slider: Anzahl Komponenten (2-3)
  - Explained Variance Bar-Chart
  - Scree-Plot (Eigenwerte)
  - Ladungs-Vektoren als Pfeile im Plot (Biplot)
- **UMAP-Controls:**
  - Slider: `n_neighbors` (5-50)
  - Slider: `min_dist` (0.0-1.0)
  - Slider: `metric` Dropdown (euclidean, manhattan, cosine) — **Callback zu Tab 2!**
- **Vergleichs-Metriken:**
  - Trustworthiness Score für beide
  - "PCA erhält globale Struktur, UMAP lokale Nachbarschaften"
- **Highlight:** Wenn Features > 3, zeigen: "Ohne Dimensionsreduktion könnten wir
  diese Daten nicht visualisieren"

---

## Tab 5: Clustering-Arena ⚔️

**Ziel:** K-Means vs. Agglomeratives Clustering verstehen und vergleichen.

### Features:

### 5a: Optimale Cluster-Anzahl finden
- **Elbow-Methode:**
  - Plotly Line-Chart: Inertia vs. k (1-10)
  - Automatische Knick-Erkennung (KneeLocator) mit Markierung
  - Kurze Erklärung: "Der 'Ellbogen' zeigt wo mehr Cluster kaum noch helfen"
- **Silhouette-Score:**
  - Plotly Line-Chart: Score vs. k (2-10)
  - Silhouette-Diagramm (Balken pro Cluster, farbcodiert) für gewähltes k
  - Erklärung: "Werte nahe 1 = gut getrennte Cluster"
- **Empfehlung:** "Basierend auf Elbow (k=X) und Silhouette (k=Y) empfehlen wir k=Z"

### 5b: Algorithmus-Vergleich
- **Side-by-Side Scatter:**
  - Links: K-Means Ergebnis
  - Rechts: Agglomeratives Clustering Ergebnis
  - Gleicher k-Wert, gleiche Daten
- **K-Means Details:**
  - Zentroide als große Marker
  - Voronoi-Regionen (Entscheidungsgrenzen)
  - Slider: `n_init` (Anzahl Neustarts)
  - Metriken: Inertia, Silhouette, Laufzeit
- **Agglomeratives Clustering Details:**
  - Dendrogram (scipy, als Plotly)
  - Dropdown: Linkage-Methode (ward, complete, average, single)
  - Schnittlinie im Dendrogram bei gewähltem k
  - Metriken: Silhouette, Laufzeit
- **Vergleichs-Tabelle:**
  - Adjusted Rand Index (vs. Ground Truth)
  - Normalized Mutual Information
  - Silhouette Score
  - "K-Means gewinnt bei X, Agglomerativ bei Y"

---

## Tab 6: Anomalie-Detektor 🔍

**Ziel:** Anomalien erkennen — erst über Clustering, dann mit spezialisierten Methoden.

### Features:

### 6a: Cluster-basierte Anomalieerkennung
- **Methode:** Distanz zum nächsten Cluster-Zentroid
- **Slider:** Threshold (Perzentil: 90-99%)
- **Visualisierung:**
  - Scatter mit Farbintensität = Distanz zum Zentroid
  - Anomalien (über Threshold) rot markiert
  - Vergleich mit echten Outliers (wenn in Tab 1 injiziert)
- **Metriken:** Precision, Recall, F1 (wenn Ground Truth vorhanden)

### 6b: Spezialisierte Algorithmen
- **Isolation Forest:**
  - Slider: `contamination` (0.01-0.2)
  - Slider: `n_estimators` (50-300)
  - Anomaly-Score Verteilung (Histogram)
- **Local Outlier Factor (LOF):**
  - Slider: `n_neighbors` (5-50)
  - LOF-Score Visualisierung
- **DBSCAN als Anomalie-Detektor:**
  - Slider: `eps`, `min_samples`
  - Noise-Punkte = Anomalien

### 6c: Methoden-Vergleich
- **4er-Grid:** Cluster-basiert | IForest | LOF | DBSCAN
- **Venn-Diagramm / Overlap-Matrix:** Welche Methoden erkennen welche Punkte?
- **Metriken-Tabelle:** Precision, Recall, F1 pro Methode
- **Insight:** "Keine Methode ist perfekt — Ensemble-Ansätze kombinieren mehrere"

---

## Durchgängige Features (alle Tabs)

### Sidebar (persistent)
- App-Titel + Logo-Bereich
- "Aktuelle Daten" Info-Box: n_samples, n_features, n_clusters
- "Skalierung aktiv" Indikator
- "Dimensionsreduktion" Indikator
- Download-Button: Daten als CSV exportieren

### UX-Details
- Alle Plots mit Plotly (interaktiv, hover, zoom)
- Konsistente Farbpalette über alle Tabs
- Loading-Spinner bei rechenintensiven Operationen (UMAP, große Daten)
- `@st.cache_data` für teure Berechnungen
- Responsive Layout mit `st.columns()`

---

## Tech-Stack

```
streamlit>=1.30
numpy
pandas
scikit-learn
umap-learn
plotly
scipy
kneed              # Für automatische Elbow-Erkennung
```

---

## Implementierungs-Reihenfolge

1. Projektstruktur + requirements.txt + virtuelle Umgebung
2. `app.py` Grundgerüst mit Tab-Navigation
3. `utils/data_gen.py` — Datengenerierung
4. Tab 1: Daten-Werkstatt (Basis für alles andere)
5. `utils/metrics.py` — Distanzberechnungen
6. Tab 2: Distanz-Explorer
7. Tab 3: Skalierungs-Labor
8. `utils/viz.py` — Gemeinsame Plot-Funktionen
9. Tab 4: Dimensionsreduktion (PCA + UMAP)
10. Tab 5: Clustering-Arena (Elbow, Silhouette, K-Means, Agglomerativ)
11. Tab 6: Anomalie-Detektor
12. `utils/explanations.py` — Theorie-Texte einfügen
13. Feinschliff: Caching, Performance, Edge-Cases
14. Testen mit verschiedenen Daten-Konfigurationen
