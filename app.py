from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).parent
REQUIRED = {
    "clusters": "spotify_clusters.csv",
    "scaled": "spotify_scaled_features.csv",
    "cleaned": "spotify_cleaned.csv",
}
FEATURES = ["danceability", "energy", "acousticness", "instrumentalness",
            "liveness", "speechiness", "valence", "tempo", "loudness"]
LABELS = {"danceability": "Danceability", "energy": "Energy", "acousticness": "Acousticness",
          "instrumentalness": "Instrumentalness", "liveness": "Liveness",
          "speechiness": "Speechiness", "valence": "Positivity (valence)",
          "tempo": "Tempo", "loudness": "Loudness"}
UNIT_FEATURES = ["danceability", "energy", "acousticness", "instrumentalness",
                 "liveness", "speechiness", "valence"]          # all between 0 and 1
# Spotify-inspired colours: vibrant palette with easy distinction
SPOTIFY_GREEN = "#1DB954"
PALETTE = ["#1DB954", "#8F7CF7", "#C6F432", "#F0679E", "#25D0C0",
           "#F6C945", "#B9F6CA", "#FF8A3D", "#5FD38D", "#5AC8FA"]
GREEN_SCALE = [[0, "#0B6E3A"], [0.5, "#1DB954"], [1, "#B9F6CA"]]            # dark green -> mint
DIVERGING = [[0, "#8F7CF7"], [0.5, "#2B2B2B"], [1, "#1ED760"]]              # below avg -> above avg
PANEL = "#181818"                                                          # dark chart background

st.set_page_config(page_title="Spotify Song Clusters", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      footer {visibility: hidden;}

      /* Hide Streamlit header anchor links and action icons beside text */
      [data-testid="stHeaderActionElements"] {display: none !important;}
      .stHeading a, a.header-anchor, a[data-testid="stHeaderAnchor"] {display: none !important;}
      a.anchorjs-link, .anchorjs-link {display: none !important;}
      h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {display: none !important;}

      /* Hero banner */
      .hero {background: linear-gradient(135deg, #1ED760 0%, #14833B 45%, #0B3D2E 100%);
             border-radius: 20px; padding: 28px 32px; margin-bottom: 18px;
             box-shadow: 0 10px 30px rgba(29,185,84,.25);}
      .hero h1 {color: #FFFFFF; margin: 0; padding: 0; font-size: 2.2rem;}
      .hero p {color: rgba(255,255,255,.88); font-size: 1.05rem; margin: 6px 0 0 0;}

      /* Metric cards */
      [data-testid="stMetric"] {background: linear-gradient(135deg, rgba(29,185,84,.20), rgba(29,185,84,.04));
             border: 1px solid rgba(29,185,84,.40); border-radius: 14px; padding: 14px 18px;}

      /* Tabs */
      button[data-baseweb="tab"][aria-selected="true"] p {color: #1DB954; font-weight: 700;}
      div[data-baseweb="tab-highlight"] {background-color: #1DB954 !important;}

      /* Buttons */
      .stButton > button {background: #1DB954; color: #0B0B0B; border: 0; border-radius: 999px;
             font-weight: 700; padding: 0.4rem 1.4rem;}
      .stButton > button:hover {background: #1ED760; color: #000000;}

      /* Sidebar glow */
      [data-testid="stSidebar"] {background-image: linear-gradient(180deg, rgba(29,185,84,.25), rgba(29,185,84,0) 55%);}
    </style>
    """,
    unsafe_allow_html=True,
)


def hex_to_rgba(hex_str, alpha=0.3):
    """Convert hex colour string to rgba CSS/Plotly format."""
    hex_str = hex_str.lstrip("#")
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def show(fig, three_d=False):
    """Give every chart the same dark Spotify look, then draw it."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": "#EDEDED"},
        legend={"bgcolor": "rgba(0,0,0,0)", "itemsizing": "constant"}
    )
    if three_d:
        axis = {"backgroundcolor": PANEL, "gridcolor": "#333333", "zerolinecolor": "#333333"}
        fig.update_layout(
            scene={
                "bgcolor": PANEL,
                "xaxis": axis,
                "yaxis": axis,
                "zaxis": axis,
                "dragmode": "orbit"
            }
        )
    st.plotly_chart(fig, theme=None, use_container_width=True, config={"displaylogo": False})


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading songs...")
def load_data():
    clusters = pd.read_csv(DATA_DIR / REQUIRED["clusters"])
    scaled = pd.read_csv(DATA_DIR / REQUIRED["scaled"])
    cleaned = pd.read_csv(DATA_DIR / REQUIRED["cleaned"], low_memory=False)

    if not (len(clusters) == len(scaled) == len(cleaned)):
        raise ValueError("The three CSV files have different numbers of rows. "
                         "Re-run the notebook so they are generated together.")
    if "track_id" in cleaned and not (clusters["track_id"].to_numpy() == cleaned["track_id"].to_numpy()).all():
        raise ValueError("Row order differs between spotify_clusters.csv and spotify_cleaned.csv.")

    songs = clusters.copy()
    songs["cluster"] = songs["cluster"].astype(int)
    songs["popularity"] = cleaned["popularity"].to_numpy()
    for f in FEATURES:
        songs[f] = cleaned[f].to_numpy()

    top_genre = songs.groupby("cluster")["track_genre"].agg(lambda s: s.value_counts().index[0])
    names = {int(c): f"Cluster {c} · {g}" for c, g in top_genre.items()}
    songs["cluster_name"] = songs["cluster"].map(names)

    profile = scaled.groupby(songs["cluster"].to_numpy()).mean()
    profile.index.name = "cluster"
    return songs, scaled, profile, names


@st.cache_data(show_spinner="Projecting songs...")
def compute_pca():
    _, scaled, _, _ = load_data()
    pca = PCA(n_components=3, random_state=0)
    coords = pca.fit_transform(scaled.to_numpy())
    return coords, pca.explained_variance_ratio_


@st.cache_data(show_spinner="Scoring clusters...")
def compute_silhouette():
    songs, scaled, _, _ = load_data()
    if songs["cluster"].nunique() < 2:
        return float("nan")
    return float(silhouette_score(scaled.to_numpy(), songs["cluster"].to_numpy(),
                                  sample_size=min(10000, len(songs)), random_state=0))


@st.cache_resource(show_spinner="Building the similarity index...")
def get_neighbour_index():
    _, scaled, _, _ = load_data()
    return NearestNeighbors(n_neighbors=11).fit(scaled.to_numpy())


def colour_map(names, order):
    return {names[c]: PALETTE[i % len(PALETTE)] for i, c in enumerate(order)}


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_overview(songs, profile, names, order, colours):
    n_songs, n_clusters = len(songs), songs["cluster"].nunique()
    sizes = songs["cluster"].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Songs clustered", f"{n_songs:,}")
    c2.metric("Clusters found", n_clusters)
    c3.metric("Largest cluster", f"{100 * sizes.iloc[0] / n_songs:.0f}% of songs")
    c4.metric("Silhouette score", f"{compute_silhouette():.3f}",
              help="How well songs fit their own cluster versus the nearest other cluster "
                   "(-1 to 1, higher is better). Values around 0.1 are normal for music features.")

    st.subheader("The song map", anchor=False)
    coords, var = compute_pca()
    left, right = st.columns([3, 1])
    with right:
        view = st.radio("View", ["3D (drag freely to rotate)", "2D"])
        axes_mode = st.radio("Axes", ["Combined dimensions (PCA)", "Choose my own features"])
        if axes_mode == "Choose my own features":
            feature_ids = list(FEATURES)
            ax_x = st.selectbox("X axis", feature_ids, index=1, format_func=LABELS.get)
            ax_y = st.selectbox("Y axis", feature_ids, index=2, format_func=LABELS.get)
            ax_z = st.selectbox("Z axis", feature_ids, index=6, format_func=LABELS.get) if view.startswith("3D") else None
        sample_n = st.slider("Songs to draw", 2000, min(20000, n_songs), min(6000, n_songs), step=1000)
        if axes_mode == "Combined dimensions (PCA)":
            used = 3 if view.startswith("3D") else 2
            st.caption(f"Each dot is a song. PCA squeezes the nine audio features into {used} combined "
                       f"axes that capture {100 * var[:used].sum():.0f}% of the variation, so similar songs "
                       f"sit close together. Drag freely in any direction to rotate, scroll to zoom, click a legend item to hide a cluster.")
        else:
            st.caption("Axes show the actual feature values, so you can see how clusters differ on any "
                       "features you pick. Drag freely in any direction to rotate, scroll to zoom.")
    with left:
        idx = np.random.default_rng(0).choice(n_songs, size=sample_n, replace=False)
        plot_df = songs.iloc[idx].copy()
        if axes_mode == "Combined dimensions (PCA)":
            cols = ["PC1", "PC2", "PC3"]
            for i, c in enumerate(cols):
                plot_df[c] = coords[idx, i]
            titles = {c: f"{c} ({100 * var[i]:.0f}%)" for i, c in enumerate(cols)}
        else:
            cols = [ax_x, ax_y] + ([ax_z] if ax_z else [])
            titles = {c: LABELS[c] for c in cols}
        hover = {"artists": True, "track_genre": True, "cluster_name": False}
        hover.update({c: False for c in cols})
        common = dict(color="cluster_name", color_discrete_map=colours,
                      category_orders={"cluster_name": [names[c] for c in order]},
                      hover_name="track_name", hover_data=hover, opacity=0.8, labels=titles)
        if view.startswith("3D"):
            fig = px.scatter_3d(plot_df, x=cols[0], y=cols[1], z=cols[2], height=640, **common)
            fig.update_traces(marker={"size": 5})
            fig.update_layout(legend_title_text="", margin={"t": 0, "b": 0, "l": 0, "r": 0},
                              scene={"camera": {"eye": {"x": 1.6, "y": 1.6, "z": 0.9}}, "dragmode": "orbit"})
        else:
            fig = px.scatter(plot_df, x=cols[0], y=cols[1], render_mode="webgl", height=520, **common)
            fig.update_traces(marker={"size": 6})
            fig.update_layout(legend_title_text="", margin={"t": 10, "b": 10})
        show(fig, three_d=view.startswith("3D"))

    a, b = st.columns(2)
    with a:
        st.subheader("Cluster sizes", anchor=False)
        size_df = pd.DataFrame({"Cluster": [names[c] for c in order],
                                "Songs": [int(sizes[c]) for c in order]})
        fig = px.bar(size_df, x="Songs", y="Cluster", orientation="h", color="Cluster",
                     color_discrete_map=colours, height=380)
        fig.update_layout(showlegend=False, yaxis={"autorange": "reversed", "title": ""}, margin={"t": 10})
        show(fig)
    with b:
        st.subheader("What defines each cluster", anchor=False)
        heat = profile.loc[order].rename(index=names, columns=LABELS)
        fig = px.imshow(heat, aspect="auto", color_continuous_scale=DIVERGING, zmin=-2, zmax=2,
                        text_auto=".1f", height=380)
        fig.update_layout(coloraxis_colorbar={"title": "z-score"}, margin={"t": 10})
        show(fig)
        st.caption("Green indicates features above dataset average, purple indicates below.")


def tab_explore(songs, profile, names, order, colours):
    choice = st.selectbox("Pick a cluster", order, format_func=lambda c: names[c])
    subset = songs[songs["cluster"] == choice]
    c_color = colours[names[choice]]

    st.markdown(
        f"<div style='display:flex; align-items:center; gap:10px; margin: 4px 0 16px 0;'>"
        f"<span style='display:inline-block; width:16px; height:16px; border-radius:50%; background-color:{c_color}; box-shadow:0 0 8px {c_color};'></span>"
        f"<span style='font-size:1.1rem; font-weight:700; color:#FFFFFF;'>{names[choice]}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Songs", f"{len(subset):,}")
    c2.metric("Share of all songs", f"{100 * len(subset) / len(songs):.1f}%")
    c3.metric("Average popularity", f"{subset['popularity'].mean():.0f} / 100")

    a, b = st.columns(2)
    with a:
        st.subheader("Audio fingerprint", anchor=False)
        cluster_means = [float(subset[f].mean()) for f in UNIT_FEATURES]
        overall_means = [float(songs[f].mean()) for f in UNIT_FEATURES]
        feat_labels = [LABELS[f] for f in UNIT_FEATURES]

        r_cluster = cluster_means + [cluster_means[0]]
        r_overall = overall_means + [overall_means[0]]
        theta_labels = feat_labels + [feat_labels[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_cluster,
            theta=theta_labels,
            fill="toself",
            name=names[choice],
            line=dict(color=c_color, width=2.5),
            fillcolor=hex_to_rgba(c_color, 0.3),
        ))
        fig.add_trace(go.Scatterpolar(
            r=r_overall,
            theta=theta_labels,
            name="All songs average",
            line=dict(color="rgba(255, 255, 255, 0.45)", dash="dash", width=1.5),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor=PANEL,
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333333", linecolor="#333333",
                                tickfont=dict(color="#888888", size=9)),
                angularaxis=dict(gridcolor="#333333", linecolor="#333333", tickfont=dict(color="#EDEDED", size=10)),
            ),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            margin=dict(t=20, b=30, l=40, r=40),
            height=380,
        )
        show(fig)
        st.caption("Average audio attributes (0 to 1) for this cluster vs. the entire music library.")
    with b:
        st.subheader("Sound profile deviations", anchor=False)
        prof = profile.loc[choice].rename(index=LABELS).sort_values()
        pdf = pd.DataFrame({"Feature": prof.index, "Difference from average": prof.values})
        pdf["Direction"] = np.where(pdf["Difference from average"] >= 0, "Above average", "Below average")
        fig = px.bar(pdf, x="Difference from average", y="Feature", orientation="h", color="Direction",
                     color_discrete_map={"Above average": "#1DB954", "Below average": "#8F7CF7"}, height=380)
        fig.update_layout(legend_title_text="", margin={"t": 10})
        show(fig)
        st.caption("Measured in standard deviations (z-scores) from the average song.")

    st.subheader("Songs in this cluster", anchor=False)
    st.caption("Top 15 most popular songs in this cluster.")
    shown = subset.sort_values("popularity", ascending=False).head(15)
    table = shown[["track_name", "artists", "track_genre", "popularity"]].rename(
        columns={"track_name": "Song", "artists": "Artist", "track_genre": "Genre", "popularity": "Popularity"})
    st.dataframe(table, hide_index=True)


def tab_song_finder(songs, names):
    st.write("Search for a song to see which cluster it belongs to and find its closest sonic matches.")
    query = st.text_input("Song name", placeholder="e.g. Bohemian Rhapsody, Yellow, Stay")
    if len(query.strip()) < 2:
        st.info("Type at least two letters to search.")
        return

    q = query.strip()
    mask = songs["track_name"].str.contains(q, case=False, regex=False, na=False)
    found = songs[mask].sort_values("popularity", ascending=False).head(100)
    if found.empty:
        st.warning("No songs matched. Try a different song title.")
        return

    labels = [f"{r.track_name} - {r.artists}  ({r.track_genre})" for r in found.itertuples()]
    picked = st.selectbox(f"{len(found)} match(es), most popular first", range(len(found)),
                          format_func=lambda i: labels[i])
    song = found.iloc[picked]
    position = songs.index.get_loc(found.index[picked])

    st.subheader(song["track_name"], anchor=False)
    st.caption(f"{song['artists']}  ·  {song['track_genre']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cluster", names[song["cluster"]])
    c2.metric("Popularity", f"{song['popularity']} / 100")
    c3.metric("Tempo", f"{song['tempo']:.0f} BPM")
    c4.metric("Loudness", f"{song['loudness']:.1f} dB")

    a, b = st.columns([1, 1])
    with a:
        st.subheader("Sound features", anchor=False)
        fdf = pd.DataFrame({"Feature": [LABELS[f] for f in UNIT_FEATURES],
                            "Value": [song[f] for f in UNIT_FEATURES]})
        fig = px.bar(fdf, x="Value", y="Feature", orientation="h", range_x=[0, 1], height=340,
                     color="Value", color_continuous_scale=GREEN_SCALE, range_color=[0, 1])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(yaxis={"autorange": "reversed", "title": ""}, margin={"t": 10})
        show(fig)
    with b:
        st.subheader("Most similar songs", anchor=False)
        _, scaled, _, _ = load_data()
        dist, idx = get_neighbour_index().kneighbors(scaled.iloc[[position]].to_numpy())
        neighbours = [(d, i) for d, i in zip(dist[0], idx[0]) if i != position][:10]
        rows = songs.iloc[[i for _, i in neighbours]]
        out = pd.DataFrame({
            "Song": rows["track_name"].to_numpy(),
            "Artist": rows["artists"].to_numpy(),
            "Genre": rows["track_genre"].to_numpy(),
            "Cluster": rows["cluster"].to_numpy(),
            "Distance": [round(float(d), 2) for d, _ in neighbours],
        })
        same = int((out["Cluster"] == song["cluster"]).sum())
        st.caption(f"{same} of these {len(out)} songs are in the same cluster. Smaller distance means more similar.")
        st.dataframe(out, hide_index=True)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    st.markdown(
        '<div class="hero"><h1>Spotify Song Clusters</h1>'
        "<p>Songs grouped by how they sound, using Union-Find hierarchical clustering.</p></div>",
        unsafe_allow_html=True,
    )

    missing = [name for name in REQUIRED.values() if not (DATA_DIR / name).exists()]
    if missing:
        st.error("Missing data files next to app.py: " + ", ".join(missing)
                 + ". Run the notebook first so it writes them.")
        st.stop()

    try:
        songs, _, profile, names = load_data()
    except ValueError as err:
        st.error(str(err))
        st.stop()

    order = songs["cluster"].value_counts().index.tolist()

    with st.sidebar:
        st.header("Guide", anchor=False)
        st.markdown(
            "- **Overview**: Visualize how songs group together in 2D or 3D space to uncover overarching patterns in the dataset.\n"
            "- **Explore clusters**: Examine the sound fingerprint, audio profiles, and top tracks of any specific cluster.\n"
            "- **Find a song**: Search for any song to discover its cluster placement and find its most similar audio matches."
        )
        st.divider()
        st.caption(f"{len(songs):,} songs · {songs['track_genre'].nunique()} genres · "
                   f"{songs['cluster'].nunique()} clusters")

    colours = colour_map(names, order)

    t1, t2, t3 = st.tabs(["Overview", "Explore clusters", "Find a song"])
    with t1:
        tab_overview(songs, profile, names, order, colours)
    with t2:
        tab_explore(songs, profile, names, order, colours)
    with t3:
        tab_song_finder(songs, names)


if __name__ == "__main__":
    main()
