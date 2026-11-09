import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
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
OPTIONAL = {
    "merges": "spotify_merge_history.csv",
    "sensitivity": "spotify_sensitivity.csv",
}
FEATURES = ["danceability", "energy", "acousticness", "instrumentalness",
            "liveness", "speechiness", "valence", "tempo", "loudness"]
LABELS = {"danceability": "Danceability", "energy": "Energy", "acousticness": "Acousticness",
          "instrumentalness": "Instrumentalness", "liveness": "Liveness",
          "speechiness": "Speechiness", "valence": "Positivity (valence)",
          "tempo": "Tempo", "loudness": "Loudness"}
UNIT_FEATURES = ["danceability", "energy", "acousticness", "instrumentalness",
                 "liveness", "speechiness", "valence"]          # all between 0 and 1
# Spotify-inspired colours: green first, then accent colours that stay easy to tell apart
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


def show(fig, three_d=False):
    """Give every chart the same dark Spotify look, then draw it."""
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font={"color": "#EDEDED"}, legend={"bgcolor": "rgba(0,0,0,0)"})
    if three_d:
        axis = {"backgroundcolor": PANEL, "gridcolor": "#333333", "zerolinecolor": "#333333"}
        fig.update_layout(scene={"bgcolor": PANEL, "xaxis": axis, "yaxis": axis, "zaxis": axis})
    st.plotly_chart(fig, theme=None)


def green_shades(k):
    """k shades from deep green to mint, used by the 'Green shades' colour mode."""
    lo, hi = (15, 138, 69), (215, 255, 230)
    out = []
    for i in range(k):
        t = i / max(k - 1, 1)
        out.append("#{:02X}{:02X}{:02X}".format(*[round(a + (b - a) * t) for a, b in zip(lo, hi)]))
    return out


# --------------------------------------------------------------------------- #
# Union-Find (same logic as the notebook; used by the playground tab)
# --------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n

    def find(self, x):
        while self.parent[x] != x:                    # path halving
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:             # union by size
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.components -= 1
        return True


def parse_pairs(text, n):
    """Turn '0-1, 1-2; 3-4' into [(0, 1), (1, 2), (3, 4)] and check the ids are valid."""
    pairs = []
    for token in re.split(r"[,;\n]+", text):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*(?:-|->|–)\s*(\d+)", token)
        if not match:
            raise ValueError(f"Could not read '{token}'. Write pairs like 0-1.")
        a, b = int(match.group(1)), int(match.group(2))
        if a >= n or b >= n:
            raise ValueError(f"'{token}' uses an id outside 0 to {n - 1}.")
        pairs.append((a, b))
    return pairs


def run_playground(n, pairs):
    """Apply the unions one by one and describe what happened at each step."""
    uf = UnionFind(n)
    steps = []
    for i, (a, b) in enumerate(pairs, start=1):
        merged = uf.union(a, b)
        steps.append({
            "Step": i,
            "Edge": f"{a} - {b}",
            "What happened": f"Merged (new group size {uf.size[uf.find(a)]})" if merged
                             else "Skipped (already in the same group)",
            "Groups left": uf.components,
        })
    groups = {}
    for x in range(n):
        groups.setdefault(uf.find(x), []).append(x)
    return pd.DataFrame(steps), list(groups.values())


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


@st.cache_data
def load_optional(key):
    path = DATA_DIR / OPTIONAL[key]
    return pd.read_csv(path) if path.exists() else None


def colour_map(names, order, mode="Spotify mix"):
    colours = green_shades(len(order)) if mode == "Green shades" else PALETTE
    return {names[c]: colours[i % len(colours)] for i, c in enumerate(order)}


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

    st.subheader("The song map")
    coords, var = compute_pca()
    left, right = st.columns([3, 1])
    with right:
        view = st.radio("View", ["3D (drag to rotate)", "2D"])
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
                       f"sit close together. Drag to rotate, scroll to zoom, click a legend item to hide a cluster.")
        else:
            st.caption("Axes show the actual feature values, so you can see how clusters differ on any "
                       "features you pick. Drag to rotate, scroll to zoom.")
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
                      hover_name="track_name", hover_data=hover, opacity=0.7, labels=titles)
        if view.startswith("3D"):
            fig = px.scatter_3d(plot_df, x=cols[0], y=cols[1], z=cols[2], height=640, **common)
            fig.update_traces(marker={"size": 2.5})
            fig.update_layout(legend_title_text="", margin={"t": 0, "b": 0, "l": 0, "r": 0},
                              scene={"camera": {"eye": {"x": 1.6, "y": 1.6, "z": 0.9}}})
        else:
            fig = px.scatter(plot_df, x=cols[0], y=cols[1], render_mode="webgl", height=520, **common)
            fig.update_traces(marker={"size": 4})
            fig.update_layout(legend_title_text="", margin={"t": 10, "b": 10})
        show(fig, three_d=view.startswith("3D"))

    a, b = st.columns(2)
    with a:
        st.subheader("Cluster sizes")
        size_df = pd.DataFrame({"Cluster": [names[c] for c in order],
                                "Songs": [int(sizes[c]) for c in order]})
        fig = px.bar(size_df, x="Songs", y="Cluster", orientation="h", color="Cluster",
                     color_discrete_map=colours, height=380)
        fig.update_layout(showlegend=False, yaxis={"autorange": "reversed", "title": ""}, margin={"t": 10})
        show(fig)
    with b:
        st.subheader("What defines each cluster")
        heat = profile.loc[order].rename(index=names, columns=LABELS)
        fig = px.imshow(heat, aspect="auto", color_continuous_scale=DIVERGING, zmin=-2, zmax=2,
                        text_auto=".1f", height=380)
        fig.update_layout(coloraxis_colorbar={"title": "z-score"}, margin={"t": 10})
        show(fig)
        st.caption("Red means the cluster is above the dataset average for that feature, blue means below.")


def tab_explore(songs, profile, names, order, colours):
    choice = st.selectbox("Pick a cluster", order, format_func=lambda c: names[c])
    subset = songs[songs["cluster"] == choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("Songs", f"{len(subset):,}")
    c2.metric("Share of all songs", f"{100 * len(subset) / len(songs):.1f}%")
    c3.metric("Average popularity", f"{subset['popularity'].mean():.0f} / 100")

    a, b = st.columns(2)
    with a:
        st.subheader("Most common genres")
        genres = (subset["track_genre"].value_counts(normalize=True).head(10) * 100).round(1)
        gdf = genres.rename_axis("Genre").reset_index(name="Share of cluster (%)")
        fig = px.bar(gdf, x="Share of cluster (%)", y="Genre", orientation="h", height=380,
                     color_discrete_sequence=[colours[names[choice]]])
        fig.update_layout(yaxis={"autorange": "reversed", "title": ""}, margin={"t": 10})
        show(fig)
    with b:
        st.subheader("Sound profile")
        prof = profile.loc[choice].rename(index=LABELS).sort_values()
        pdf = pd.DataFrame({"Feature": prof.index, "Difference from average": prof.values})
        pdf["Direction"] = np.where(pdf["Difference from average"] >= 0, "Above average", "Below average")
        fig = px.bar(pdf, x="Difference from average", y="Feature", orientation="h", color="Direction",
                     color_discrete_map={"Above average": "#1DB954", "Below average": "#8F7CF7"}, height=380)
        fig.update_layout(legend_title_text="", margin={"t": 10})
        show(fig)
        st.caption("Measured in standard deviations from the average song.")

    st.subheader("Songs in this cluster")
    mode = st.radio("Show", ["Most popular", "Random sample"], horizontal=True)
    if mode == "Most popular":
        shown = subset.sort_values("popularity", ascending=False).head(15)
    else:
        if "seed" not in st.session_state:
            st.session_state["seed"] = 0
        if st.button("Shuffle"):
            st.session_state["seed"] += 1
        shown = subset.sample(min(15, len(subset)), random_state=st.session_state["seed"])
    table = shown[["track_name", "artists", "track_genre", "popularity"]].rename(
        columns={"track_name": "Song", "artists": "Artist", "track_genre": "Genre", "popularity": "Popularity"})
    st.dataframe(table, hide_index=True)


def tab_song_finder(songs, names):
    st.write("Search for a song or artist to see which cluster it landed in and which songs sound most like it.")
    query = st.text_input("Song or artist", placeholder="e.g. Comedy, Coldplay, Bohemian Rhapsody")
    if len(query.strip()) < 2:
        st.info("Type at least two letters to search.")
        return

    q = query.strip()
    mask = (songs["track_name"].str.contains(q, case=False, regex=False, na=False)
            | songs["artists"].str.contains(q, case=False, regex=False, na=False))
    found = songs[mask].sort_values("popularity", ascending=False).head(100)
    if found.empty:
        st.warning("No songs matched. Try a shorter search.")
        return

    labels = [f"{r.track_name} - {r.artists}  ({r.track_genre})" for r in found.itertuples()]
    picked = st.selectbox(f"{len(found)} match(es), most popular first", range(len(found)),
                          format_func=lambda i: labels[i])
    song = found.iloc[picked]
    position = songs.index.get_loc(found.index[picked])

    st.markdown(f"### {song['track_name']}")
    st.caption(f"{song['artists']}  ·  {song['track_genre']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cluster", names[song["cluster"]])
    c2.metric("Popularity", f"{song['popularity']} / 100")
    c3.metric("Tempo", f"{song['tempo']:.0f} BPM")
    c4.metric("Loudness", f"{song['loudness']:.1f} dB")

    a, b = st.columns([1, 1])
    with a:
        st.subheader("Sound features")
        fdf = pd.DataFrame({"Feature": [LABELS[f] for f in UNIT_FEATURES],
                            "Value": [song[f] for f in UNIT_FEATURES]})
        fig = px.bar(fdf, x="Value", y="Feature", orientation="h", range_x=[0, 1], height=340,
                     color="Value", color_continuous_scale=GREEN_SCALE, range_color=[0, 1])
        fig.update_coloraxes(showscale=False)
        fig.update_layout(yaxis={"autorange": "reversed", "title": ""}, margin={"t": 10})
        show(fig)
    with b:
        st.subheader("Most similar songs")
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


def tab_how_it_works():
    st.subheader("The method in five steps")
    st.markdown(
        """
        1. **Prepare the songs.** Remove duplicates and broken rows, then standardise nine audio features
           (danceability, energy, tempo and so on) so each counts equally.
        2. **Link close songs.** Every song is connected to its nearest neighbours, which gives a graph of
           candidate links ordered by distance.
        3. **Merge with Union-Find.** Links are processed from shortest to longest. If the two songs are in
           different groups, the groups are merged. This is the same process as building a minimum spanning tree
           (Kruskal's algorithm), with Union-Find tracking the groups.
        4. **Stop before one giant group forms.** Two groups that are both already large are not merged. The size
           limit `M` is chosen automatically: the best silhouette score among values where no cluster holds more
           than 40% of the songs. Nobody sets the number of clusters.
        5. **Evaluate.** Clusters are scored and checked against music genres.
        """
    )

    merges = load_optional("merges")
    if merges is not None:
        st.subheader("How the merges unfolded")
        a, b = st.columns(2)
        with a:
            fig = px.line(merges.reset_index(), x="index", y="Distance", height=320,
                          labels={"index": "Merge number", "Distance": "Distance joined"})
            fig.update_traces(line_color=SPOTIFY_GREEN)
            fig.update_layout(margin={"t": 10})
            show(fig)
            st.caption("Merges get longer over time: close songs join first, distant groups last.")
        with b:
            fig = px.line(merges.assign(Largest=merges["New Size"].cummax()).reset_index(), x="index", y="Largest",
                          log_y=True, height=320,
                          labels={"index": "Merge number", "Largest": "Largest group so far"})
            fig.update_traces(line_color=SPOTIFY_GREEN)
            fig.update_layout(margin={"t": 10})
            show(fig)
            st.caption("The size cap keeps the biggest group from swallowing everything.")

    sens = load_optional("sensitivity")
    if sens is not None:
        st.subheader("Are the results sensitive to the settings?")
        st.caption("The whole pipeline was re-run with different neighbour counts (K) and distance limits. "
                   "Similar results across rows mean the clusters do not depend on one lucky setting.")
        st.dataframe(sens, hide_index=True)

    st.subheader("Try Union-Find yourself")
    st.caption("Enter links between numbered items and watch the groups form.")
    c1, c2 = st.columns([1, 3])
    n_items = c1.number_input("Number of items", min_value=2, max_value=50, value=8, step=1)
    text = c2.text_input("Links (pairs like 0-1, separated by commas)", value="0-1, 2-3, 1-2, 4-5, 0-3, 6-7, 5-6")
    try:
        pairs = parse_pairs(text, int(n_items))
    except ValueError as err:
        st.error(str(err))
        return
    if not pairs:
        st.info("Add at least one link to run the demo.")
        return
    steps, groups = run_playground(int(n_items), pairs)
    st.dataframe(steps, hide_index=True)
    st.markdown("**Final groups:** " + "  ".join("{" + ", ".join(map(str, g)) + "}" for g in groups))

    with st.expander("Show the Union-Find code"):
        st.code(
            '''class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):                      # path halving
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):                  # union by size
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True''',
            language="python",
        )


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
        st.header("Guide")
        st.markdown(
            "- **Overview**: the song map and what defines each cluster\n"
            "- **Explore clusters**: genres, sound profile and example songs\n"
            "- **Find a song**: look up a track and its closest matches\n"
            "- **How it works**: the method and a Union-Find demo"
        )
        st.divider()
        colour_mode = st.radio("Cluster colours", ["Spotify mix", "Green shades"],
                               help="Spotify mix uses green plus accent colours that are easy to tell apart. "
                                    "Green shades colours the clusters from deep green to mint.")
        st.divider()
        st.caption(f"{len(songs):,} songs · {songs['track_genre'].nunique()} genres · "
                   f"{songs['cluster'].nunique()} clusters")

    colours = colour_map(names, order, colour_mode)

    t1, t2, t3, t4 = st.tabs(["Overview", "Explore clusters", "Find a song", "How it works"])
    with t1:
        tab_overview(songs, profile, names, order, colours)
    with t2:
        tab_explore(songs, profile, names, order, colours)
    with t3:
        tab_song_finder(songs, names)
    with t4:
        tab_how_it_works()


if __name__ == "__main__":
    main()
