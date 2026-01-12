import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Crossing Dashboard", layout="wide")


# ----------------------------
# Data + helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pitch_shapes_vertical(length=120, width=80):
    """
    Vertical pitch (attack upwards): x axis = width (0..80), y axis = length (0..120)
    """
    shapes = []
    # Outer boundaries
    shapes.append(dict(type="rect", x0=0, y0=0, x1=width, y1=length, line=dict(width=2)))
    # Halfway line
    shapes.append(dict(type="line", x0=0, y0=length / 2, x1=width, y1=length / 2, line=dict(width=2)))

    # Penalty areas (bottom + top)
    pa_y = 18
    pa_x0, pa_x1 = 18, 62
    shapes.append(dict(type="rect", x0=pa_x0, y0=0, x1=pa_x1, y1=pa_y, line=dict(width=2)))
    shapes.append(dict(type="rect", x0=pa_x0, y0=length - pa_y, x1=pa_x1, y1=length, line=dict(width=2)))

    # 6-yard boxes (bottom + top)
    sb_y = 6
    sb_x0, sb_x1 = 30, 50
    shapes.append(dict(type="rect", x0=sb_x0, y0=0, x1=sb_x1, y1=sb_y, line=dict(width=2)))
    shapes.append(dict(type="rect", x0=sb_x0, y0=length - sb_y, x1=sb_x1, y1=length, line=dict(width=2)))

    # Center circle
    shapes.append(
        dict(type="circle", x0=width / 2 - 10, y0=length / 2 - 10, x1=width / 2 + 10, y1=length / 2 + 10,
             line=dict(width=2))
    )
    return shapes


def make_segment_trace(x0, y0, x1, y1, name="Cross"):
    """
    Create a single Scattergl trace containing many line segments separated by NaN.
    """
    xs = np.column_stack([x0, x1, np.full_like(x0, np.nan)]).ravel()
    ys = np.column_stack([y0, y1, np.full_like(y0, np.nan)]).ravel()

    return go.Scattergl(
        x=xs, y=ys,
        mode="lines",
        line=dict(width=2),
        hoverinfo="skip",
        showlegend=False,
        name=name
    )


def classify_zone(start_x: float, start_y: float) -> str:
    """
    Classify cross *start location* into zones (StatsBomb coords):
      - Start X: 0..120 (length)
      - Start Y: 0..80 (width)

    Approximation to your diagram:
      - Diagonals: half-space areas outside the box
      - Whipped: wide channel outside box
      - Driven: wide channel closer to byline
      - Cutbacks & stand ups: very close to byline, wide
    """
    # If not in the attacking third-ish, label as Other
    if start_x < 80:
        return "Other"

    wide = (start_y <= 20) or (start_y >= 60)
    halfspace = ((20 < start_y <= 34) or (46 <= start_y < 60))

    # Very close to byline
    if wide and start_x >= 114:
        return "Cutbacks & stand ups"

    # Close to byline
    if wide and 108 <= start_x < 114:
        return "Driven"

    # Wider but not right on byline
    if wide and 96 <= start_x < 108:
        return "Whipped"

    # Half-space crossing area
    if halfspace and 84 <= start_x < 108:
        return "Diagonals"

    return "Other"


def add_zone_overlays(fig: go.Figure, attack_up: bool, flip_len: bool):
    """
    Add translucent zone rectangles for visual reference.
    Zones are defined in original coords (X length, Y width).
    """
    def maybe_flip_x(x):
        return 120 - x if flip_len else x

    def rect(xmin, xmax, ymin, ymax):
        # flip length if needed
        x0, x1 = maybe_flip_x(xmin), maybe_flip_x(xmax)
        x_low, x_high = (min(x0, x1), max(x0, x1))

        if attack_up:
            # vertical pitch: plot x=Y, y=X
            return dict(
                type="rect",
                x0=ymin, x1=ymax,
                y0=x_low, y1=x_high,
                line=dict(width=1, dash="dot"),
                opacity=0.15
            )
        else:
            # horizontal pitch: x=X, y=Y
            return dict(
                type="rect",
                x0=x_low, x1=x_high,
                y0=ymin, y1=ymax,
                line=dict(width=1, dash="dot"),
                opacity=0.15
            )

    # Diagonals (half-spaces)
    fig.add_shape(rect(84, 108, 20, 34))
    fig.add_shape(rect(84, 108, 46, 60))

    # Whipped (wide)
    fig.add_shape(rect(96, 108, 0, 20))
    fig.add_shape(rect(96, 108, 60, 80))

    # Driven (wide, closer)
    fig.add_shape(rect(108, 114, 0, 20))
    fig.add_shape(rect(108, 114, 60, 80))

    # Cutbacks & stand ups (very close)
    fig.add_shape(rect(114, 120, 0, 20))
    fig.add_shape(rect(114, 120, 60, 80))


# ----------------------------
# App
# ----------------------------
st.title("Crossing Dashboard (Zone-based start location filters)")

left, right = st.columns([1, 2], vertical_alignment="top")

with left:
    st.subheader("Data")
    uploaded = st.file_uploader("Upload Events CSV (optional)", type=["csv"])
    default_path = "Events.csv"

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.caption("Using uploaded file.")
    else:
        try:
            df = load_csv(default_path)
            st.caption(f"Using local file: {default_path}")
        except Exception:
            st.error("No file uploaded and Events.csv not found. Please upload your CSV.")
            st.stop()

    required_cols = ["Start X", "Start Y", "End X", "End Y"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    st.subheader("Filters")

    # Team filter
    teams = sorted(df["Team"].dropna().unique()) if "Team" in df.columns else []
    team = None
    if teams:
        default_team = "Manchester United" if "Manchester United" in teams else teams[0]
        team = st.selectbox("Team", teams, index=teams.index(default_team))

    # Cross height filter
    pass_heights = sorted(df["Pass height"].dropna().unique()) if "Pass height" in df.columns else []
    selected_heights = None
    if pass_heights:
        height_mode = st.radio("Cross height", ["All", "High only", "Low + Ground only", "Custom"], index=0)
        if height_mode == "Custom":
            selected_heights = st.multiselect("Select heights", pass_heights, default=pass_heights)
        elif height_mode == "High only":
            selected_heights = ["high"] if "high" in pass_heights else pass_heights[:1]
        elif height_mode == "Low + Ground only":
            selected_heights = [h for h in ["low", "ground"] if h in pass_heights] or pass_heights
        else:
            selected_heights = pass_heights

    # Outcome / Match / Player
    outcomes = sorted(df["Outcome"].dropna().unique()) if "Outcome" in df.columns else []
    selected_outcomes = st.multiselect("Outcome", outcomes, default=outcomes) if outcomes else None

    matches = sorted(df["Match"].dropna().unique()) if "Match" in df.columns else []
    selected_matches = st.multiselect("Match", matches, default=matches) if matches else None

    players = sorted(df["Player"].dropna().unique()) if "Player" in df.columns else []
    selected_players = st.multiselect("Player", players, default=players) if players else None

    st.subheader("Start location zone filter")
    st.caption("Zones are assigned from Start X/Start Y (your data coordinates).")

    # View options
    attack_up = st.toggle("Attack upwards (rotate pitch)", value=True)
    flip_direction = st.toggle(
        "Flip direction",
        value=False,
        help="If your data is opposite, this mirrors pitch length so the team attacks the other way."
    )
    show_zone_overlays = st.toggle("Show zone overlays", value=True)
    show_endpoints = st.toggle("Show endpoints", value=False)
    show_table = st.toggle("Show data table", value=True)

# ----------------------------
# Apply filters
# ----------------------------
f = df.copy()

if team and "Team" in f.columns:
    f = f[f["Team"] == team]

# All rows are crosses (per your data) — no Event type filtering.

if selected_heights is not None and "Pass height" in f.columns:
    f = f[f["Pass height"].isin(selected_heights)]

if selected_outcomes is not None and "Outcome" in f.columns:
    f = f[f["Outcome"].isin(selected_outcomes)]

if selected_matches is not None and "Match" in f.columns:
    f = f[f["Match"].isin(selected_matches)]

if selected_players is not None and "Player" in f.columns:
    f = f[f["Player"].isin(selected_players)]

f = f.dropna(subset=["Start X", "Start Y", "End X", "End Y"]).copy()

# If flipping direction, mirror *length* (X axis in StatsBomb coords).
# We use these mirrored values for both zone assignment and plotting so it matches what you see.
if flip_direction:
    f["_sx_plot"] = 120 - f["Start X"]
    f["_ex_plot"] = 120 - f["End X"]
else:
    f["_sx_plot"] = f["Start X"]
    f["_ex_plot"] = f["End X"]

f["_sy_plot"] = f["Start Y"]
f["_ey_plot"] = f["End Y"]

# Zone assignment (based on start location)
f["Start Zone"] = [
    classify_zone(x, y)
    for x, y in zip(f["_sx_plot"].to_numpy(), f["_sy_plot"].to_numpy())
]

zone_options = ["Diagonals", "Whipped", "Driven", "Cutbacks & stand ups", "Other"]

with left:
    selected_zones = st.multiselect(
        "Zones",
        zone_options,
        default=["Diagonals", "Whipped", "Driven", "Cutbacks & stand ups"]
    )

if selected_zones:
    f = f[f["Start Zone"].isin(selected_zones)]
else:
    f = f.iloc[0:0]

# ----------------------------
# Coordinates for plotting
# Attack up: x = Y, y = X
# ----------------------------
pitch_len, pitch_wid = 120, 80
plot_start_x = f["_sx_plot"].to_numpy()
plot_start_y = f["_sy_plot"].to_numpy()
plot_end_x = f["_ex_plot"].to_numpy()
plot_end_y = f["_ey_plot"].to_numpy()

if attack_up:
    sx, sy = plot_start_y, plot_start_x
    ex, ey = plot_end_y, plot_end_x
    x_range, y_range = (0, pitch_wid), (0, pitch_len)
else:
    sx, sy = plot_start_x, plot_start_y
    ex, ey = plot_end_x, plot_end_y
    x_range, y_range = (0, pitch_len), (0, pitch_wid)

# Hover text
hover_cols = [c for c in ["Start Zone", "Player", "Recipient player", "Outcome", "Pass height", "Match", "Date", "Minute", "Second"] if c in f.columns]


def row_to_hover(row):
    parts = []
    for c in hover_cols:
        v = row.get(c, "")
        if pd.isna(v):
            continue
        parts.append(f"<b>{c}</b>: {v}")
    parts.append(f"<b>Start</b>: ({row['Start X']:.1f}, {row['Start Y']:.1f})")
    parts.append(f"<b>End</b>: ({row['End X']:.1f}, {row['End Y']:.1f})")
    return "<br>".join(parts)


hover_text = f.apply(row_to_hover, axis=1).to_list()

# ----------------------------
# Plot
# ----------------------------
with right:
    st.subheader("Pitch map")
    st.caption(f"Showing {len(f):,} crosses after filters.")

    fig = go.Figure()

    if attack_up:
        fig.update_layout(
            xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False),
            shapes=build_pitch_shapes_vertical(length=pitch_len, width=pitch_wid),
            margin=dict(l=10, r=10, t=10, b=10),
        )
    else:
        # Horizontal pitch shapes
        shapes = []
        shapes.append(dict(type="rect", x0=0, y0=0, x1=pitch_len, y1=pitch_wid, line=dict(width=2)))
        shapes.append(dict(type="line", x0=pitch_len / 2, y0=0, x1=pitch_len / 2, y1=pitch_wid, line=dict(width=2)))
        pa_x = 18
        pa_y0, pa_y1 = 18, 62
        shapes.append(dict(type="rect", x0=0, y0=pa_y0, x1=pa_x, y1=pa_y1, line=dict(width=2)))
        shapes.append(dict(type="rect", x0=pitch_len - pa_x, y0=pa_y0, x1=pitch_len, y1=pa_y1, line=dict(width=2)))
        sb_x = 6
        sb_y0, sb_y1 = 30, 50
        shapes.append(dict(type="rect", x0=0, y0=sb_y0, x1=sb_x, y1=sb_y1, line=dict(width=2)))
        shapes.append(dict(type="rect", x0=pitch_len - sb_x, y0=sb_y0, x1=pitch_len, y1=sb_y1, line=dict(width=2)))
        shapes.append(dict(type="circle", x0=pitch_len / 2 - 10, y0=pitch_wid / 2 - 10, x1=pitch_len / 2 + 10,
                           y1=pitch_wid / 2 + 10, line=dict(width=2)))

        fig.update_layout(
            xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False),
            shapes=shapes,
            margin=dict(l=10, r=10, t=10, b=10),
        )

    if show_zone_overlays:
        add_zone_overlays(fig, attack_up=attack_up, flip_len=flip_direction)

    # Lines
    fig.add_trace(make_segment_trace(sx, sy, ex, ey))

    # Start points
    fig.add_trace(go.Scattergl(
        x=sx, y=sy,
        mode="markers",
        marker=dict(size=7, opacity=0.75),
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=False
    ))

    # Optional endpoints
    if show_endpoints:
        fig.add_trace(go.Scattergl(
            x=ex, y=ey,
            mode="markers",
            marker=dict(size=6, opacity=0.6, symbol="x"),
            hoverinfo="skip",
            showlegend=False
        ))

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Crosses", f"{len(f):,}")
    with m2:
        if "Outcome" in f.columns:
            comp = (f["Outcome"].astype(str).str.lower() == "complete").sum()
            st.metric("Completed", f"{comp:,}")
        else:
            st.metric("Completed", "—")
    with m3:
        if "Distance" in f.columns and len(f) > 0:
            st.metric("Avg distance", f"{f['Distance'].mean():.1f}")
        else:
            st.metric("Avg distance", "—")

    # Table + download
    if show_table:
        st.subheader("Filtered data")
        drop_cols = [c for c in ["_sx_plot", "_sy_plot", "_ex_plot", "_ey_plot"] if c in f.columns]
        st.dataframe(f.drop(columns=drop_cols), use_container_width=True, height=350)

    drop_cols = [c for c in ["_sx_plot", "_sy_plot", "_ex_plot", "_ey_plot"] if c in f.columns]
    csv_out = f.drop(columns=drop_cols).to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_out, file_name="filtered_crosses.csv", mime="text/csv")
