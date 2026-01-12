import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Crossing Dashboard", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize booleans if they came in as strings
    for c in ["In attacking third", "In attacking half", "Inside attacking third", "Inside attacking half"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
    return df

def build_pitch_shapes_vertical(length=120, width=80):
    """
    Vertical pitch (attack upwards): x axis = width (0..80), y axis = length (0..120)
    """
    shapes = []

    # Outer boundaries
    shapes.append(dict(type="rect", x0=0, y0=0, x1=width, y1=length, line=dict(width=2)))

    # Halfway line
    shapes.append(dict(type="line", x0=0, y0=length/2, x1=width, y1=length/2, line=dict(width=2)))

    # Penalty areas (bottom + top)
    # Standard (StatsBomb): penalty area depth 18, width from 18..62 when width=80
    pa_y = 18
    pa_x0, pa_x1 = 18, 62

    # Bottom box
    shapes.append(dict(type="rect", x0=pa_x0, y0=0, x1=pa_x1, y1=pa_y, line=dict(width=2)))
    # Top box
    shapes.append(dict(type="rect", x0=pa_x0, y0=length-pa_y, x1=pa_x1, y1=length, line=dict(width=2)))

    # 6-yard boxes (depth 6, width 20 (30..50))
    sb_y = 6
    sb_x0, sb_x1 = 30, 50
    shapes.append(dict(type="rect", x0=sb_x0, y0=0, x1=sb_x1, y1=sb_y, line=dict(width=2)))
    shapes.append(dict(type="rect", x0=sb_x0, y0=length-sb_y, x1=sb_x1, y1=length, line=dict(width=2)))

    # Center circle (radius 10)
    shapes.append(dict(type="circle", x0=width/2-10, y0=length/2-10, x1=width/2+10, y1=length/2+10, line=dict(width=2)))

    return shapes

def make_segment_trace(x0, y0, x1, y1, customdata, name="Cross", showlegend=False):
    """
    Create a single Scattergl trace containing many line segments separated by None.
    """
    xs = np.column_stack([x0, x1, np.full_like(x0, np.nan)]).ravel()
    ys = np.column_stack([y0, y1, np.full_like(y0, np.nan)]).ravel()

    # customdata must align with points; we only want hover on the start point markers, not on lines.
    # For lines, we set hoverinfo="skip".
    return go.Scattergl(
        x=xs, y=ys,
        mode="lines",
        line=dict(width=2),
        hoverinfo="skip",
        name=name,
        showlegend=showlegend
    )

# ----------------------------
# UI: Data source
# ----------------------------
st.title("Crossing Dashboard (Interactive Pitch Map)")

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

    # Basic validation
    required_cols = ["Start X", "Start Y", "End X", "End Y"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Sidebar-like controls
    st.subheader("Filters")

    # Team filter (if available)
    teams = sorted(df["Team"].dropna().unique()) if "Team" in df.columns else []
    team = None
    if teams:
        default_team = "Manchester United" if "Manchester United" in teams else teams[0]
        team = st.selectbox("Team", teams, index=teams.index(default_team))
    else:
        st.info("No Team column found; showing all rows.")

    # Cross height filter
    pass_heights = sorted(df["Pass height"].dropna().unique()) if "Pass height" in df.columns else []
    height_mode = None
    if pass_heights:
        height_mode = st.radio(
            "Cross height",
            ["All", "High only", "Low + Ground only", "Custom"],
            index=0,
            horizontal=False
        )
        if height_mode == "Custom":
            selected_heights = st.multiselect("Select heights", pass_heights, default=pass_heights)
        elif height_mode == "High only":
            selected_heights = ["high"] if "high" in pass_heights else pass_heights[:1]
        elif height_mode == "Low + Ground only":
            selected_heights = [h for h in ["low", "ground"] if h in pass_heights]
            if not selected_heights:  # fallback
                selected_heights = pass_heights
        else:
            selected_heights = pass_heights
    else:
        selected_heights = None
        st.info("No Pass height column found; height filter disabled.")

    # Outcome filter
    outcomes = sorted(df["Outcome"].dropna().unique()) if "Outcome" in df.columns else []
    selected_outcomes = st.multiselect("Outcome", outcomes, default=outcomes) if outcomes else None

    # Match filter
    matches = sorted(df["Match"].dropna().unique()) if "Match" in df.columns else []
    selected_matches = st.multiselect("Match", matches, default=matches) if matches else None

    # Player filter
    players = sorted(df["Player"].dropna().unique()) if "Player" in df.columns else []
    selected_players = st.multiselect("Player", players, default=players) if players else None

    st.subheader("View options")
    attack_up = st.toggle("Attack upwards (rotate pitch)", value=True, help="Rotate so the attacking goal is at the top.")
    flip_direction = st.toggle("Flip direction", value=False, help="Mirror the pitch so the team attacks downwards instead.")

    show_endpoints = st.toggle("Show endpoints", value=False)
    show_table = st.toggle("Show data table", value=True)

# ----------------------------
# Apply filters
# ----------------------------
f = df.copy()

if team and "Team" in f.columns:
    f = f[f["Team"] == team]

# User note: "every event is a cross" — so we don't filter by Event type unless user wants later.
if selected_heights is not None and "Pass height" in f.columns:
    f = f[f["Pass height"].isin(selected_heights)]

if selected_outcomes is not None and "Outcome" in f.columns and len(outcomes) > 0:
    f = f[f["Outcome"].isin(selected_outcomes)]

if selected_matches is not None and "Match" in f.columns and len(matches) > 0:
    f = f[f["Match"].isin(selected_matches)]

if selected_players is not None and "Player" in f.columns and len(players) > 0:
    f = f[f["Player"].isin(selected_players)]

# Drop rows without coordinates
f = f.dropna(subset=["Start X", "Start Y", "End X", "End Y"]).copy()

# ----------------------------
# Coordinate transforms
# Original: Start X in [0..120] (length), Start Y in [0..80] (width)
# Vertical pitch (attack up): x = Start Y, y = Start X
# ----------------------------
if attack_up:
    sx, sy = f["Start Y"].to_numpy(), f["Start X"].to_numpy()
    ex, ey = f["End Y"].to_numpy(), f["End X"].to_numpy()
    pitch_len, pitch_wid = 120, 80
    x_range, y_range = (0, pitch_wid), (0, pitch_len)
else:
    # Horizontal (traditional): x = Start X, y = Start Y
    sx, sy = f["Start X"].to_numpy(), f["Start Y"].to_numpy()
    ex, ey = f["End X"].to_numpy(), f["End Y"].to_numpy()
    pitch_len, pitch_wid = 120, 80
    x_range, y_range = (0, pitch_len), (0, pitch_wid)

# Flip direction if requested (mirror over center line of length dimension)
if flip_direction:
    if attack_up:
        # y is length
        sy = pitch_len - sy
        ey = pitch_len - ey
    else:
        # x is length
        sx = pitch_len - sx
        ex = pitch_len - ex

# Build hover text for start markers
hover_cols = [c for c in ["Player", "Recipient player", "Outcome", "Pass height", "Match", "Date", "Minute", "Second"] if c in f.columns]
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
        # Horizontal pitch shapes (reuse vertical by swapping axes via transform)
        # For simplicity, approximate by drawing the same geometry but rotated.
        # Outer boundaries
        shapes = []
        shapes.append(dict(type="rect", x0=0, y0=0, x1=pitch_len, y1=pitch_wid, line=dict(width=2)))
        shapes.append(dict(type="line", x0=pitch_len/2, y0=0, x1=pitch_len/2, y1=pitch_wid, line=dict(width=2)))
        # Penalty areas (left + right)
        pa_x = 18
        pa_y0, pa_y1 = 18, 62
        shapes.append(dict(type="rect", x0=0, y0=pa_y0, x1=pa_x, y1=pa_y1, line=dict(width=2)))
        shapes.append(dict(type="rect", x0=pitch_len-pa_x, y0=pa_y0, x1=pitch_len, y1=pa_y1, line=dict(width=2)))
        # 6-yard
        sb_x = 6
        sb_y0, sb_y1 = 30, 50
        shapes.append(dict(type="rect", x0=0, y0=sb_y0, x1=sb_x, y1=sb_y1, line=dict(width=2)))
        shapes.append(dict(type="rect", x0=pitch_len-sb_x, y0=sb_y0, x1=pitch_len, y1=sb_y1, line=dict(width=2)))
        # Center circle
        shapes.append(dict(type="circle", x0=pitch_len/2-10, y0=pitch_wid/2-10, x1=pitch_len/2+10, y1=pitch_wid/2+10, line=dict(width=2)))

        fig.update_layout(
            xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False),
            shapes=shapes,
            margin=dict(l=10, r=10, t=10, b=10),
        )

    # Lines (cross paths)
    fig.add_trace(make_segment_trace(sx, sy, ex, ey, customdata=None))

    # Start points
    fig.add_trace(go.Scattergl(
        x=sx, y=sy,
        mode="markers",
        marker=dict(size=7, opacity=0.75),
        hovertext=hover_text,
        hoverinfo="text",
        name="Start"
    ))

    # Optional endpoints
    if show_endpoints:
        fig.add_trace(go.Scattergl(
            x=ex, y=ey,
            mode="markers",
            marker=dict(size=6, opacity=0.6, symbol="x"),
            hoverinfo="skip",
            name="End"
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
        if "Distance" in f.columns:
            st.metric("Avg distance", f"{f['Distance'].mean():.1f}")
        else:
            st.metric("Avg distance", "—")

    # Data table + download
    if show_table:
        st.subheader("Filtered data")
        st.dataframe(f, use_container_width=True, height=350)

    csv_out = f.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_out, file_name="filtered_crosses.csv", mime="text/csv")
