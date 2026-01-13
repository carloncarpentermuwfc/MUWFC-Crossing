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
    """Vertical pitch (attack up): x axis = width (0..80), y axis = length (0..120)"""
    shapes = []
    shapes.append(dict(type="rect", x0=0, y0=0, x1=width, y1=length, line=dict(width=2)))
    shapes.append(dict(type="line", x0=0, y0=length / 2, x1=width, y1=length / 2, line=dict(width=2)))

    pa_y = 18
    pa_x0, pa_x1 = 18, 62
    shapes.append(dict(type="rect", x0=pa_x0, y0=0, x1=pa_x1, y1=pa_y, line=dict(width=2)))
    shapes.append(dict(type="rect", x0=pa_x0, y0=length - pa_y, x1=pa_x1, y1=length, line=dict(width=2)))

    sb_y = 6
    sb_x0, sb_x1 = 30, 50
    shapes.append(dict(type="rect", x0=sb_x0, y0=0, x1=sb_x1, y1=sb_y, line=dict(width=2)))
    shapes.append(dict(type="rect", x0=sb_x0, y0=length - sb_y, x1=sb_x1, y1=length, line=dict(width=2)))

    shapes.append(
        dict(
            type="circle",
            x0=width / 2 - 10,
            y0=length / 2 - 10,
            x1=width / 2 + 10,
            y1=length / 2 + 10,
            line=dict(width=2),
        )
    )
    return shapes


def make_segment_trace(x0, y0, x1, y1, color: str):
    """Many line segments separated by NaN."""
    xs = np.column_stack([x0, x1, np.full_like(x0, np.nan)]).ravel()
    ys = np.column_stack([y0, y1, np.full_like(y0, np.nan)]).ravel()
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(width=2, color=color),
        hoverinfo="skip",
        showlegend=False,
    )


def classify_zone(start_x: float, start_y: float) -> str:
    """
    Zone classification from your diagram, using StatsBomb coords:
      X 0..120 (length), Y 0..80 (width)
    """
    if start_x < 80:
        return "Other"

    wide = (start_y <= 20) or (start_y >= 60)
    halfspace = ((20 < start_y <= 34) or (46 <= start_y < 60))

    if wide and start_x >= 114:
        return "Cutbacks & stand ups"
    if wide and 108 <= start_x < 114:
        return "Driven"
    if wide and 96 <= start_x < 108:
        return "Whipped"
    if halfspace and 84 <= start_x < 108:
        return "Diagonals"

    return "Other"


def add_zone_overlays(fig: go.Figure, attack_up: bool, flip_len: bool):
    """Translucent start-zone overlays for visual reference."""
    def maybe_flip_x(x):
        return 120 - x if flip_len else x

    def rect(xmin, xmax, ymin, ymax):
        x0, x1 = maybe_flip_x(xmin), maybe_flip_x(xmax)
        x_low, x_high = (min(x0, x1), max(x0, x1))

        if attack_up:  # x=Y, y=X
            return dict(
                type="rect",
                x0=ymin,
                x1=ymax,
                y0=x_low,
                y1=x_high,
                line=dict(width=1, dash="dot"),
                opacity=0.15,
            )
        else:  # x=X, y=Y
            return dict(
                type="rect",
                x0=x_low,
                x1=x_high,
                y0=ymin,
                y1=ymax,
                line=dict(width=1, dash="dot"),
                opacity=0.15,
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
st.title("Crossing Dashboard (Start + End Location Filters)")

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

    # Team
    teams = sorted(df["Team"].dropna().unique()) if "Team" in df.columns else []
    team = None
    if teams:
        default_team = "Manchester United" if "Manchester United" in teams else teams[0]
        team = st.selectbox("Team", teams, index=teams.index(default_team))

    # Height
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

    # View options
    st.subheader("View options")
    attack_up = st.toggle("Attack upwards (rotate pitch)", value=True)
    flip_direction = st.toggle(
        "Flip direction",
        value=False,
        help="Mirrors pitch length (X) so zones/filters match what you see.",
    )
    show_zone_overlays = st.toggle("Show start-zone overlays", value=True)
    show_endpoints = st.toggle("Show endpoints", value=False)
    show_table = st.toggle("Show data table", value=True)

    # START ZONE filter
    st.subheader("Start zone filter")
    zone_options = ["Diagonals", "Whipped", "Driven", "Cutbacks & stand ups", "Other"]
    selected_zones = st.multiselect(
        "Start Zones",
        zone_options,
        default=["Diagonals", "Whipped", "Driven", "Cutbacks & stand ups"],
    )

    # END LOCATION filters (sliders)
    st.subheader("End location filters")
    use_end_filters = st.toggle("Enable end-location filtering", value=False)
    end_x_range = (0.0, 120.0)
    end_y_range = (0.0, 80.0)
    in_box_only = False
    if use_end_filters:
        end_x_range = st.slider("End X range (0–120)", 0.0, 120.0, (0.0, 120.0), 1.0)
        end_y_range = st.slider("End Y range (0–80)", 0.0, 80.0, (0.0, 80.0), 1.0)
        in_box_only = st.toggle("End location inside penalty area only", value=False)


# ----------------------------
# Apply filters
# ----------------------------
f = df.copy()

if team and "Team" in f.columns:
    f = f[f["Team"] == team]

# all rows are crosses per your data

if selected_heights is not None and "Pass height" in f.columns:
    f = f[f["Pass height"].isin(selected_heights)]

if selected_outcomes is not None and "Outcome" in f.columns:
    f = f[f["Outcome"].isin(selected_outcomes)]

if selected_matches is not None and "Match" in f.columns:
    f = f[f["Match"].isin(selected_matches)]

if selected_players is not None and "Player" in f.columns:
    f = f[f["Player"].isin(selected_players)]

f = f.dropna(subset=["Start X", "Start Y", "End X", "End Y"]).copy()

# Flip length for plotting + classification if requested
if flip_direction:
    f["_sx"] = 120 - f["Start X"]
    f["_ex"] = 120 - f["End X"]
else:
    f["_sx"] = f["Start X"]
    f["_ex"] = f["End X"]

f["_sy"] = f["Start Y"]
f["_ey"] = f["End Y"]

# Start zone assignment (based on START only)
f["Start Zone"] = [
    classify_zone(x, y) for x, y in zip(f["_sx"].to_numpy(), f["_sy"].to_numpy())
]

# Start zone filter
if selected_zones:
    f = f[f["Start Zone"].isin(selected_zones)]
else:
    f = f.iloc[0:0]

# End location filtering (based on END)
if use_end_filters:
    f = f[
        (f["_ex"].between(end_x_range[0], end_x_range[1]))
        & (f["_ey"].between(end_y_range[0], end_y_range[1]))
    ]
    if in_box_only:
        # Penalty area on StatsBomb pitch: X >= 102 and Y between 18 and 62 (attacking end)
        # If flipped, penalty area is still at high X because we already flipped _ex.
        f = f[(f["_ex"] >= 102) & (f["_ey"].between(18, 62))]

# Success flag for colouring
if "Outcome" in f.columns:
    success = f["Outcome"].astype(str).str.lower().eq("complete").to_numpy()
else:
    # If Outcome missing, treat all as "unsuccessful" for colouring consistency
    success = np.zeros(len(f), dtype=bool)

# ----------------------------
# Coordinate transforms for plot
# Attack up: x = Y, y = X
# ----------------------------
pitch_len, pitch_wid = 120, 80

plot_start_x = f["_sx"].to_numpy()
plot_start_y = f["_sy"].to_numpy()
plot_end_x = f["_ex"].to_numpy()
plot_end_y = f["_ey"].to_numpy()

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

    # Layout (vertical pitch only; keeping your existing view)
    fig.update_layout(
        xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False),
        shapes=build_pitch_shapes_vertical(length=pitch_len, width=pitch_wid) if attack_up else [],
        margin=dict(l=10, r=10, t=10, b=10),
    )

    # If not attack_up, draw horizontal pitch shapes quickly
    if not attack_up:
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
        shapes.append(dict(type="circle", x0=pitch_len / 2 - 10, y0=pitch_wid / 2 - 10, x1=pitch_len / 2 + 10, y1=pitch_wid / 2 + 10, line=dict(width=2)))
        fig.update_layout(shapes=shapes)

    if show_zone_overlays:
        add_zone_overlays(fig, attack_up=attack_up, flip_len=flip_direction)

    # Split into success/fail traces so we can colour GREEN / RED
    sx_s, sy_s, ex_s, ey_s = sx[success], sy[success], ex[success], ey[success]
    sx_f, sy_f, ex_f, ey_f = sx[~success], sy[~success], ex[~success], ey[~success]

    # Lines
    if len(sx_s) > 0:
        fig.add_trace(make_segment_trace(sx_s, sy_s, ex_s, ey_s, color="green"))
    if len(sx_f) > 0:
        fig.add_trace(make_segment_trace(sx_f, sy_f, ex_f, ey_f, color="red"))

    # Start markers
    if len(sx_s) > 0:
        fig.add_trace(go.Scattergl(
            x=sx_s, y=sy_s,
            mode="markers",
            marker=dict(size=7, opacity=0.85, color="green"),
            hovertext=np.array(hover_text, dtype=object)[success],
            hoverinfo="text",
            showlegend=False
        ))
    if len(sx_f) > 0:
        fig.add_trace(go.Scattergl(
            x=sx_f, y=sy_f,
            mode="markers",
            marker=dict(size=7, opacity=0.85, color="red"),
            hovertext=np.array(hover_text, dtype=object)[~success],
            hoverinfo="text",
            showlegend=False
        ))

    # Optional endpoints (keep neutral; or you can also color them similarly)
    if show_endpoints and len(f) > 0:
        fig.add_trace(go.Scattergl(
            x=ex, y=ey,
            mode="markers",
            marker=dict(size=6, opacity=0.6, symbol="x"),
            hoverinfo="skip",
            showlegend=False
        ))

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Crosses", f"{len(f):,}")
    with m2:
        st.metric("Completed", f"{int(success.sum()):,}")
    with m3:
        if "Distance" in f.columns and len(f) > 0:
            st.metric("Avg distance", f"{f['Distance'].mean():.1f}")
        else:
            st.metric("Avg distance", "—")

    # Table + download
    if show_table:
        st.subheader("Filtered data")
        drop_cols = [c for c in ["_sx", "_sy", "_ex", "_ey"] if c in f.columns]
        st.dataframe(f.drop(columns=drop_cols), use_container_width=True, height=350)

    drop_cols = [c for c in ["_sx", "_sy", "_ex", "_ey"] if c in f.columns]
    csv_out = f.drop(columns=drop_cols).to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_out, file_name="filtered_crosses.csv", mime="text/csv")
