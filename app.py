import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -----------------------------
# Config: column names
# -----------------------------
COL_TEAM = "team_name"
COL_PLAYER = "player_id"
COL_PLAYER_NAME = "player_name"  # optional; app falls back to player_id if missing
COL_ACTION = "action"
COL_IS_CROSS = "is_cross"

COL_SX = "start_adj_x"
COL_SY = "start_adj_y"

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


# -----------------------------
# Helpers
# -----------------------------
def to_105x68_option2(x_52_5: pd.Series, y_34: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert from 52.5x34-centered system to a 105x68 pitch (Option 2)."""
    x = x_52_5 + 52.5
    y = 34.0 - y_34
    return x, y


def is_cross_row(df: pd.DataFrame) -> pd.Series:
    """Identify crosses using is_cross if present, otherwise action contains 'cross'."""
    if COL_IS_CROSS in df.columns:
        return df[COL_IS_CROSS].fillna(0).astype(int) == 1
    if COL_ACTION in df.columns:
        return df[COL_ACTION].fillna("").str.contains("cross", case=False, na=False)
    # fallback: nothing
    return pd.Series([False] * len(df), index=df.index)


def draw_pitch(ax: plt.Axes, show_grid: bool = True) -> None:
    """Draw a standard 105x68 football pitch (top-down)."""
    # Background
    ax.set_facecolor("white")

    # Outer boundaries
    ax.plot([0, PITCH_LENGTH], [0, 0], color="black", lw=1.5)
    ax.plot([0, PITCH_LENGTH], [PITCH_WIDTH, PITCH_WIDTH], color="black", lw=1.5)
    ax.plot([0, 0], [0, PITCH_WIDTH], color="black", lw=1.5)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH], [0, PITCH_WIDTH], color="black", lw=1.5)

    # Halfway line
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH], color="black", lw=1.5)

    # Centre circle + spot
    centre = (PITCH_LENGTH / 2, PITCH_WIDTH / 2)
    cc = plt.Circle(centre, 9.15, fill=False, color="black", lw=1.2)
    ax.add_patch(cc)
    ax.plot(centre[0], centre[1], marker="o", color="black", ms=3)

    # Penalty areas (16.5m deep, 40.32m wide)
    box_w = 40.32
    box_d = 16.5
    y0 = (PITCH_WIDTH - box_w) / 2

    # Left penalty area
    ax.plot([0, box_d], [y0, y0], color="black", lw=1.2)
    ax.plot([0, box_d], [y0 + box_w, y0 + box_w], color="black", lw=1.2)
    ax.plot([box_d, box_d], [y0, y0 + box_w], color="black", lw=1.2)

    # Right penalty area
    ax.plot([PITCH_LENGTH, PITCH_LENGTH - box_d], [y0, y0], color="black", lw=1.2)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH - box_d], [y0 + box_w, y0 + box_w], color="black", lw=1.2)
    ax.plot([PITCH_LENGTH - box_d, PITCH_LENGTH - box_d], [y0, y0 + box_w], color="black", lw=1.2)

    # 6-yard boxes (5.5m deep, 18.32m wide)
    six_w = 18.32
    six_d = 5.5
    y1 = (PITCH_WIDTH - six_w) / 2

    # Left 6-yard
    ax.plot([0, six_d], [y1, y1], color="black", lw=1.0)
    ax.plot([0, six_d], [y1 + six_w, y1 + six_w], color="black", lw=1.0)
    ax.plot([six_d, six_d], [y1, y1 + six_w], color="black", lw=1.0)

    # Right 6-yard
    ax.plot([PITCH_LENGTH, PITCH_LENGTH - six_d], [y1, y1], color="black", lw=1.0)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH - six_d], [y1 + six_w, y1 + six_w], color="black", lw=1.0)
    ax.plot([PITCH_LENGTH - six_d, PITCH_LENGTH - six_d], [y1, y1 + six_w], color="black", lw=1.0)

    # Penalty spots
    ax.plot(11, PITCH_WIDTH / 2, marker="o", color="black", ms=3)
    ax.plot(PITCH_LENGTH - 11, PITCH_WIDTH / 2, marker="o", color="black", ms=3)

    # Penalty arcs (outside the box)
    # Left arc: x = 16.5 + sqrt(r^2 - (y-34)^2)
    r = 9.15
    yy = np.linspace((PITCH_WIDTH / 2) - r, (PITCH_WIDTH / 2) + r, 300)
    xx_left = box_d + np.sqrt(np.maximum(0, r**2 - (yy - (PITCH_WIDTH / 2)) ** 2))
    xx_right = (PITCH_LENGTH - box_d) - np.sqrt(np.maximum(0, r**2 - (yy - (PITCH_WIDTH / 2)) ** 2))
    ax.plot(xx_left, yy, color="black", lw=1.0)
    ax.plot(xx_right, yy, color="black", lw=1.0)

    # Optional grid
    if show_grid:
        for gx in np.arange(0, PITCH_LENGTH + 0.1, 10):
            ax.plot([gx, gx], [0, PITCH_WIDTH], color="lightgray", lw=0.6, ls="--", alpha=0.4)
        for gy in np.arange(0, PITCH_WIDTH + 0.1, 10):
            ax.plot([0, PITCH_LENGTH], [gy, gy], color="lightgray", lw=0.6, ls="--", alpha=0.4)

    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_heatmap(ax: plt.Axes, x: np.ndarray, y: np.ndarray, method: str, bins: int = 24) -> None:
    """Overlay a heatmap of start locations on the pitch."""
    if len(x) == 0:
        return

    if method == "KDE (smooth)" and SCIPY_OK and len(np.unique(np.vstack([x, y]).T, axis=0)) >= 3:
        # KDE on a grid
        xi, yi = np.mgrid[0:PITCH_LENGTH:300j, 0:PITCH_WIDTH:300j]
        vals = np.vstack([x, y])

        try:
            kde = gaussian_kde(vals, bw_method=0.35)
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            ax.contourf(xi, yi, zi, levels=20, cmap="YlOrRd", alpha=0.6)
            ax.contour(xi, yi, zi, levels=8, colors="white", linewidths=0.8, alpha=0.5)
            return
        except Exception:
            # fallback to histogram below
            pass

    # Histogram heatmap (always works)
    H, xedges, yedges = np.histogram2d(x, y, bins=[bins, bins], range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]])
    # Transpose to match x/y axes
    ax.imshow(
        H.T,
        origin="lower",
        extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
        cmap="YlOrRd",
        alpha=0.6,
        aspect="auto",
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Cross Start Heatmaps", layout="wide")
st.title("Cross Start-Location Heatmaps (Manchester United Women)")

uploaded = st.file_uploader("Upload your event CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)

# Basic checks
missing = [c for c in [COL_TEAM, COL_PLAYER, COL_SX, COL_SY] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.write("Columns found:", list(df.columns))
    st.stop()

# Filter team
teams = sorted(df[COL_TEAM].dropna().unique().tolist())
default_team = "Manchester United Women" if "Manchester United Women" in teams else (teams[0] if teams else None)
team = st.selectbox("Team", teams, index=teams.index(default_team) if default_team in teams else 0)

team_df = df[df[COL_TEAM] == team].copy()
team_df = team_df[is_cross_row(team_df)].copy()

if team_df.empty:
    st.warning("No crosses found for this team with the current cross-identification rules.")
    st.stop()

# Player selection
if COL_PLAYER_NAME in team_df.columns:
    # nice label
    team_df["_player_label"] = team_df[COL_PLAYER_NAME].fillna(team_df[COL_PLAYER].astype(str))
else:
    team_df["_player_label"] = team_df[COL_PLAYER].astype(str)

player_labels = sorted(team_df["_player_label"].unique().tolist())
player_label = st.selectbox("Player", player_labels)

p_df = team_df[team_df["_player_label"] == player_label].copy()

st.sidebar.header("Heatmap settings")
method = st.sidebar.selectbox("Heatmap method", ["KDE (smooth)" if SCIPY_OK else "Histogram (binned)", "Histogram (binned)"])
bins = st.sidebar.slider("Histogram bins (if using histogram)", 10, 50, 24, 1)
show_grid = st.sidebar.checkbox("Show pitch grid", value=True)

st.sidebar.header("Coordinate conversion")
st.sidebar.write("Your starts are assumed to be from 52.5×34. Using Option 2 conversion.")
# (If needed later, you can add alternate conversions/toggles)

# Convert coordinates to 105x68
x, y = to_105x68_option2(p_df[COL_SX].astype(float), p_df[COL_SY].astype(float))
# Clip safely
x = x.clip(0, PITCH_LENGTH).to_numpy()
y = y.clip(0, PITCH_WIDTH).to_numpy()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
draw_pitch(ax, show_grid=show_grid)
plot_heatmap(ax, x, y, method=method, bins=bins)

ax.set_title(f"{team} — {player_label}\nCross start locations (n={len(p_df)})", fontsize=14)

st.pyplot(fig, clear_figure=True)

with st.expander("Show filtered data (crosses only)"):
    st.dataframe(p_df.drop(columns=["_player_label"], errors="ignore"))
