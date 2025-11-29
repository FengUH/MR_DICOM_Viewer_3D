import os

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import streamlit as st

# ============================================================
# 📄 Page configuration
# ============================================================
st.set_page_config(
    page_title="DICOM Viewer",
    page_icon="🩻",  # You can replace this with a custom PNG path
)
st.title("🩻 DICOM Viewer 3D")

# ============================================================
# 📂 Locate the DICOM folder (relative to this file)
# ============================================================
# We assume your DICOM files are stored under:
#   <project_root>/sample_data/MRI_dataset
# so that the app can be shipped as a self-contained folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dicom_dir = os.path.join(BASE_DIR, "sample_data", "MRI_dataset")

if not os.path.exists(dicom_dir):
    st.error(f"❌ DICOM folder does not exist: {dicom_dir}")
    st.stop()

# ============================================================
# 🔧 Helper: sort DICOM files by InstanceNumber
# ============================================================
def get_sorted_dicom_files(dicom_dir: str):
    """
    Scan a folder, keep only .dcm files, and sort them
    by the DICOM tag `InstanceNumber`.

    This ensures slices are in the correct anatomical order.
    """
    items = []

    for fname in os.listdir(dicom_dir):
        if not fname.endswith(".dcm"):
            continue

        path = os.path.join(dicom_dir, fname)

        try:
            # We only read metadata here (no pixel data yet),
            # which is faster and lighter.
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            inst_num = getattr(ds, "InstanceNumber", 0)
        except Exception:
            # If anything goes wrong, fall back to 0 so that
            # the file is still included but may not be well-ordered.
            inst_num = 0

        items.append((inst_num, fname))

    # Sort ascending by InstanceNumber
    items.sort(key=lambda x: x[0])

    # Return just the file names, now in order
    return [f for _, f in items]


# ============================================================
# 📚 Load the full 3D volume and cache the result
# ============================================================
def _cache_func():
    """
    Return the appropriate cache decorator depending on the
    Streamlit version. Newer versions expose `st.cache_data`,
    older ones only have `st.cache`.
    """
    if hasattr(st, "cache_data"):
        # Modern API (no deprecation warning on new versions)
        return st.cache_data(show_spinner=True)
    else:
        # Fallback for older Streamlit releases on Streamlit Cloud
        return st.cache(show_spinner=True)

@_cache_func()
def load_dicom_volume(dicom_dir: str):
    """
    Read all DICOM slices in the given folder, stack them into
    a 3D numpy array, and return both the volume and list of files.

    Returned volume shape: (Z, Y, X).
    """
    files = get_sorted_dicom_files(dicom_dir)
    if len(files) == 0:
        raise RuntimeError("No .dcm files found in the target folder.")

    slices = []
    for fname in files:
        path = os.path.join(dicom_dir, fname)
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slices.append(arr)

    volume = np.stack(slices, axis=0)  # (Z, Y, X)
    return volume, files


# Try loading the volume once; if anything fails, stop the app gracefully.
try:
    volume, dicom_files = load_dicom_volume(dicom_dir)
except Exception as e:
    st.error("❌ Error loading DICOM volume:")
    st.error(str(e))
    st.stop()

# Basic shape information
n_z, n_y, n_x = volume.shape

# ============================================================
# 🎛 Sidebar: orientation and slice navigation
# ============================================================
st.sidebar.header("View & Slice Navigation")

# Human-readable labels for slice orientation
plane_label_list = ["Axial", "Coronal", "Sagittal"]
plane = st.sidebar.radio("Panel Orientation", plane_label_list, index=0)

# Internal keys used for indexing the volume
plane_key_map = {
    "Axial": "axial",
    "Coronal": "coronal",
    "Sagittal": "sagittal",
}
plane_key = plane_key_map[plane]

# Number of slices for each orientation
plane_n_slices = {
    "axial": n_z,     # Z dimension
    "coronal": n_y,   # Y dimension
    "sagittal": n_x,  # X dimension
}

# Keep one slice index per orientation in session_state
if "slice_idx" not in st.session_state:
    st.session_state.slice_idx = {
        "axial": n_z // 2,
        "coronal": n_y // 2,
        "sagittal": n_x // 2,
    }

n_slices = plane_n_slices[plane_key]
cur_idx = st.session_state.slice_idx[plane_key]

# Make sure the index is always within valid bounds
cur_idx = max(0, min(n_slices - 1, cur_idx))
st.session_state.slice_idx[plane_key] = cur_idx

# Simple previous / next buttons for slice navigation
col1, col2 = st.sidebar.columns(2)
if col1.button("⬆ Last"):
    st.session_state.slice_idx[plane_key] = max(0, cur_idx - 1)
if col2.button("⬇ Next"):
    st.session_state.slice_idx[plane_key] = min(n_slices - 1, cur_idx + 1)

cur_idx = st.session_state.slice_idx[plane_key]
st.sidebar.write(f"Current: **{cur_idx + 1} / {n_slices}**")

# ============================================================
# 🎨 Sidebar: contrast window & colormap
# ============================================================
# We apply percentile-based clipping to improve visualization:
# low percentile -> dark background, high percentile -> bright tissue.
contrast_range = st.sidebar.slider(
    "Window Length (Percentile)",
    0.0,
    100.0,
    (5.0, 95.0),
    step=1.0,
    help=(
        "Intensity window based on percentiles. "
        "Lower values increase background detail; "
        "higher values emphasize bright structures."
    ),
)

# Matplotlib colormap selection for the main image
cmap_name = st.sidebar.selectbox(
    "Colormap",
    ["gray", "bone", "hot", "jet", "viridis"],
    index=0,
)

# ============================================================
# 🧠 Extract the current slice from the 3D volume
# ============================================================
if plane_key == "axial":
    # Slice across Z (top-to-bottom view of the head)
    raw_slice = volume[cur_idx, :, :]
elif plane_key == "coronal":
    # Slice across Y (front-to-back view)
    raw_slice = volume[:, cur_idx, :]
else:  # sagittal
    # Slice across X (left-to-right view)
    raw_slice = volume[:, :, cur_idx]

raw = raw_slice.astype(np.float32)

# Percentile-based intensity clipping
vmin_p, vmax_p = contrast_range
low = np.percentile(raw, vmin_p)
high = np.percentile(raw, vmax_p)
if high <= low:
    high = low + 1e-6  # avoid division by zero

# Clip and normalize to [0, 1] for display
img = np.clip(raw, low, high)
img = (img - low) / (high - low + 1e-8)

# ============================================================
# 🖼 Main slice view
# ============================================================
st.subheader(f"{plane} - Slice {cur_idx + 1} / {n_slices}")

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(img, cmap=cmap_name, vmin=0.0, vmax=1.0)
ax.axis("off")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Normalized Intensity", rotation=270, labelpad=15)

fig.tight_layout()
st.pyplot(fig)

# ============================================================
# 📊 Intensity histogram for the current slice
# ============================================================
st.subheader("Intensity Histogram")

vals = raw.flatten()
fig_hist, ax_hist = plt.subplots(figsize=(7, 3.2))

ax_hist.hist(
    vals,
    bins=80,
    alpha=0.9,
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.3,
)

# Restrict the x-axis to a robust range (1–99 percentile)
x1, x99 = np.percentile(vals, [1, 99])
ax_hist.set_xlim(x1, x99)

ax_hist.set_xlabel("Intensity", fontsize=11)
ax_hist.set_ylabel("Voxel Count", fontsize=11)
ax_hist.grid(True, linestyle="--", alpha=0.3)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

fig_hist.tight_layout()
st.pyplot(fig_hist)

# ============================================================
# 🔉 Simulated tSNR map (single-slice approximation)
# ============================================================
st.subheader("tSNR Map (Simulated)")

# Estimate noise from low-intensity background voxels.
bg_threshold = np.percentile(raw, 20.0)
noise_vals = raw[raw <= bg_threshold]
sigma_noise = np.std(noise_vals) if noise_vals.size > 10 else np.std(raw)
sigma_noise = max(sigma_noise, 1e-6)

# Simple SNR = signal / noise estimate
snr_map = raw / sigma_noise

# Clip SNR to a robust range for visualization
snr_low = np.percentile(snr_map, 5.0)
snr_high = np.percentile(snr_map, 95.0)
snr_vis = np.clip(snr_map, snr_low, snr_high)

fig_snr, ax_snr = plt.subplots(figsize=(6, 6))
im_snr = ax_snr.imshow(snr_vis, cmap="jet")
ax_snr.axis("off")

cbar_snr = fig_snr.colorbar(im_snr, ax=ax_snr, fraction=0.046, pad=0.04)
cbar_snr.set_label("SNR (a.u.)", rotation=270, labelpad=15)

fig_snr.tight_layout()
st.pyplot(fig_snr)

# ============================================================
# 📈 Simulated voxel time series (fMRI-style demo)
# ============================================================
st.subheader("Voxel Time Series (Simulated)")

# This is NOT computed from the DICOM data. It is a synthetic
# example that mimics an fMRI-like time course with:
# - a small linear drift,
# - a block-design activation,
# - and additive Gaussian noise.
n_timepoints = 200
time = np.arange(n_timepoints)

baseline = 1000
trend = np.linspace(-20, 20, n_timepoints)

block = np.zeros(n_timepoints)
block[100:180] += 80  # "activation" period

noise = np.random.randn(n_timepoints) * 20
voxel_ts = baseline + trend + block + noise

fig_ts, ax_ts = plt.subplots(figsize=(7, 3.2))
ax_ts.plot(time, voxel_ts, linewidth=1.8, color="#DD8452")

ax_ts.set_xlabel("Time (TR)", fontsize=11)
ax_ts.set_ylabel("Signal (a.u.)", fontsize=11)
ax_ts.grid(True, linestyle="--", alpha=0.3)
ax_ts.spines["top"].set_visible(False)
ax_ts.spines["right"].set_visible(False)

fig_ts.tight_layout()
st.pyplot(fig_ts)
