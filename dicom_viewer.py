import streamlit as st
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="DICOM Viewer",
    page_icon="🩻"   # or your png icon path
)
st.title("🩻 DICOM Viewer 3D")

# -----------------------------
# Your DICOM folder
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dicom_dir = os.path.join(BASE_DIR, "sample_data", "MRI_dataset")

if not os.path.exists(dicom_dir):
    st.error(f"❌ DICOM folder does not exist: {dicom_dir}")
    st.stop()

# -----------------------------
# Helper function: sort files by InstanceNumber
# -----------------------------
def get_sorted_dicom_files(dicom_dir):
    items = []
    for f in os.listdir(dicom_dir):
        if not f.endswith(".dcm"):
            continue
        path = os.path.join(dicom_dir, f)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            inst = getattr(ds, "InstanceNumber", 0)
        except Exception:
            inst = 0
        items.append((inst, f))
    items.sort(key=lambda x: x[0])
    return [f for _, f in items]

# -----------------------------
# Load the whole volume (stack into 3D)
# -----------------------------
@st.cache_resource
def load_dicom_volume(dicom_dir: str):
    files = get_sorted_dicom_files(dicom_dir)
    if len(files) == 0:
        raise RuntimeError("No .dcm files found.")

    slices = []
    for f in files:
        path = os.path.join(dicom_dir, f)
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slices.append(arr)

    volume = np.stack(slices, axis=0)  # (Z, Y, X)
    return volume, files

try:
    volume, dicom_files = load_dicom_volume(dicom_dir)
except Exception as e:
    st.error("❌ Error loading DICOM volume:")
    st.error(str(e))
    st.stop()

# Volume shape
n_z, n_y, n_x = volume.shape

# -----------------------------
# Slice orientation selection
# -----------------------------
st.sidebar.header("View & Slice Navigation")

plane_label_list = [
    "Axial",
    "Coronal",
    "Sagittal",
]
plane = st.sidebar.radio("Panel Orientation", plane_label_list, index=0)

plane_key_map = {
    "Axial": "axial",
    "Coronal": "coronal",
    "Sagittal": "sagittal",
}
plane_key = plane_key_map[plane]

plane_n_slices = {
    "axial": n_z,
    "coronal": n_y,
    "sagittal": n_x,
}

if "slice_idx" not in st.session_state:
    st.session_state.slice_idx = {
        "axial": n_z // 2,
        "coronal": n_y // 2,
        "sagittal": n_x // 2,
    }

n_slices = plane_n_slices[plane_key]
cur_idx = st.session_state.slice_idx[plane_key]

cur_idx = max(0, min(n_slices - 1, cur_idx))
st.session_state.slice_idx[plane_key] = cur_idx

col1, col2 = st.sidebar.columns(2)
if col1.button("⬆ Last"):
    st.session_state.slice_idx[plane_key] = max(0, cur_idx - 1)
if col2.button("⬇ Next"):
    st.session_state.slice_idx[plane_key] = min(n_slices - 1, cur_idx + 1)

cur_idx = st.session_state.slice_idx[plane_key]
st.sidebar.write(f"Current: **{cur_idx + 1} / {n_slices}**")

contrast_range = st.sidebar.slider(
    "Window Length (Percentile)",
    0.0, 100.0,
    (5.0, 95.0),
    step=1.0,
)

cmap_name = st.sidebar.selectbox(
    "Colormap",
    ["gray", "bone", "hot", "jet", "viridis"],
    index=0
)

# -----------------------------
# Extract slice
# -----------------------------
if plane_key == "axial":
    raw_slice = volume[cur_idx, :, :]
elif plane_key == "coronal":
    raw_slice = volume[:, cur_idx, :]
else:
    raw_slice = volume[:, :, cur_idx]

raw = raw_slice.astype(np.float32)

# Percentile clipping
vmin_p, vmax_p = contrast_range
low = np.percentile(raw, vmin_p)
high = np.percentile(raw, vmax_p)
if high <= low:
    high = low + 1e-6

img = np.clip(raw, low, high)
img = (img - low) / (high - low + 1e-8)

# -----------------------------
# Display image
# -----------------------------
st.subheader(f"{plane} - Slice {cur_idx + 1} / {n_slices}")

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(img, cmap=cmap_name, vmin=0.0, vmax=1.0)
ax.axis("off")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Normalized Intensity", rotation=270, labelpad=15)
fig.tight_layout()
st.pyplot(fig)

# =============================================================
#   🔄 SWITCH ORDER: Histogram FIRST → SNR Map SECOND
# =============================================================

# -----------------------------
# Intensity histogram (FIRST)
# -----------------------------
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

x1, x99 = np.percentile(vals, [1, 99])
ax_hist.set_xlim(x1, x99)
ax_hist.set_xlabel("Intensity", fontsize=11)
ax_hist.set_ylabel("Voxel Count", fontsize=11)
ax_hist.grid(True, linestyle="--", alpha=0.3)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

fig_hist.tight_layout()
st.pyplot(fig_hist)

# -----------------------------
# SNR Map (SECOND)
# -----------------------------
st.subheader("tSNR Map (Simulated)")

bg_threshold = np.percentile(raw, 20.0)
noise_vals = raw[raw <= bg_threshold]
sigma_noise = np.std(noise_vals) if noise_vals.size > 10 else np.std(raw)
sigma_noise = max(sigma_noise, 1e-6)

snr_map = raw / sigma_noise

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

# -----------------------------
# Simulated time series
# -----------------------------
st.subheader("Voxel Time Series (Simulated)")

n_timepoints = 200
time = np.arange(n_timepoints)

baseline = 1000
trend = np.linspace(-20, 20, n_timepoints)
block = np.zeros(n_timepoints)
block[100:180] += 80
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
