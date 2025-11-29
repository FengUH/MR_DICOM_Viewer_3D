# 🩻 MR DICOM Viewer 3D

An interactive and lightweight **3D DICOM MRI Viewer** built with Streamlit.  
This app provides a clean interface for browsing MRI volumes, adjusting contrast,
viewing intensity histograms, visualizing simulated SNR, and exploring fMRI-style
voxel time series.

---

## 🚀 Live Demo
https://fenguh-mr-dicom-viewer-3d-dicom-viewer-p49tae.streamlit.app

---

## ✨ Key Features

- **Multi-plane slice viewer**  
  Navigate MRI data in Axial, Coronal, and Sagittal views with simple Previous/Next buttons.

- **Contrast adjustments**  
  Percentile-based intensity windowing for quick tuning and enhanced visibility.

- **Colormap selection**  
  Choose from `gray`, `bone`, `hot`, `jet`, `viridis`.

- **Slice intensity histogram**  
  Visualize voxel intensity distribution for the current slice.

- **Simulated tSNR map**  
  Demonstrates signal-to-noise estimation using a simple noise model.

- **Simulated fMRI time series**  
  Includes drift, block activation, and noise for teaching demonstrations.

- **Streamlit-version-compatible caching**  
  Automatically switches between `st.cache_data` (new) and `st.cache` (older Streamlit Cloud), ensuring full compatibility.

---
