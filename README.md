# Streamlit Pushover

This Streamlit web app performs **pushover analysis post-processing** using user-input structural data.  
Users can enter **story weights**, **mode shapes**, and **capacity curves** directly—**no need to upload CSV files**.  
The app automatically computes the **capacity spectrum** and identifies the **performance point** based on user-defined **demand spectrum parameters**.

---

## 🚀 Features

- Input:
  - Story weights
  - Mode shapes
  - Capacity curve (base shear vs. roof displacement)
- Built-in calculation of:
  - Modal participation factor
  - Effective modal mass and height
  - Capacity spectrum (Sd vs. Sa)
- Custom demand spectrum input (user-defined parameters)
- Auto-calculation of performance point (capacity–demand intersection)
- Interactive charts using Matplotlib
- Clean web interface, no CSV upload required

---

📦 Requirements
Python 3.8+

streamlit

numpy

pandas

matplotlib

All dependencies are listed in requirements.txt.