# 🌊 FluidFlow

> **Early warning system for crowd crush prevention using physics-informed neural networks and computer vision.**

FluidFlow detects when a crowd is about to become dangerous — before anyone can see it happening. It treats dense crowds as physical fluids and computes real-time turbulence, pressure, and Reynolds number metrics from existing CCTV footage, issuing predictive alerts **30–120 seconds before a crowd crush becomes visible**.

---

## 📸 Demo

### Normal Flow — Laminar State
![Normal Flow Demo](docs/images/demo_normal.png)
> *Green gauge · TI < 0.15 · Re < 1500 · Crowd density below safe threshold*

### Transitional State
![Transitional Demo](docs/images/demo_transitional.png)
> *Yellow gauge · Density rising · LSTM forecaster detecting upward trend*

### Stampede Precursor — Alert Fired
![Alert Demo](docs/images/demo_alert.png)
> *Red gauge · TI > 0.30 · Pressure heatmap showing force accumulation zones · Alert logged 45 seconds before visible crush*

---

<!-- 
  HOW TO ADD YOUR DEMO IMAGES:
  1. Create a folder: docs/images/ in the project root
  2. Take screenshots of each scenario while the app is running
  3. Save them as:
       docs/images/demo_normal.png
       docs/images/demo_transitional.png
       docs/images/demo_alert.png
  4. Optionally add a screen recording as a GIF:
       docs/images/demo_full.gif
     and add this line below the table:
       ![Full Demo](docs/images/demo_full.gif)
  Tools for GIF recording:
    - Windows: ShareX (free) or ScreenToGif
    - CachyOS/Linux: Peek or ffmpeg screen capture
-->

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10.x
- Git
- 8GB RAM minimum
- Internet connection (for dependency install and YOLOv8 weight download)

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fluidflow.git
cd fluidflow
```

### Set Up Virtual Environment

**CachyOS / Linux:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install Dependencies
```bash
# Step 1 — PyTorch CPU (must be installed first)
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2 — All other dependencies
pip install -r requirements.txt

# Step 3 — Verify no conflicts
pip check
```

### Add Demo Videos

Place three video clips in `data/videos/`:
```
data/
└── videos/
    ├── normal_flow.mp4           # Steady crowd, density < 2 p/m²
    ├── transitional.mp4          # Crowd thickening, density 2–4 p/m²
    └── stampede_precursor.mp4    # High density, visible compression waves
```

> Videos should be 60–90 seconds, 720p, under 50MB each.  
> Compress using ffmpeg if needed: `ffmpeg -i input.mp4 -crf 28 output.mp4`

### Generate Training Data and Train LSTM
```bash
python scripts/generate_synthetic.py
# Expected: Saved 1200 scenarios. Shape: (1200, 60, 4)

python scripts/train_lstm.py
# Takes ~5–8 minutes on CPU
# Expected: Done. Best val: 0.001x. Saved to models/lstm_forecaster.pt
```

### Precompute Metrics (Required Before Demo)
```bash
python scripts/precompute.py data/videos/normal_flow.mp4
python scripts/precompute.py data/videos/transitional.mp4
python scripts/precompute.py data/videos/stampede_precursor.mp4
```

### Run the App
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## ⚙️ How It Works

| Step | What happens |
|------|-------------|
| **Watch** | Ingests existing CCTV footage — no new hardware required |
| **See** | YOLOv8-nano maps persons onto a 32×32 density grid (persons/m²) |
| **Feel** | Farneback Optical Flow computes a dense (vx, vy) velocity field per pixel |
| **Understand** | NumPy physics engine derives Turbulence Intensity, Reynolds Number, Pressure Proxy, and Shockwave Flag |
| **Predict** | PyTorch LSTM reads a 30-frame window and forecasts TI 30 seconds ahead |
| **Alert** | Level 2 alert fires when predicted TI > 0.30 or Re > 3000 — before the crush is visible |

---

## 📊 Alert Levels

| Level | Status | Condition |
|-------|--------|-----------|
| 🟢 0 | LAMINAR — Safe | TI < 0.15, Re < 1500, density < 2.0 p/m² |
| 🟡 1 | TRANSITIONAL — Monitor | TI < 0.30, Re < 3000 |
| 🔴 2 | TURBULENT — EVACUATE | TI ≥ 0.30 or Re ≥ 3000 or density > 4.0 p/m² |

---

## 🧪 Tech Stack

| Component | Tool | 
|-----------|------|
| UI | Streamlit | 
| Visualization | Plotly | 
| Detection | Ultralytics YOLOv8 | 
| Video / CV | OpenCV headless | 
| Numerical | NumPy | 
| Deep Learning | PyTorch CPU | 
| Database | SQLite3 | 

---

### Localhost
streamlit run app.py

## 📄 Research Basis

FluidFlow is grounded in peer-reviewed crowd dynamics research:

> Helbing, D., Johansson, A., & Al-Abideen, H. (2007). *Dynamics of crowd disasters: An empirical study.* Physical Review E, 75(4).  
> [Full text — arXiv:physics/0701203](https://arxiv.org/abs/physics/0701203)

---

## 📜 License

This project is developed for academic and research purposes.

---

*"From seeing crowds to understanding them."*
