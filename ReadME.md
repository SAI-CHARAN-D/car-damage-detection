# 🚗 AI-Powered Insurance Fraud Detection System

An end-to-end intelligent system that automates vehicle insurance claim verification using computer vision and machine learning. The system detects physical damage from uploaded vehicle images using YOLOv11, validates claim consistency through rule-based and ML-driven feature engineering, and predicts fraud probability using an XGBoost model. It integrates Explainable AI (SHAP, LIME, Grad-CAM) to provide transparent decision-making and is deployed via FastAPI with an interactive Streamlit dashboard.

---

## 🔥 What Makes This Project Strong

This is NOT just image classification or fraud detection in isolation.

This is a **Vision + Tabular ML + Reasoning system** — a full end-to-end pipeline where every component feeds the next:

- YOLOv11 detects physical damage from car photos
- OCR validates the license plate against a vehicle database
- Claim parser reads what the person claims was damaged
- Match scorer compares YOLO detections vs claimed damages
- XGBoost predicts fraud probability from all combined features
- SHAP + LIME explain WHY the model made its decision

---

## 🧠 System Architecture
User uploads image + types claim text
↓
Streamlit Frontend
↓
POST /claim → FastAPI
↓
┌─────────────────────────────────────┐
│  yolo_service   → damages detected  │
│  ocr_service    → plate_valid       │
│  claim_parser   → damages claimed   │
│  match_service  → match_score       │
└─────────────────────────────────────┘
↓
fraud_service → XGBoost → fraud_probability
↓
xai_service → SHAP + LIME → reasons + charts
↓
decision_service → APPROVE / REVIEW / REJECT
↓
JSON response back to Streamlit
↓
┌──────────────────────────────────────┐
│  Annotated image with YOLO boxes     │
│  Decision badge (colour coded)       │
│  Fraud probability score             │
│  SHAP waterfall chart                │
│  Top 3 plain English reasons         │
└──────────────────────────────────────┘

---

## 📂 Project Structure
insurance-fraud-ai/
│
├── data/
│   ├── processed/              # YOLO formatted images and labels
│   ├── fraud_dataset.csv       # 2000-row synthetic fraud dataset
│   ├── vehicle_db.json         # Mock vehicle plate database
│   └── data.yaml               # YOLO training config
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis
│   ├── 02_model_eval.ipynb     # Model evaluation and metrics
│   └── 03_xai_analysis.ipynb   # SHAP, LIME, Grad-CAM analysis
│
├── training/
│   ├── train_yolo.py           # YOLOv11 fine-tuning script
│   ├── train_fraud_model.py    # XGBoost training script
│   └── generate_synthetic_data.py  # Fraud dataset generator
│
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── routes.py               # API endpoints
│   ├── services/
│   │   ├── yolo_service.py     # Damage detection
│   │   ├── ocr_service.py      # Plate extraction and validation
│   │   ├── claim_parser.py     # Claim text parsing
│   │   ├── match_service.py    # Detection vs claim matching
│   │   ├── fraud_service.py    # XGBoost fraud prediction
│   │   ├── decision_service.py # Threshold decision logic
│   │   └── xai_service.py      # SHAP and LIME explanations
│   └── utils/
│       ├── image_utils.py      # Image processing helpers
│       └── config.py           # Paths and constants
│
├── frontend/
│   └── app.py                  # Streamlit UI
│
├── models/
│   ├── yolo_damage.pt          # Fine-tuned YOLOv11 weights
│   └── fraud_xgb.pkl           # Trained XGBoost model
│
├── outputs/                    # Per-request generated files
├── tests/
├── requirements.txt
└── README.md

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/insurance-fraud-ai.git
cd insurance-fraud-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your models

Place your trained models in the `models/` folder:
models/
├── yolo_damage.pt
└── fraud_xgb.pkl

### 4. Run the backend

```bash
uvicorn app.main:app --reload
```

### 5. Run the frontend

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 How To Use

1. Upload a vehicle photo
2. Describe the damage in the claim text box
3. Enter the claim amount and vehicle age
4. Click **Analyse Claim**
5. The system returns a decision with full explanation

---

## 📊 Models

### YOLOv11 — Damage Detection

Fine-tuned on a car damage dataset with 4 classes:

| Class | Description |
|---|---|
| `missing_part` | Missing vehicle component |
| `crack` | Tear or crack in body |
| `dent` | Dented surface |
| `scratch` | Surface scratch |

### XGBoost — Fraud Classification

Trained on a 2000-row synthetic dataset with the following features:

| Feature | Description |
|---|---|
| `plate_valid` | License plate found in vehicle database |
| `match_score` | Overlap between YOLO detections and claim text |
| `num_detected` | Number of damages detected by YOLO |
| `num_claimed` | Number of damages stated in claim text |
| `claim_amount` | Monetary value of the claim |
| `vehicle_age` | Age of the vehicle in years |


---

## 🔍 Decision Logic

| Fraud Probability | Decision | Meaning |
|---|---|---|
| < 0.40 | ✅ APPROVE | Claim appears legitimate |
| 0.40 – 0.70 | ⚠️ REVIEW | Moderate suspicion, needs manual check |
| > 0.70 | ❌ REJECT | High fraud probability detected |

---

## 🧠 Explainable AI

This system uses three XAI techniques to explain every decision:

- **SHAP** — Waterfall chart showing each feature's contribution to the fraud score
- **LIME** — Top 3 human-readable reasons for the prediction
- **Grad-CAM** — Heatmap overlay on the car image showing which region triggered YOLO detection

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Damage Detection | YOLOv11 (Ultralytics) |
| OCR | EasyOCR |
| Fraud Classification | XGBoost |
| Explainability | SHAP, LIME |
| Backend | FastAPI |
| Frontend | Streamlit |
| Image Processing | OpenCV |

---