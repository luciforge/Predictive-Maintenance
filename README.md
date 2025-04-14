# Predictive Maintenance Chatbot 🚀

This project demonstrates a full pipeline for predictive maintenance using machine learning models and a conversational AI interface. It uses C-MAPSS FD001 sensor data to predict Remaining Useful Life (RUL) and failure classifications with SHAP-based explainability.

## 🔧 Key Features
- Preprocessing pipeline with sensor renaming, noise injection, and drift simulation
- RUL regression using RandomForest, XGBoost, and TCN
- Health status classification (Green/Yellow/Red)
- SHAP explainability for classification predictions
- Flask API to serve predictions and explanations
- Gradio chatbot frontend with natural language query support

## 📁 Project Structure
```
pred_maintenance/
├── app/
│   ├── app.py                # Flask API server
│   └── gradio_app.py         # Chatbot interface
├── scripts/
│   ├── preprocess_dataset.py
│   ├── train_fd001.py
│   ├── fd001_classification.py
│   ├── fd001_explain.py
│   ├── fd001_tcn.py
├── CMAPSSData/               # Raw and transformed dataset storage
├── saved_models/             # Trained models and scalers
├── outputs/                  # Plots and SHAP visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧪 Example Prompts (Chatbot)
- “What is the status of unit 3?”
- “Status of unit 5 at cycle 50”
- “How much life is remaining for unit 8?”

## 🛠️ Setup & Usage

### 1. Preprocess Dataset
```bash
python scripts/preprocess_dataset.py
```

### 2. Train Models
```bash
python scripts/train_fd001.py
python scripts/fd001_classification.py
python scripts/fd001_tcn.py
```

### 3. Run Backend API
```bash
python app/app.py
```

### 4. Launch Chatbot UI
```bash
python app/gradio_app.py
```

## 📊 Evaluation Modes
Use `scripts/test.py` to evaluate classifier:
- `"tail"` mode: last N cycles of each unit
- `"stratified"` mode: fixed number per class

## 📈 Visualization Output
SHAP summary and waterfall plots are saved in the `outputs/` folder.

---

Made with ❤️ to showcase predictive maintenance.
