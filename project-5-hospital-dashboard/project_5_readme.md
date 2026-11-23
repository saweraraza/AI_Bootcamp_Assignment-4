# Project 5: City Hospital — Live Training Dashboard

An interactive **Streamlit** and **PyTorch** web application built for City Hospital's data team to monitor training vs. validation curves in real time, diagnose overfitting and underfitting, tune regularization, and evaluate cancer screening sensitivity before clinical review.

---

## Table of Contents
1. [Problem Statement & Objectives](#problem-statement--objectives)
2. [Implementation Architecture](#implementation-architecture)
3. [VS Code Setup & Execution](#vs-code-setup--execution)
4. [Experiment Log & Results](#experiment-log--results)
5. [Recommended Settings & Final Test](#recommended-settings--final-test)
6. [Clinical Readiness & Limitations](#clinical-readiness--limitations)

---

## Problem Statement & Objectives
City Hospital's initial cancer screening model reached a training loss of `0.000`, leading physicians to reject it due to suspected memorization.

**Key Objectives:**
1. Load the Wisconsin Breast Cancer dataset ($569$ patients, $30$ diagnostic measurements; `Malignant = 1`, `Benign = 0`) using a stratified **60% train / 20% validation / 20% test** split.
2. Prevent data leakage by fitting `StandardScaler` strictly on training rows (with a toggle to test unscaled feature behavior).
3. Stream live training and validation loss curves every 5 epochs in Streamlit alongside a progress bar.
4. Automatically classify each training run as **underfitting**, **overfitting**, or **healthy**, and report **Cancers Found (Recall)** and **False Alarms**.
5. Log all experiments in a downloadable CSV table, support single-patient test screening, and enforce a strict one-time final test evaluation.

---

## Implementation Architecture
* **`trainer.py` (PyTorch Training & Diagnostic Engine):**
  * `prepare_data()`: Stratified 60/20/20 split and training-only `StandardScaler` transformation.
  * `ScreeningMLP`: Configurable two-hidden-layer ReLU MLP with `nn.Dropout` and `AdamW` weight decay.
  * `train()`: Runs epoch-by-epoch optimization, saves the best validation checkpoint in memory, triggers live UI callbacks (`on_epoch`), and supports early stopping (`patience`).
  * `diagnose()`: Flags **underfitting** if final training loss $> 0.30$, **overfitting** if final validation loss exceeds $1.30 \times \text{best validation loss}$, and **healthy** otherwise.
* **`app.py` (Streamlit Interactive Dashboard):**
  * **Sidebar Controls:** Neurons per layer (`16–256`), Dropout (`0.0–0.6`), Weight Decay (`0.0–0.1`), Learning Rate, Max Epochs, Early Stopping Patience, Random Seed, and Feature Scaling toggle.
  * **Live Observability:** Real-time dual-line chart (`st.empty()`) and progress bar (`st.progress()`).
  * **Experiment Tracking:** Persistent `st.session_state.runs` dataframe with one-click CSV export (`hospital_screening_experiments.csv`).
  * **Clinical Inspection:** Interactive patient slider on held-out test records and a guarded one-time Final Test button.

---

## VS Code Setup & Execution

### 1. Navigate to the Project Folder
```bash
cd project-5-hospital-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## Experiment Log & Results

Systematic one-variable-at-a-time experiments conducted in the dashboard (exported to `hospital_screening_experiments.csv`):

| Run | Configuration | Curve Diagnosis | Validation Cancers Found | Validation False Alarms |
| :--- | :--- | :---: | :---: | :---: |
| **1. Baseline** | `256` neurons, `dropout=0.0`, `decay=0.0`, `patience=0`, `scaled=True` | `overfitting` | `41 of 43` | `3` |
| **2. + Dropout & Decay** | `256` neurons, `dropout=0.3`, `decay=0.05`, `patience=0`, `scaled=True` | `overfitting` | `40 of 43` | `2` |
| **3. + Early Stopping** | `256` neurons, `dropout=0.3`, `decay=0.05`, `patience=20`, `scaled=True` | `healthy` | `40 of 43` | `3` |
| **4. Right-Sized Model** | `16` neurons, `dropout=0.3`, `decay=0.05`, `patience=20`, `scaled=True` | `healthy` | `40 of 43` | **`1`** |
| **5. Unscaled Trap** | `16` neurons, `dropout=0.3`, `decay=0.05`, `patience=20`, `scaled=False` | `jumps around` | `33 of 43` | `2` |

---

## Recommended Settings & Final Test
* **Optimal Hyperparameters:**
  * **Neurons per layer:** `16`
  * **Dropout:** `0.3`
  * **Weight decay:** `0.05`
  * **Learning rate:** `0.001`
  * **Early stopping patience:** `20`
  * **Feature scaling:** `Enabled (True)`
* **One-Time Final Test Result (Held-Out Test Set):**
  * **Accuracy:** `95.6%` (`0.956`)
  * **Cancers Found (Recall):** `41 of 42` (`97.6%`)
  * **False Alarms:** `4`

---

## Clinical Readiness & Limitations

**Is this model ready for autonomous hospital deployment? No.**

1. **Single-Cohort Dataset Limitation:** Trained on only 569 patient samples from a single dataset; performance may degrade across different hospitals, biopsy procedures, or patient demographics.
2. **False Negative Clinical Risk:** The model still missed **1 malignant tumor** (`41 of 42`) on the unseen test set—missing an active cancer carries severe patient safety consequences.
3. **Lack of Explainability & Clinical Validation:** Produces black-box probability scores without feature-level explanations (e.g., SHAP) or prospective clinical trial approval. It must only be used as a **screening aid reviewed by a qualified physician**.