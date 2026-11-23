import pandas as pd
import torch
import streamlit as st
import trainer

st.set_page_config(page_title="Hospital Screening Dashboard", layout="wide")
st.title("Screening model: training dashboard")
st.caption("Change the settings in the sidebar, press Train, and watch training vs validation loss live.")

# Initialize session state
if "runs" not in st.session_state:
    st.session_state.runs = []
if "latest_model" not in st.session_state:
    st.session_state.latest_model = None
if "latest_data" not in st.session_state:
    st.session_state.latest_data = trainer.prepare_data(seed=0, scale=True)
if "latest_summary" not in st.session_state:
    st.session_state.latest_summary = None
if "final_test_done" not in st.session_state:
    st.session_state.final_test_done = None

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    hidden = st.select_slider("Neurons per layer", options=[16, 32, 64, 128, 256], value=256)
    dropout = st.slider("Dropout", min_value=0.0, max_value=0.6, value=0.0, step=0.05)
    weight_decay = st.select_slider("Weight decay", options=[0.0, 0.001, 0.01, 0.05, 0.1], value=0.0)
    lr = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    epochs = st.slider("Max epochs", min_value=50, max_value=500, value=300, step=25)
    patience = st.slider("Early stopping patience (0 = off)", min_value=0, max_value=50, value=0, step=5)
    seed = st.number_input("Random seed", min_value=0, max_value=999, value=0, step=1)
    scale = st.checkbox("Scale the features", value=True)
    train_clicked = st.button("Train", type="primary", use_container_width=True)

COLORS = ["#d97706", "#2563eb"]

# Live chart & progress bar placeholders
chart_slot = st.empty()
progress_bar = st.progress(0.0)

if train_clicked:
    cfg = {
        "hidden": hidden,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "seed": int(seed),
        "scaled": scale,
    }
    data = trainer.prepare_data(seed=cfg["seed"], scale=cfg["scaled"])

    def redraw(epoch, hist):
        progress_bar.progress(min(epoch / cfg["epochs"], 1.0))
        if epoch % 5 == 0 or epoch == 1 or epoch == cfg["epochs"]:
            df_curve = pd.DataFrame(
                {
                    "training loss": hist["train"],
                    "validation loss": hist["val"],
                }
            )
            chart_slot.line_chart(df_curve, color=COLORS)

    model, hist, summary = trainer.train(cfg, data, on_epoch=redraw)
    progress_bar.progress(1.0)

    # Final chart render
    chart_slot.line_chart(
        pd.DataFrame({"training loss": hist["train"], "validation loss": hist["val"]}),
        color=COLORS,
    )

    st.session_state.latest_model = model
    st.session_state.latest_data = data
    st.session_state.latest_summary = summary
    st.session_state.final_test_done = None  # Reset one-time test for newly trained model

    vm = summary["val_metrics"]
    st.session_state.runs.append(
        {
            **cfg,
            "epochs_run": summary["epochs_run"],
            "best_val_loss": summary["best_val_loss"],
            "diagnosis": summary["diagnosis"],
            "val_cancers_found": f"{vm['cancers_found']}/{vm['total_cancers']}",
            "val_recall": vm["recall"],
            "val_false_alarms": vm["false_alarms"],
        }
    )

# Display latest diagnosis & metrics
if st.session_state.latest_summary:
    s = st.session_state.latest_summary
    vm = s["val_metrics"]
    verdict = s["diagnosis"]

    if verdict == "healthy":
        st.success("Diagnosis: healthy — training and validation curves stay close together.")
    elif verdict == "overfitting":
        st.warning("Diagnosis: overfitting — validation loss went back up by >30% above its best.")
    else:
        st.error("Diagnosis: underfitting — training loss remained above 0.30.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epochs run", s["epochs_run"])
    c2.metric("Best validation loss", f"{s['best_val_loss']:.3f}", f"epoch {s['best_epoch']}")
    c3.metric("Cancers found (validation)", f"{vm['cancers_found']} of {vm['total_cancers']}")
    c4.metric("False alarms (validation)", vm["false_alarms"])

st.divider()

# Experiments table + CSV download
st.subheader("Experiments so far")
if st.session_state.runs:
    runs_df = pd.DataFrame(st.session_state.runs)
    st.dataframe(runs_df, use_container_width=True)
    csv_bytes = runs_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download as CSV",
        data=csv_bytes,
        file_name="hospital_screening_experiments.csv",
        mime="text/csv",
    )
else:
    st.info("No runs recorded yet. Adjust the sidebar settings and click Train.")

st.divider()

# Check a single unseen test patient & One-time final test
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Check a patient (held-out test set)")
    data = st.session_state.latest_data
    num_test = len(data["y_te"])
    patient_idx = st.slider("Select test patient index", 0, num_test - 1, 0)

    if st.session_state.latest_model is not None:
        model = st.session_state.latest_model
        model.eval()
        x_patient = data["X_te"][patient_idx : patient_idx + 1]
        true_label = int(data["y_te"][patient_idx].item())

        with torch.no_grad():
            probs = torch.softmax(model(x_patient), dim=1)[0]
            pred_label = int(probs.argmax().item())
            malignant_prob = float(probs[1].item())

        st.write(f"**Model Prediction:** {'Malignant (1)' if pred_label == 1 else 'Benign (0)'} ({malignant_prob:.1%} risk)")
        st.write(f"**True Diagnosis:** {'Malignant (1)' if true_label == 1 else 'Benign (0)'}")
    else:
        st.caption("Train a model first to screen an individual patient.")

with col_right:
    st.subheader("One-time Final Test")
    st.caption("Run only after tuning hyperparameters on validation curves.")
    if st.session_state.latest_model is not None:
        if st.button("Run Final Test Set Evaluation", disabled=(st.session_state.final_test_done is not None)):
            res = trainer.evaluate_split(
                st.session_state.latest_model,
                st.session_state.latest_data["X_te"],
                st.session_state.latest_data["y_te"],
            )
            st.session_state.final_test_done = res

        if st.session_state.final_test_done:
            ft = st.session_state.final_test_done
            st.info(
                f"**Final Test Result:** Accuracy `{ft['accuracy']}` · "
                f"**{ft['cancers_found']} of {ft['total_cancers']}** cancers found · "
                f"**{ft['false_alarms']}** false alarms"
            )
    else:
        st.caption("Train your final candidate model first.")