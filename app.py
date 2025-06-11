import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(layout="wide")
st.title("🤖 ML/DL Model Comparison Dashboard")

# --- Dataset Overview ---
st.sidebar.header("📁 Dataset Overview")
try:
    data = pd.read_csv("data/winequality-red.csv")
    st.sidebar.subheader("Wine Quality Dataset Info")
    st.sidebar.write("Shape:", data.shape)
    st.sidebar.write("Columns:", list(data.columns))
    st.sidebar.write("Target distribution:")
    st.sidebar.bar_chart(data["quality"].value_counts().sort_index())
    if st.sidebar.checkbox("Show raw data"):
        st.sidebar.dataframe(data.head())
except:
    st.sidebar.error("Upload 'winequality-red.csv' to /data first.")

# --- ML Evaluation ---
st.header("🔢 Machine Learning Models (Wine Quality Dataset)")
ml_models = {}
ml_reports = {}
ml_accuracies = {}
try:
    X = data.drop("quality", axis=1)
    y = data["quality"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_files = {
        "Logistic Regression": "models/logistic_regression_model.pkl",
        "Random Forest": "models/random_forest_model.pkl",
        "SVM": "models/svm_model.pkl"
    }

    for name, path in model_files.items():
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        ml_models[name] = model
        ml_accuracies[name] = np.round((y_pred == y_test).mean(), 2)
        ml_reports[name] = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        with st.expander(f"📉 {name} Confusion Matrix"):
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"{name} - Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    # Accuracy bar chart
    st.subheader("📊 Accuracy Comparison")
    st.bar_chart(ml_accuracies)

    # Classification reports
    for name, report in ml_reports.items():
        st.subheader(f"📋 {name} Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

except:
    st.warning("⚠️ ML models or data not found. Please run `2_train_models.py` and ensure data is present.")

# --- DL Evaluation ---
st.header("🖼️ Deep Learning Model (MNIST Dataset)")
try:
    X_test_mnist = np.load("data/X_test.npy")
    y_test_mnist = np.load("data/y_test.npy")
    cnn_model = tf.keras.models.load_model("models/cnn_mnist.h5")

    # Evaluate
    loss, cnn_acc = cnn_model.evaluate(X_test_mnist, y_test_mnist, verbose=0)
    st.metric("CNN Accuracy", f"{cnn_acc:.2%}")

    # Prediction report
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_mnist), axis=1)
    report = classification_report(y_test_mnist, y_pred_cnn, output_dict=True)
    cm = confusion_matrix(y_test_mnist, y_pred_cnn)

    with st.expander("📋 CNN Classification Report"):
        st.dataframe(pd.DataFrame(report).transpose())

    with st.expander("📉 CNN Confusion Matrix"):
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
        ax.set_title("CNN - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    if st.button("🎲 Show Random MNIST Prediction"):
        idx = np.random.randint(0, len(X_test_mnist))
        img = X_test_mnist[idx].reshape(28, 28)
        pred = y_pred_cnn[idx]
        true = y_test_mnist[idx]
        plt.figure(figsize=(3, 3))
        plt.imshow(img, cmap="gray")
        plt.title(f"Predicted: {pred}, Actual: {true}")
        plt.axis("off")
        st.pyplot(plt)

except:
    st.warning("⚠️ CNN model or MNIST data not found. Please run `2_train_models.py` to train it.")

# --- Comparison Summary Table ---
st.header("📊 Overall Model Comparison")
comparison_data = {
    "Model": ["Logistic Regression", "Random Forest", "SVM", "CNN"],
    "Accuracy": [
        ml_accuracies.get("Logistic Regression", 0),
        ml_accuracies.get("Random Forest", 0),
        ml_accuracies.get("SVM", 0),
        round(cnn_acc, 2) if 'cnn_acc' in locals() else 0
    ],
    "Best For": [
        "Linear relationships",
        "Complex patterns",
        "High-dimensional data",
        "Image data"
    ]
}
st.dataframe(pd.DataFrame(comparison_data))
