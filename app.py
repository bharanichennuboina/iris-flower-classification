import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Iris AI Classifier",
    page_icon="🌸",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
iris = load_iris()
X = iris.data
y = iris.target
species = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- TRAIN MODELS ----------------
knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

knn_accuracy = accuracy_score(y_test, knn.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))

# ---------------- APP DESIGN ----------------
st.title("🌸 Iris Flower Classification")
st.write(
    "An interactive machine learning application "
    "using KNN and Random Forest."
)

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("🔍 Classify a New Flower")
st.write("Enter the flower measurements in centimeters.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sepal_length = st.number_input(
        "Sepal length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

with col3:
    petal_length = st.number_input(
        "Petal length (cm)",
        min_value=0.1,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

with col4:
    petal_width = st.number_input(
        "Petal width (cm)",
        min_value=0.1,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

if st.button("🌼 Predict Flower Species", type="primary",
             use_container_width=True):

    knn_prediction = knn.predict(flower)[0]
    rf_prediction = rf.predict(flower)[0]

    knn_probabilities = knn.predict_proba(flower)[0]
    rf_probabilities = rf.predict_proba(flower)[0]

    st.subheader("🌺 Prediction Results")

    result1, result2 = st.columns(2)

    with result1:
        st.markdown("### K-Nearest Neighbors")
        st.success(f"Predicted species: **{species[knn_prediction].title()}**")
        st.write(
            f"Model probability for predicted class: "
            f"{knn_probabilities[knn_prediction] * 100:.2f}%"
        )

    with result2:
        st.markdown("### Random Forest")
        st.success(f"Predicted species: **{species[rf_prediction].title()}**")
        st.write(
            f"Model probability for predicted class: "
            f"{rf_probabilities[rf_prediction] * 100:.2f}%"
        )

    if knn_prediction == rf_prediction:
        st.info("Both models predict the same flower species.")
    else:
        st.warning("The models predict different species.")

st.divider()

# ---------------- ACCURACY COMPARISON ----------------
st.subheader("📊 Model Accuracy Comparison")

metric1, metric2 = st.columns(2)

with metric1:
    st.metric("KNN Accuracy", f"{knn_accuracy * 100:.2f}%")

with metric2:
    st.metric("Random Forest Accuracy", f"{rf_accuracy * 100:.2f}%")

fig, ax = plt.subplots(figsize=(7, 4))

bars = ax.bar(
    ["KNN", "Random Forest"],
    [knn_accuracy * 100, rf_accuracy * 100]
)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Test Set Accuracy")
ax.set_ylim(0, 105)
ax.bar_label(bars, fmt="%.2f%%")
ax.grid(axis="y", alpha=0.25)

st.pyplot(fig)
plt.close(fig)

# ---------------- OPTIONAL DETAILS ----------------
with st.expander("View model evaluation details"):
    st.write("### KNN Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            confusion_matrix(y_test, knn.predict(X_test)),
            index=species,
            columns=species
        )
    )

    st.write("### Random Forest Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            confusion_matrix(y_test, rf.predict(X_test)),
            index=species,
            columns=species
        )
    )

    st.write("### Classification Reports")

    report1, report2 = st.columns(2)

    with report1:
        st.write("**KNN**")
        st.text(
            classification_report(
                y_test,
                knn.predict(X_test),
                target_names=species,
                zero_division=0
            )
        )

    with report2:
        st.write("**Random Forest**")
        st.text(
            classification_report(
                y_test,
                rf.predict(X_test),
                target_names=species,
                zero_division=0
            )
        )

st.caption(
    "Educational machine learning project. "
    "Predictions depend on the entered measurements and trained models."
)