import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_final_filled.csv")

df = load_data()

# Define the correct target column name
target_column = "Output"

# Check for target column presence
if target_column not in df.columns:
    st.error(f"Target column '{target_column}' not found in the dataset.")
else:
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    @st.cache_resource
    def train_model():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    model = train_model()

    # Custom background and style
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
            padding: 20px;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .alert {
            padding: 20px;
            color: white;
            margin-top: 20px;
            border-radius: 5px;
            text-align: center;
        }
        .green {background-color: #4CAF50;}
        .red {background-color: #f44336;}
        </style>
    """, unsafe_allow_html=True)

    st.title("💓 Heart Disease Prediction App")
    st.write("Provide the following medical information:")

    user_input = {}
    for col in X.columns:
        if df[col].dtype in ['float64', 'int64']:
            user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        else:
            user_input[col] = st.selectbox(f"{col}", df[col].unique())

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader("Prediction Confidence")
        fig1, ax1 = plt.subplots()
        ax1.bar(["No Heart Disease", "Heart Disease"], proba, color=["green", "red"])
        ax1.set_ylabel("Probability")
        ax1.set_ylim(0, 1)
        st.pyplot(fig1)

        if prediction == 1:
            st.markdown(
                f'<div class="alert red">🔴 Alert: The model predicts <b>Heart Disease</b> with confidence {proba[1]:.2f}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="alert green">🟢 Good News: The model predicts <b>No Heart Disease</b> with confidence {proba[0]:.2f}</div>',
                unsafe_allow_html=True)

    st.subheader("🔍 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
