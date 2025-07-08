import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Iris_Flower import train_model

st.set_page_config(page_title="🌸 Iris Predictor", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        h1, h2, h3 {
            color: #ffd700;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

model, df, X = train_model()

st.title("🌺 Iris Flower Species Predictor")

with st.form(key='input_form'):
    st.subheader("Enter Iris Flower Measurements")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, step=0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, step=0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, step=0.1)
    submit_button = st.form_submit_button(label='🔍 Compute Prediction')

if submit_button:
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)

    st.subheader("🌼 Predicted Species:")
    st.success(prediction[0].capitalize())

    # Show Feature Importance
    st.subheader("📊 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))

    # Append input data to df and label with prediction
    input_df = pd.DataFrame(input_data, columns=X.columns)
    input_df['species'] = prediction[0]
    combined_df = pd.concat([df, input_df], ignore_index=True)

    # Pairplot
    st.subheader("🔬 Pairplot of Dataset with Your Input")
    fig1 = sns.pairplot(combined_df, hue='species')
    st.pyplot(fig1)

    # Correlation Heatmap
    st.subheader("🧮 Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(combined_df.drop('species', axis=1).corr(), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig2)
