from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "size_price_linear_regression.pkl"
DATA_PATH = APP_DIR / "house_size_price_linear_dataset.csv"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
)

st.title("🏡 House Price Predictor")
st.caption("A beginner-friendly Linear Regression demo")
st.write(
    "Enter a house's area to estimate its price. This model uses only house size, "
    "so treat the result as an educational estimate—not a real property valuation."
)

@st.cache_resource
def load_or_train_model():
    # Prefer the model trained and saved in the Google Colab notebook.
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), "Loaded your saved Colab model", None

    # If the model file is not present, train from the same dataset as the notebook.
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        required = {"Size_in_SqFt", "Price_in_Lakhs"}
        if not required.issubset(df.columns):
            raise ValueError(
                "The CSV must contain columns named Size_in_SqFt and Price_in_Lakhs."
            )
        df = df.dropna(subset=["Size_in_SqFt", "Price_in_Lakhs"])
        X = df[["Size_in_SqFt"]]
        y = df["Price_in_Lakhs"]
        model = Pipeline([("linear_regression", LinearRegression())])
        model.fit(X, y)
        return model, "Trained from the supplied dataset", df

    raise FileNotFoundError(
        "No model or dataset found. Add size_price_linear_regression.pkl "
        "or house_size_price_linear_dataset.csv to this app folder."
    )

try:
    model, model_status, training_df = load_or_train_model()
except Exception as exc:
    st.error("The app could not load its model.")
    st.write(str(exc))
    st.markdown(
        """
        **How to fix this**
        1. In Google Colab, run the notebook through **Step 11 — Save the trained model**.
        2. In the left **Files** panel, open `saved_model`.
        3. Download `size_price_linear_regression.pkl`.
        4. Put that file in the same folder as `app.py`.
        """
    )
    st.stop()

st.success(model_status)

if training_df is not None:
    min_area = float(training_df["Size_in_SqFt"].min())
    max_area = float(training_df["Size_in_SqFt"].max())
else:
    # A practical input range for the dataset described in the notebook.
    min_area, max_area = 500.0, 5000.0

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("Enter house details")
    area = st.number_input(
        "House area (square feet)",
        min_value=1,
        max_value=100000,
        value=int(min(max(1500, min_area), max_area)),
        step=50,
        help="The model uses house area as its only input feature.",
    )
    predict_clicked = st.button("Predict house price", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame({"Size_in_SqFt": [float(area)]})
        predicted_price = float(model.predict(input_df)[0])
        st.metric("Estimated house price", f"₹ {predicted_price:,.2f} lakh")
        st.caption(f"Approximately ₹ {predicted_price * 100000:,.0f}")
        if area < min_area or area > max_area:
            st.warning(
                f"This area is outside the training data range "
                f"({min_area:,.0f}–{max_area:,.0f} sq ft). Predictions outside that range "
                "are extrapolations and may be less reliable."
            )
        st.info("Actual prices also depend on location, age, condition, amenities, and many other factors.")

with right:
    st.subheader("How the model predicts")
    if training_df is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(
            training_df["Size_in_SqFt"],
            training_df["Price_in_Lakhs"],
            alpha=0.65,
            label="Actual dataset prices",
        )
        x_line = np.linspace(
            float(training_df["Size_in_SqFt"].min()),
            float(training_df["Size_in_SqFt"].max()),
            100,
        )
        y_line = model.predict(pd.DataFrame({"Size_in_SqFt": x_line}))
        ax.plot(x_line, y_line, linewidth=2.5, label="Regression line")
        ax.set_xlabel("House area (sq ft)")
        ax.set_ylabel("Price (lakhs of rupees)")
        ax.set_title("House size vs. price")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write(
            "Add the original CSV next to `app.py` if you also want to display "
            "the dataset points alongside the regression line."
        )

st.divider()
st.subheader("About this project")
st.write(
    "Algorithm: Linear Regression · Input: house area in square feet · "
    "Output: estimated price in lakhs of rupees."
)
st.caption(
    "Educational demo based on the project notebook. It is not financial, legal, "
    "or property valuation advice."
)
