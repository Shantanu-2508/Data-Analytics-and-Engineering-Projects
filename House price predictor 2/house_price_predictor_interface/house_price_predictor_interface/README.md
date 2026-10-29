# House Price Predictor — Streamlit Interface

This interface is designed for the `House_Size_Price_Linear_Regression` Google Colab notebook.
It uses the same feature and target column names:
- `Size_in_SqFt`
- `Price_in_Lakhs`

## 1. Get the trained model from Google Colab

1. Open your notebook in Google Colab.
2. Run the notebook through **Step 11 — Save the trained model** (or run all cells).
3. In the left sidebar, click the **Files** icon.
4. Open the `saved_model` folder.
5. Right-click `size_price_linear_regression.pkl` and download it.
6. Put `size_price_linear_regression.pkl` in the same folder as `app.py`.

Optional: download `house_size_price_linear_dataset.csv` from Colab and put it beside `app.py` too. This lets the app draw the actual data points and regression line.

## 2. Run the app on your computer

Install Python if needed, then open a terminal/Command Prompt in this folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will show a local URL (usually `http://localhost:8501`). Open it in your browser.

## 3. Run it in Google Colab (alternative)

Upload `app.py`, `requirements.txt`, and your `.pkl` model into the Colab session. Then run:

```python
!pip install -q streamlit
```

Colab does not expose a Streamlit page directly by default; for a first project, running locally is the simplest option. If you need a public link, deploy the project to Streamlit Community Cloud from a GitHub repository.

## Files

- `app.py` — the interface
- `requirements.txt` — Python packages
- `README.md` — setup guide

## Important notes

- The app first loads the trained `.pkl` model from the same folder.
- If the model file is missing but the CSV exists, it trains a model from that CSV automatically.
- The estimate is expressed in **lakhs of rupees**, matching the notebook target column.
- Predictions outside the training data's size range are less reliable.
- This model uses area alone; it does not account for location, age, condition, or amenities.
