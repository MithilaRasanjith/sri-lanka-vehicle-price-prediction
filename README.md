# Predicting used-car asking prices in Sri Lanka

This project compares bagging (Random Forest), boosting (XGBoost), and stacking (a ridge regression combining their predictions) on advertisements from [Prasad Nirmal’s Sri Lankan vehicle price dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset). The dashboard estimates an asking price from brand, model, manufacture year, engine capacity, transmission, fuel type and mileage.

## Run the dashboard

The saved models are included. From this directory on Windows, use Python 3.13 and run the following in PowerShell.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal. Use **Estimate a price** to try a listing and **Compare models** to inspect test errors and Random Forest feature importance.

## Reproduce the experiment

Open `vehicle_prices.ipynb` in Google Colab and run the cells from top to bottom with an internet connection. The notebook downloads the dataset, removes exact duplicates after dropping the saved row number, trains and evaluates all three models, and exports `vehicle_models.joblib`. The dashboard loads this bundle from the same directory as `app.py`.

## Results

All three models trained on 7,747 earlier listings and were evaluated on 2,023 listings dated February 2, 2025 or later. Random Forest had a mean absolute error of **5.16**, followed by stacking at **5.46** and XGBoost at **6.02**. The stack improved on XGBoost, while Random Forest alone had the smallest average error on this test period. Year of manufacture had the largest permutation importance for Random Forest. Estimates and errors use the original dataset units because the conversion to rupees is unverified. The target is the advertised asking price.

## AI usage acknowledgment

ChatGPT assisted with dataset selection, notebook code for preprocessing and evaluation, training and comparing the models, interpreting the results, building and testing the Streamlit dashboard, and preparing the README and video outline.
