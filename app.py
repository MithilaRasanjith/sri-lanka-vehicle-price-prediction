from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Sri Lankan car prices", page_icon="🚗")


@st.cache_resource
def load_dashboard(path, modified_ns):
    return joblib.load(path)


bundle = Path(__file__).with_name("vehicle_models.joblib")
data = load_dashboard(bundle, bundle.stat().st_mtime_ns)
st.title("Sri Lankan car asking prices")
estimate, comparison = st.tabs(["Estimate a price", "Compare models"])

with estimate:
    left, right = st.columns(2)
    brand = left.selectbox("Brand", sorted(data["catalog"]))
    model = right.selectbox("Car model", sorted(data["catalog"][brand]))
    sample = data["catalog"][brand][model]
    left, right = st.columns(2)
    year = left.number_input("Year of manufacture", value=int(sample["YOM"]), step=1,
                             min_value=data["years"][0], max_value=data["years"][1])
    mileage = right.number_input("Mileage (km)", min_value=0, value=int(sample["Millage(KM)"]), step=1000)
    engine = left.number_input("Engine capacity (cc)", min_value=0, value=int(sample["Engine (cc)"]), step=50)
    gears = data["options"]["Gear"]
    fuels = data["options"]["Fuel Type"]
    gear = right.selectbox("Transmission", gears, index=gears.index(sample["Gear"]))
    fuel = left.selectbox("Fuel type", fuels, index=fuels.index(sample["Fuel Type"]))
    car = pd.DataFrame([{"Brand": brand, "Model": model, "YOM": year, "Engine (cc)": engine,
                         "Gear": gear, "Fuel Type": fuel, "Millage(KM)": mileage}])
    st.subheader("Estimated asking price")
    for column, (name, predictor) in zip(st.columns(len(data["models"])), data["models"].items()):
        column.metric(data["paradigms"][name], f"{predictor.predict(car)[0]:,.2f}")
        column.caption(name)
    st.caption("Values use the dataset’s original price units. The rupee conversion is unverified.")

with comparison:
    st.subheader("Error on later listings")
    st.write(f"Trained on {data['train_count']:,} listings and tested on {data['test_count']:,} "
             f"listings dated {data['cutoff']} or later. Lower mean absolute error is better.")
    scores = pd.Series(data["scores"], name="Mean absolute error").round(2).to_frame()
    scores.insert(0, "Paradigm", pd.Series(data["paradigms"]))
    st.table(scores)
    st.subheader("Which details mattered to Random Forest?")
    st.write("Each bar shows how much the test error increased when one input was shuffled. "
             "Larger increases mean the model relied more on that input for these listings.")
    st.bar_chart(pd.Series(data["importance"], name="Increase in mean absolute error"),
                 horizontal=True, sort=False)
