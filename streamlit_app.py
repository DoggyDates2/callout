
import streamlit as st
import pandas as pd
import re
from collections import defaultdict

st.set_page_config(page_title="Dog Reassignment", layout="wide")
st.title("🐕 Dog Reassignment System")

MAX_DOMINO_DEPTH = 5

# Dynamic proximity tier expansion generator
def generate_radius_steps(start=0.25, max_radius=2.2, step=0.25):
    current = start
    while current <= max_radius:
        yield round(current, 3)
        current += step

# Upload or link sheets
st.sidebar.header("CSV Links")
url_map = st.sidebar.text_input("Google Sheet CSV for MAP")
url_matrix = st.sidebar.text_input("Google Sheet CSV for DISTANCE MATRIX")

def load_csv(url):
    if "docs.google.com" in url and "edit" in url:
        sheet_id = url.split("/d/")[1].split("/")[0]
        gid = url.split("gid=")[-1]
        return pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}", dtype=str)
    return pd.read_csv(url, dtype=str)

if url_map and url_matrix:
    matrix_df = load_csv(url_matrix)
    map_df = load_csv(url_map)

    # Clean
    map_df["Dog ID"] = map_df["Dog ID"].astype(str).str.strip()
    map_df["Name"] = map_df["Name"].astype(str).str.strip()
    map_df["Group"] = map_df["Group"].astype(str).str.replace("LM", "", case=False).str.strip()

    dogs_going_today = {}
    for _, row in map_df.iterrows():
        dog_id = row.get("Dog ID", "").strip()
        driver = row.get("Name", "").strip()
        group_str = row.get("Group", "").strip()
        try:
            num_dogs = int(float(row.get("Number of dogs", "1")))
        except:
            num_dogs = 1
        if dog_id and driver and group_str:
            groups = sorted(set(int(g) for g in re.findall(r'\d', group_str)))
            dogs_going_today[dog_id] = {
                'assignment': f"{driver}:{group_str}",
                'num_dogs': num_dogs,
                'driver': driver,
                'groups': groups,
                'address': row.get("Address", ""),
                'dog_name': row.get("Dog Name", "")
            }

    st.success("Files loaded. Sample of Dogs Going Today:")
    st.dataframe(pd.DataFrame(dogs_going_today).T)
