import streamlit as st
import pandas as pd
import re
from collections import defaultdict

st.set_page_config(page_title="Dog Reassignment System", layout="wide")
st.title("🐶 Dog Reassignment System")

MAX_DOMINO_DEPTH = 5
EVICTION_DISTANCE_LIMIT = 0.5

@st.cache_data
def load_csv(url):
    return pd.read_csv(url, dtype=str)

def generate_radius_steps(start=0.25, max_radius=2.2, step=0.25):
    current = start
    while current <= max_radius:
        yield round(current, 3)
        current += step

def is_adjacent(g1, g2):
    return abs(g1 - g2) == 1

def weighted_distance(dist, from_group, to_groups):
    if any(g in to_groups for g in from_group):
        return dist
    elif any(is_adjacent(g1, g2) for g1 in from_group for g2 in to_groups):
        return dist * 2
    return float('inf')

# Load hardcoded sheets
url_map = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
url_matrix = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"

map_df = load_csv(url_map)
matrix_df = load_csv(url_matrix)

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

def parse_cap(val):
    try:
        return 9 if val in ["", "X", "NAN"] else int(val)
    except:
        return 9

driver_capacities = {}
driver_callouts = {}
for _, row in map_df.iterrows():
    drv = str(row.get("Driver", "")).strip()
    if not drv or drv in driver_capacities:
        continue
    g1 = parse_cap(row.get("Group 1", ""))
    g2 = parse_cap(row.get("Group 2", ""))
    g3 = parse_cap(row.get("Group 3", ""))
    driver_capacities[drv] = {'group1': g1, 'group2': g2, 'group3': g3}
    driver_callouts[drv] = {
        'group1': str(row.get("Group 1", "")).strip().upper() == "X",
        'group2': str(row.get("Group 2", "")).strip().upper() == "X",
        'group3': str(row.get("Group 3", "")).strip().upper() == "X"
    }

st.subheader("📋 Driver Callout")
selected_driver = st.selectbox("Driver to call out", sorted(set(info['driver'] for info in dogs_going_today.values())))
selected_groups = st.multiselect("Groups affected", [1, 2, 3], default=[1, 2, 3])

if st.button("Reassign Dogs"):
    for g in selected_groups:
        driver_callouts[selected_driver][f"group{g}"] = True

    dogs_to_reassign = []
    for dog_id, info in dogs_going_today.items():
        if info["driver"] != selected_driver:
            continue
        affected = [g for g in info["groups"] if driver_callouts[selected_driver].get(f"group{g}", False)]
        if affected:
            dogs_to_reassign.append({
                'dog_id': dog_id,
                'original_driver': selected_driver,
                'original_groups': info['groups'],
                'affected_groups': affected,
                'dog_info': info
            })

    # Build distance matrix
    distance_matrix = {}
    dog_ids = matrix_df.columns[1:]
    for i, row in matrix_df.iterrows():
        row_id = str(row.iloc[0]).strip()
        distance_matrix[row_id] = {}
        for j, col_id in enumerate(dog_ids):
            try:
                val = float(row.iloc[j + 1])
            except:
                val = 0.0
            distance_matrix[row_id][str(col_id).strip()] = val

    driver_loads = defaultdict(lambda: {'group1': 0, 'group2': 0, 'group3': 0})
    for dog_id, info in dogs_going_today.items():
        for g in info['groups']:
            driver_loads[info['driver']][f'group{g}'] += info['num_dogs']

    final_rows = []
    seen = set()

    for dog in dogs_to_reassign:
        dog_id = dog['dog_id']
        dog_groups = dog['original_groups']
        num_dogs = dog['dog_info']['num_dogs']
        distances = distance_matrix.get(dog_id, {})
        best_driver = None
        best_dist = float('inf')
        for other_id, dist in distances.items():
            if dist == 0 or other_id not in dogs_going_today:
                continue
            candidate_driver = dogs_going_today[other_id]['driver']
            if candidate_driver == selected_driver or candidate_driver not in driver_capacities:
                continue
            other_groups = dogs_going_today[other_id]['groups']
            weighted = weighted_distance(dist, dog_groups, other_groups)
            if weighted == float('inf') or weighted > best_dist:
                continue
            fits = all(driver_loads[candidate_driver][f"group{g}"] + num_dogs <= driver_capacities[candidate_driver][f"group{g}"] for g in dog_groups)
            if fits:
                best_driver = candidate_driver
                best_dist = weighted

        if best_driver:
            for g in dog_groups:
                driver_loads[best_driver][f"group{g}"] += num_dogs
            dogs_going_today[dog_id]['driver'] = best_driver
            dogs_going_today[dog_id]['assignment'] = f"{best_driver}:{'&'.join(map(str, dog_groups))}"
            seen.add(dog_id)
            final_rows.append({
                "Dog ID": dog_id,
                "New Assignment": f"{best_driver}:{'&'.join(map(str, dog_groups))}",
                "Distance": round(best_dist, 3)
            })

    result_df = pd.DataFrame(final_rows)[["Dog ID", "New Assignment", "Distance"]]
    st.success("✅ Final Reassignments")
    st.dataframe(result_df)
