import streamlit as st
import pandas as pd
import re
from collections import defaultdict

st.set_page_config(page_title="Dog Reassignment System", layout="wide")
st.title("🐶 Dog Reassignment System")

MAX_DOMINO_DEPTH = 5

@st.cache_data
def load_csv(url):
    if "docs.google.com" in url and "edit" in url:
        sheet_id = url.split("/d/")[1].split("/")[0]
        gid = url.split("gid=")[-1]
        return pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}", dtype=str)
    return pd.read_csv(url, dtype=str)

def generate_radius_steps(start=0.25, max_radius=2.2, step=0.25):
    current = start
    while current <= max_radius:
        yield round(current, 3)
        current += step

# UI to load sheets
st.sidebar.header("🔗 Google Sheets")
url_map = st.sidebar.text_input("📋 Map Sheet URL (with Driver & Group data)")
url_matrix = st.sidebar.text_input("📏 Distance Matrix Sheet URL")

if url_map and url_matrix:
    matrix_df = load_csv(url_matrix)
    map_df = load_csv(url_map)

    map_df["Dog ID"] = map_df["Dog ID"].astype(str).str.strip()
    map_df["Name"] = map_df["Name"].astype(str).str.strip()
    map_df["Group"] = map_df["Group"].astype(str).str.replace("LM", "", case=False).str.strip()

    # Build dogs_going_today
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

    # Parse driver callouts and capacities
    def parse_cap(val):
        try:
            return 9 if val in ["", "X", "NAN"] else int(val)
        except:
            return 9

    driver_capacities = {}
    driver_callouts = {}
    for _, row in map_df.iterrows():
        driver_val = row.get("Driver", "")
        driver = str(driver_val).strip() if pd.notna(driver_val) else ""
        if not driver or driver in driver_capacities:
            continue
        g1 = parse_cap(row.get("Group 1", ""))
        g2 = parse_cap(row.get("Group 2", ""))
        g3 = parse_cap(row.get("Group 3", ""))
        driver_capacities[driver] = {'group1': g1, 'group2': g2, 'group3': g3}
        driver_callouts[driver] = {
            'group1': str(row.get("Group 1", "")).strip().upper() == "X",
            'group2': str(row.get("Group 2", "")).strip().upper() == "X",
            'group3': str(row.get("Group 3", "")).strip().upper() == "X"
        }

    def is_adjacent(g1, g2):
        return (g1 == 1 and g2 == 2) or (g1 == 2 and g2 in [1, 3]) or (g1 == 3 and g2 == 2)

    def register_new_assignment(dog_id, to_driver, groups, dog_name="", address="", num_dogs=1):
        dogs_going_today[dog_id] = {
            'assignment': f"{to_driver}:{'&'.join(map(str, groups))}",
            'num_dogs': num_dogs,
            'driver': to_driver,
            'groups': groups,
            'address': address,
            'dog_name': dog_name
        }

    st.subheader("📋 Choose Driver to Call Out")
    called_out_driver = st.selectbox("Driver to reassign:", sorted(set(d['driver'] for d in dogs_going_today.values())))
    if st.button("Reassign Dogs"):
        # Flag driver called out
        for g in ['group1', 'group2', 'group3']:
            driver_callouts[called_out_driver][g] = True

        # Prepare dogs for reassignment
        dogs_to_reassign = []
        for dog_id, info in dogs_going_today.items():
            if info['driver'] != called_out_driver:
                continue
            affected = [g for g in info['groups'] if driver_callouts[called_out_driver].get(f'group{g}', False)]
            if affected:
                dogs_to_reassign.append({
                    'dog_id': dog_id,
                    'original_driver': info['driver'],
                    'original_groups': info['groups'],
                    'affected_groups': affected,
                    'dog_info': info
                })

        # Load distance matrix into lookup
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
            driver = info['driver']
            for g in info['groups']:
                driver_loads[driver][f'group{g}'] += info['num_dogs']

        # Greedy reassignment loop
        assignments = []
        for dog in dogs_to_reassign:
            dog_id = dog['dog_id']
            dog_groups = dog['original_groups']
            num_dogs = dog['dog_info']['num_dogs']
            distances = distance_matrix.get(dog_id, {})
            match_found = False
            for radius in generate_radius_steps():
                for other_id, dist in distances.items():
                    if dist == 0 or dist > radius or other_id not in dogs_going_today:
                        continue
                    candidate_driver = dogs_going_today[other_id]['driver']
                    if candidate_driver == called_out_driver:
                        continue
                    if any(driver_callouts.get(candidate_driver, {}).get(f'group{g}', False) for g in dog_groups):
                        continue
                    other_groups = dogs_going_today[other_id]['groups']
                    matched = [g for g in dog_groups if g in other_groups or any(is_adjacent(g, og) for og in other_groups)]
                    if set(matched) != set(dog_groups):
                        continue
                    if all(driver_loads[candidate_driver][f'group{g}'] + num_dogs <= driver_capacities[candidate_driver][f'group{g}'] for g in dog_groups):
                        for g in dog_groups:
                            driver_loads[candidate_driver][f'group{g}'] += num_dogs
                        register_new_assignment(dog_id, candidate_driver, dog_groups,
                                                dog_name=dog['dog_info']['dog_name'],
                                                address=dog['dog_info']['address'],
                                                num_dogs=num_dogs)
                        assignments.append({
                            'Dog ID': dog_id,
                            'Dog Name': dog['dog_info']['dog_name'],
                            'New Assignment': f"{candidate_driver}:{'&'.join(map(str, dog_groups))}",
                            'Distance': round(dist, 3)
                        })
                        match_found = True
                        break
                if match_found:
                    break

        st.success(f"✅ Reassigned {len(assignments)} dogs from {called_out_driver}")
        st.dataframe(pd.DataFrame(assignments))
