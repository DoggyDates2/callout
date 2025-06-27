
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import re

st.set_page_config(page_title="Dog Reassignment System", layout="wide")

st.title("🐕 Dog Reassignment System")

# Sidebar inputs
st.sidebar.header("🔗 Google Sheet CSV Links")
distance_url = st.sidebar.text_input("Distance Matrix CSV URL", "")
map_url = st.sidebar.text_input("Map CSV URL", "")
driver_url = st.sidebar.text_input("Driver Counts CSV URL", "")

if st.sidebar.button("Run Reassignment"):

    def load_csv(url):
        if "docs.google.com" in url and "edit" in url:
            sheet_id = url.split("/d/")[1].split("/")[0]
            gid = url.split("gid=")[-1]
            return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return url

    # Load and process files
    try:
        matrix_df = pd.read_csv(load_csv(distance_url), dtype=str)
        map_df = pd.read_csv(load_csv(map_url), dtype=str)
        driver_counts_df = pd.read_csv(load_csv(driver_url), dtype=str)

        # Parse distance matrix
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

        # Parse dog assignments
        dogs_going_today = {}
        for _, row in map_df.iterrows():
            dog_id = str(row.get("Dog ID", "")).strip()
            assignment = str(row.get("Today", "")).strip()
            try:
                num_dogs = int(float(row.get("Number of dogs", "1")))
            except:
                num_dogs = 1
            if dog_id and assignment and ":" in assignment and "XX" not in assignment:
                dogs_going_today[dog_id] = {
                    'assignment': assignment,
                    'num_dogs': num_dogs,
                    'address': row.get("Address", ""),
                    'dog_name': row.get("Dog Name", "")
                }

        # Parse driver capacities
        driver_capacities = {}
        driver_callouts = {}
        for _, row in driver_counts_df.iterrows():
            driver = str(row.get("Driver", "")).strip()
            if not driver:
                continue
            def parse_group(val): return val.strip().upper() if isinstance(val, str) else ""
            g1, g2, g3 = parse_group(row.get("Group 1", "")), parse_group(row.get("Group 2", "")), parse_group(row.get("Group 3", ""))
            driver_callouts[driver] = {
                'group1': g1 == "X",
                'group2': g2 == "X",
                'group3': g3 == "X"
            }
            def cap(v): return 9 if v in ["", "X"] else int(v)
            driver_capacities[driver] = {
                'group1': cap(g1),
                'group2': cap(g2),
                'group3': cap(g3)
            }

        def parse_group_assignment(assign):
            if ':' not in assign or 'XX' in assign:
                return None, []
            d, g = assign.split(':', 1)
            groups = sorted(set(int(x) for x in re.findall(r'[123]', g)))
            return d.strip(), groups

        def is_adjacent(g1, g2):
            return (g1 == 1 and g2 == 2) or (g1 == 2 and g2 in [1, 3]) or (g1 == 3 and g2 == 2)

        # Find dogs to reassign
        dogs_to_reassign = []
        for dog_id, info in dogs_going_today.items():
            driver, groups = parse_group_assignment(info['assignment'])
            if not driver or not groups or driver not in driver_callouts:
                continue
            callout = driver_callouts[driver]
            affected = [g for g in groups if (g == 1 and callout['group1']) or (g == 2 and callout['group2']) or (g == 3 and callout['group3'])]
            if affected:
                dogs_to_reassign.append({
                    'dog_id': dog_id,
                    'original_driver': driver,
                    'original_groups': groups,
                    'affected_groups': affected,
                    'dog_info': info
                })

        # Current load per driver
        driver_loads = {}
        for dog_id, info in dogs_going_today.items():
            driver, groups = parse_group_assignment(info['assignment'])
            if not driver or not groups:
                continue
            if driver not in driver_loads:
                driver_loads[driver] = {'group1': 0, 'group2': 0, 'group3': 0}
            for g in groups:
                driver_loads[driver][f'group{g}'] += info['num_dogs']

        reassignments = []
        unassigned = []

        for dog in dogs_to_reassign:
            dog_id = dog['dog_id']
            dog_name = dog['dog_info']['dog_name']
            dog_groups = dog['original_groups']
            num_dogs = dog['dog_info']['num_dogs']
            distances = distance_matrix.get(dog_id, {})
            candidates = []
            for other_id, dist in distances.items():
                if dist == 0 or other_id not in dogs_going_today:
                    continue
                other_driver, other_groups = parse_group_assignment(dogs_going_today[other_id]['assignment'])
                if not other_driver or not other_groups or other_driver == dog['original_driver']:
                    continue
                if any(driver_callouts.get(other_driver, {}).get(f'group{g}', False) for g in dog_groups):
                    continue
                matched = [g for g in dog_groups if g in other_groups or any(is_adjacent(g, og) for og in other_groups)]
                if set(matched) == set(dog_groups):
                    can_fit = True
                    for g in dog_groups:
                        k = f'group{g}'
                        if driver_loads.get(other_driver, {}).get(k, 0) + num_dogs > driver_capacities.get(other_driver, {}).get(k, 9):
                            can_fit = False
                            break
                    if can_fit:
                        candidates.append((other_driver, dist))
            if candidates:
                best_driver, best_dist = sorted(candidates, key=lambda x: x[1])[0]
                for g in dog_groups:
                    driver_loads[best_driver][f'group{g}'] += num_dogs
                reassignments.append({
                    'Dog ID': dog_id,
                    'Dog Name': dog_name,
                    'From Driver': dog['original_driver'],
                    'To Driver': best_driver,
                    'Groups': "&".join(map(str, dog_groups)),
                    'Distance': round(best_dist, 3)
                })
            else:
                unassigned.append({
                    'Dog ID': dog_id,
                    'Dog Name': dog_name,
                    'From Driver': dog['original_driver'],
                    'Groups': "&".join(map(str, dog_groups)),
                    'Reason': "No match within capacity or distance"
                })

        st.subheader("✅ Successful Reassignments")
        st.dataframe(pd.DataFrame(reassignments))

        st.subheader("⚠️ Unassigned Dogs")
        st.dataframe(pd.DataFrame(unassigned))

    except Exception as e:
        st.error(f"Error: {e}")
