import streamlit as st
import pandas as pd
import re
from collections import defaultdict

st.set_page_config(page_title="Dog Reassignment", layout="wide")
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

st.sidebar.header("🔗 Load Data")
url_map = st.sidebar.text_input("📋 Map Sheet URL")
url_matrix = st.sidebar.text_input("📏 Distance Matrix Sheet URL")

if url_map and url_matrix:
    matrix_df = load_csv(url_matrix)
    map_df = load_csv(url_map)

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

    driver_capacities = {}
    driver_callouts = {}
    def parse_cap(val):
        try:
            return 9 if val in ["", "X", "NAN"] else int(val)
        except:
            return 9

    for _, row in map_df.iterrows():
        driver_val = row.get("Driver", "")
        driver = str(driver_val).strip() if pd.notna(driver_val) else ""
        if not driver or driver in driver_capacities:
            continue
        g1 = parse_cap(row.get("Group 1", ""))
        g2 = parse_cap(row.get("Group 2", ""))
        g3 = parse_cap(row.get("Group 3", ""))
        driver_capacities[driver] = {'group1': g1, 'group2': g2, 'group3': g3}
        driver_callouts[driver] = {'group1': False, 'group2': False, 'group3': False}

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

    def find_best_eviction_target(candidate_driver, group_num, dog_to_fit):
        distances = distance_matrix.get(dog_to_fit['dog_id'], {})
        viable_dogs = []
        for other_dog_id, other_dog in dogs_going_today.items():
            if other_dog['driver'] != candidate_driver or other_dog_id == dog_to_fit['dog_id']:
                continue
            if group_num not in other_dog['groups']:
                continue
            if other_dog_id not in distances:
                continue
            dist = distances[other_dog_id]
            if dist == 0 or dist > 0.75:
                continue
            viable_dogs.append((dist, other_dog_id))
        if not viable_dogs:
            return None
        return sorted(viable_dogs, key=lambda x: x[0])[0][1]

    st.subheader("📋 Driver Callout")
    selected_driver = st.selectbox("Select driver calling out:", sorted(driver_capacities.keys()))
    selected_groups = st.multiselect("Select group(s) they're missing:", [1, 2, 3])

    if st.button("Reassign Dogs"):
        for g in selected_groups:
            driver_callouts[selected_driver][f'group{g}'] = True

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

        dogs_to_reassign = []
        for dog_id, info in dogs_going_today.items():
            if info['driver'] != selected_driver:
                continue
            affected = [g for g in info['groups'] if driver_callouts[selected_driver].get(f'group{g}', False)]
            if affected:
                dogs_to_reassign.append({
                    'dog_id': dog_id,
                    'original_driver': selected_driver,
                    'original_groups': info['groups'],
                    'affected_groups': affected,
                    'dog_info': info
                })

        assignments = []
        evicted_global_set = set()
        queue = dogs_to_reassign.copy()
        depth = 0

        while queue and depth < MAX_DOMINO_DEPTH:
            new_queue = []
            for dog in queue:
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
                        if candidate_driver == selected_driver:
                            continue
                        if any(driver_callouts[candidate_driver].get(f'group{g}', False) for g in dog_groups):
                            continue
                        other_groups = dogs_going_today[other_id]['groups']
                        matched = [g for g in dog_groups if g in other_groups or any(is_adjacent(g, og) for og in other_groups)]
                        if set(matched) != set(dog_groups):
                            continue

                        fits = True
                        evicted = []
                        for g in dog_groups:
                            key = f'group{g}'
                            if driver_loads[candidate_driver][key] + num_dogs > driver_capacities[candidate_driver][key]:
                                evicted_dog_id = find_best_eviction_target(candidate_driver, g, dog)
                                if evicted_dog_id:
                                    driver_loads[candidate_driver][key] -= dogs_going_today[evicted_dog_id]['num_dogs']
                                    evicted.append(evicted_dog_id)
                                    evicted_global_set.add(evicted_dog_id)
                                    new_queue.append({
                                        'dog_id': evicted_dog_id,
                                        'original_driver': candidate_driver,
                                        'original_groups': dogs_going_today[evicted_dog_id]['groups'],
                                        'affected_groups': dogs_going_today[evicted_dog_id]['groups'],
                                        'dog_info': dogs_going_today[evicted_dog_id]
                                    })
                                else:
                                    fits = False
                                    break

                        if fits:
                            for g in dog_groups:
                                driver_loads[candidate_driver][f'group{g}'] += num_dogs
                            register_new_assignment(dog_id, candidate_driver, dog_groups, dog['dog_info']['dog_name'], dog['dog_info']['address'], num_dogs)
                            assignments.append({
                                'Dog ID': dog_id,
                                'Dog Name': dog['dog_info']['dog_name'],
                                'New Assignment': f"{candidate_driver}:{'&'.join(map(str, dog_groups))}",
                                'Distance': round(dist, 3),
                                'Evicted': ", ".join(evicted) if evicted else ""
                            })
                            match_found = True
                            break
                    if match_found:
                        break
            queue = new_queue
            depth += 1

        st.success(f"✅ Reassigned {len(assignments)} dogs from {selected_driver}")
        st.dataframe(pd.DataFrame(assignments))

