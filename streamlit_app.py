import streamlit as st
import pandas as pd
import re
from collections import defaultdict

st.set_page_config(page_title="🐶 Dog Assignment Tool", layout="wide")
st.title("🐶 Dog Assignment Tool")

# Configuration
MAX_REASSIGNMENT_DISTANCE = 5.0
url_map = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
url_matrix = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"

# Smart Caching Strategy
@st.cache_data(ttl=604800)  # 7 DAYS = 604800 seconds (distance matrix changes weekly)
def load_distance_matrix_csv(url):
    """Distance matrix - cached for 1 WEEK since it rarely changes"""
    return pd.read_csv(url, dtype=str)

@st.cache_data(ttl=3600)  # 1 HOUR = 3600 seconds (assignments change daily)
def load_assignments_csv(url):
    """Assignments data - cached for 1 HOUR since it changes daily"""
    return pd.read_csv(url, dtype=str)

@st.cache_data(ttl=604800)  # 7 DAYS (distance matrix processing - very heavy)
def build_distance_matrix(matrix_df):
    """Build distance matrix - CACHED FOR 1 WEEK (heaviest operation)"""
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
    
    return distance_matrix

@st.cache_data(ttl=3600)  # 1 HOUR
def process_dogs_data(map_df):
    """Process dogs data - cached for 1 HOUR"""
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
            all_groups = get_all_groups(group_str)  # Get ALL groups including 0
            delivery_groups = get_delivery_groups(group_str)  # Get only delivery groups (1,2,3)
            
            dogs_going_today[dog_id] = {
                'assignment': f"{driver}:{group_str}

@st.cache_data(ttl=60)  # 1 MINUTE (callouts change constantly)
def process_driver_data(map_df):
    """Process driver data - cached for 1 MINUTE (callouts change frequently)"""
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
    
    return driver_capacities, driver_callouts

def get_reassignment_priority(dog_data):
    """Calculate priority for dog reassignment. Lower number = higher priority."""
    num_dogs = dog_data['dog_info']['num_dogs']
    num_groups = len(dog_data['original_groups'])
    
    if num_dogs == 1 and num_groups == 1:
        return 1  # Highest priority - easiest to place
    elif num_dogs == 1 and num_groups > 1:
        return 2  # Single dog but multiple groups
    elif num_dogs > 1 and num_groups == 1:
        return 3  # Multiple dogs but single group
    else:
        return 4  # Lowest priority - hardest to place

def is_adjacent(g1, g2):
    return abs(g1 - g2) == 1

def parse_cap(val):
    try:
        return 9 if val in ["", "X", "NAN"] else int(val)
    except:
        return 9

def get_groups(group_str):
    """Extract valid group numbers (1, 2, 3) from group string. Ignore group 0."""
    group_str = group_str.replace("LM", "")
    all_groups = sorted(set(int(g) for g in re.findall(r'\d', group_str)))
    # Filter out group 0 since it represents start/end location, not a delivery group
    valid_groups = [g for g in all_groups if g in [1, 2, 3]]
    return valid_groups

def get_all_groups(group_str):
    """Get ALL group numbers including 0 - for proximity calculations"""
    group_str = group_str.replace("LM", "")
    return sorted(set(int(g) for g in re.findall(r'\d', group_str)))

def get_delivery_groups(group_str):
    """Get only delivery groups (1, 2, 3) - for capacity and reassignment"""
    group_str = group_str.replace("LM", "")
    all_groups = sorted(set(int(g) for g in re.findall(r'\d', group_str)))
    return [g for g in all_groups if g in [1, 2, 3]]

# Manual refresh controls
st.subheader("🔄 Force Data Refresh")
st.info("💡 Use these buttons to get the latest data immediately (bypasses cache)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🆕 **REFRESH ALL DATA**", type="primary", help="Get latest assignments & callouts now"):
        # Clear all non-matrix caches
        load_assignments_csv.clear()
        process_dogs_data.clear() 
        process_driver_data.clear()
        st.success("🔄 All data refreshed! Reloading latest from Google Sheets...")
        st.rerun()

with col2:
    if st.button("📋 Refresh Assignments", help="Update dog assignments only"):
        load_assignments_csv.clear()
        process_dogs_data.clear()
        st.success("📋 Assignments refreshed!")
        st.rerun()

with col3:
    if st.button("🚨 Refresh Callouts", help="Update driver callouts only"):
        process_driver_data.clear()
        st.success("🚨 Callouts refreshed!")
        st.rerun()

with col4:
    if st.button("🗺️ Refresh Matrix", help="Rebuild distance matrix (weekly)"):
        st.cache_data.clear()
        st.success("🗺️ Full rebuild triggered!")
        st.rerun()

# Load and process data with smart caching
import time
total_start_time = time.time()

# Load raw data with different cache strategies
matrix_df = load_distance_matrix_csv(url_matrix)
map_df = load_assignments_csv(url_map)

# Clean data
map_df["Dog ID"] = map_df["Dog ID"].astype(str).str.strip()
map_df["Name"] = map_df["Name"].astype(str).str.strip()
map_df["Group"] = map_df["Group"].astype(str).str.strip()

# Process with smart caching
dogs_going_today = process_dogs_data(map_df)
driver_capacities, driver_callouts = process_driver_data(map_df)

# Build distance matrix (heavily cached)
distance_matrix = build_distance_matrix(matrix_df)

# Current status summary
st.subheader("📊 Current Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    delivery_dogs = sum(1 for info in dogs_going_today.values() if not info.get('is_waypoint', False))
    waypoint_dogs = sum(1 for info in dogs_going_today.values() if info.get('is_waypoint', False))
    st.metric("Total Dogs", len(dogs_going_today), delta=f"{delivery_dogs} delivery + {waypoint_dogs} waypoints")
with col2:
    st.metric("Active Drivers", len(set(info['driver'] for info in dogs_going_today.values())))
with col3:
    callout_count = sum(1 for callouts in driver_callouts.values() if any(callouts.values()))
    st.metric("Drivers Called Out", callout_count)
with col4:
    total_capacity = sum(sum(cap.values()) for cap in driver_capacities.values())
    st.metric("Total Driver Capacity", total_capacity)

# Show info about Group 0 handling
st.info("ℹ️ **Group 0 Policy:** Waypoints (Group 0) stay with original driver when calling out, but help find nearby reassignments"

# Show current callouts
st.subheader("🚨 Current Driver Callouts")
callout_found = False
for driver, callouts in driver_callouts.items():
    called_out_groups = []
    if callouts['group1']: called_out_groups.append("Group 1")
    if callouts['group2']: called_out_groups.append("Group 2") 
    if callouts['group3']: called_out_groups.append("Group 3")
    
    if called_out_groups:
        st.warning(f"🚨 **{driver}** calling out: {', '.join(called_out_groups)}")
        callout_found = True

if not callout_found:
    st.success("✅ No drivers calling out today!")

# Manual callout override
st.subheader("🔧 Manual Callout Override")
st.info("💡 Defaults to all three groups - adjust if driver is only calling out specific groups")
col1, col2 = st.columns(2)

with col1:
    # This should now be instant since data is cached
    driver_list = sorted(set(info['driver'] for info in dogs_going_today.values()))
    selected_driver = st.selectbox("Driver to call out", ["None"] + driver_list)

with col2:
    selected_groups = st.multiselect("Groups affected", [1, 2, 3], default=[1, 2, 3])

# Process reassignments
assignments = []
process_reassignments = False

if selected_driver != "None" and selected_groups and st.button("🚀 Run Manual Callout Reassignment"):
    st.subheader("🔄 Processing Manual Callout Reassignment")
    # Apply manual callout override
    for g in selected_groups:
        driver_callouts[selected_driver][f"group{g}"] = True
    process_reassignments = True
    st.warning(f"🔧 Manual override: {selected_driver} calling out Groups {selected_groups}")

elif callout_found:
    st.subheader("🔄 Processing Automatic Reassignments")
    if st.button("🚀 Run Automatic Reassignment"):
        process_reassignments = True

if process_reassignments:
    # Find dogs that need reassignment
    dogs_to_reassign = []
