import streamlit as st
import pandas as pd
import re
from collections import defaultdict

.set_page_config(page_title="🐶 Dog Assignment Tool", layout="wide")
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
            dogs_going_today[dog_id] = {
                'assignment': f"{driver}:{group_str}",
                'num_dogs': num_dogs,
                'driver': driver,
                'groups': get_groups(group_str),
                'available_for_proximity': is_available_for_proximity(group_str),
                'address': row.get("Address", ""),
                'dog_name': row.get("Dog Name", "")
            }
    
    return dogs_going_today

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
    group_str = group_str.replace("LM", "")
    
    # Special case: "2XX2" should be treated as "2"
    if "XX" in group_str and any(char.isdigit() for char in group_str):
        # Extract digits from strings like "2XX2" -> "2"
        digits_only = ''.join(char for char in group_str if char.isdigit())
        if digits_only:
            return sorted(set(int(g) for g in digits_only))
    
    # Regular case: extract all digits
    return sorted(set(int(g) for g in re.findall(r'\d', group_str)))

def is_available_for_proximity(group_str):
    """Check if this assignment can be used for proximity matching"""
    # "XX" (without digits) is not available for proximity
    if group_str.strip().upper() == "XX":
        return False
    return True

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
    st.metric("Total Dogs", len(dogs_going_today))
with col2:
    st.metric("Active Drivers", len(set(info['driver'] for info in dogs_going_today.values())))
with col3:
    callout_count = sum(1 for callouts in driver_callouts.values() if any(callouts.values()))
    st.metric("Drivers Called Out", callout_count)
with col4:
    total_capacity = sum(sum(cap.values()) for cap in driver_capacities.values())
    st.metric("Total Driver Capacity", total_capacity)

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

# Show XX handling info
st.info("ℹ️ **Special Cases:** Group 0 (waypoints) and XX (admin slots) count toward capacity but stay with original driver. Exception: 2XX2 = Group 2")

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

if selected_driver != "None" and selected_groups:
    if st.button("🚀 Run Manual Callout Reassignment"):
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
    for dog_id, info in dogs_going_today.items():
        driver = info["driver"]
        if driver not in driver_callouts:
            continue
        
        affected = []
        for g in info["groups"]:
            # MINIMAL FIX: Only process groups 1, 2, 3 for callouts
            if g in [1, 2, 3] and driver_callouts[driver].get(f"group{g}", False):
                affected.append(g)
        
        if affected:
            dogs_to_reassign.append({
                'dog_id': dog_id,
                'original_driver': driver,
                'original_groups': info['groups'],
                'dog_info': info
            })

    if dogs_to_reassign:
        # Sort by priority
        dogs_to_reassign.sort(key=get_reassignment_priority)
        st.info(f"🎯 Processing {len(dogs_to_reassign)} dogs in priority order")

        # Calculate current driver loads
        driver_loads = defaultdict(lambda: {'group1': 0, 'group2': 0, 'group3': 0})
        for dog_id, info in dogs_going_today.items():
            for g in info['groups']:
                # MINIMAL FIX: Only count groups 1, 2, 3 toward capacity
                if g in [1, 2, 3]:
                    driver_loads[info['driver']][f'group{g}'] += info['num_dogs']

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Process each dog
        start_time = time.time()
        for idx, dog in enumerate(dogs_to_reassign):
            progress = (idx + 1) / len(dogs_to_reassign)
            progress_bar.progress(progress)
            status_text.text(f"Processing dog {idx + 1}/{len(dogs_to_reassign)}: {dog['dog_id']}")

            dog_id = dog['dog_id']
            dog_groups = dog['original_groups']
            # MINIMAL FIX: Filter out group 0 from reassignment consideration
            delivery_groups = [g for g in dog_groups if g in [1, 2, 3]]
            if not delivery_groups:  # Skip if only group 0
                continue
                
            num_dogs = dog['dog_info']['num_dogs']
            distances = distance_matrix.get(dog_id, {})
            best_driver, best_dist = None, float('inf')
            
            for other_id, dist in distances.items():
                # Skip invalid distances
                if dist == 0 or dist > MAX_REASSIGNMENT_DISTANCE or other_id not in dogs_going_today:
                    continue
                
                # Skip if this location is not available for proximity (like "XX")
                if not dogs_going_today[other_id].get('available_for_proximity', True):
                    continue
                    
                candidate_driver = dogs_going_today[other_id]['driver']
                if candidate_driver == dog['original_driver'] or candidate_driver not in driver_capacities:
                    continue
                
                # Skip if candidate driver is also calling out
                skip_candidate = False
                for g in delivery_groups:  # Use filtered delivery groups
                    if driver_callouts[candidate_driver].get(f"group{g}", False):
                        skip_candidate = True
                        break
                if skip_candidate:
                    continue
                
                # Check group compatibility
                other_groups = [g for g in dogs_going_today[other_id]['groups'] if g in [1, 2, 3]]  # Filter other groups too
                exact = any(g in other_groups for g in delivery_groups)
                adjacent = any(is_adjacent(g1, g2) for g1 in delivery_groups for g2 in other_groups)
                if not (exact or adjacent):
                    continue
                
                # Apply distance limits
                weighted = dist if exact else dist * 2
                max_allowed = 3.0 if exact else 1.5
                
                if weighted > max_allowed or weighted > best_dist:
                    continue
                
                # Check capacity
                fits = all(driver_loads[candidate_driver][f"group{g}"] + num_dogs <= driver_capacities[candidate_driver][f"group{g}"] for g in delivery_groups)
                if fits:
                    best_driver, best_dist = candidate_driver, weighted

            # Make assignment
            if best_driver:
                for g in delivery_groups:  # Use filtered delivery groups
                    driver_loads[best_driver][f"group{g}"] += num_dogs
                dogs_going_today[dog_id]['driver'] = best_driver
                dogs_going_today[dog_id]['assignment'] = f"{best_driver}:{'&'.join(map(str, delivery_groups))}"
                assignments.append({
                    "Dog ID": dog_id,
                    "Dog Name": dog['dog_info'].get('dog_name', 'Unknown'),
                    "Original Driver": dog['original_driver'],
                    "New Driver": best_driver,
                    "Groups": '&'.join(map(str, delivery_groups)),
                    "Distance": round(best_dist, 2),
                    "Match Type": "Exact" if best_dist <= 3.0 else "Adjacent"
                })

        processing_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Reassignment processing complete in {processing_time:.1f} seconds!")

        # Show results
        if assignments:
            st.success(f"✅ Successfully reassigned {len(assignments)} dogs!")
            result_df = pd.DataFrame(assignments)
            st.dataframe(result_df, use_container_width=True)
            
            # Download button for results
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Reassignments CSV",
                data=csv,
                file_name="dog_reassignments.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No dogs could be reassigned within distance constraints")
    else:
        st.info("ℹ️ No dogs need reassignment")

# Driver capacity analysis
st.subheader("📈 Driver Capacity Analysis")
if st.button("📊 Show Current Driver Loads"):
    driver_loads = defaultdict(lambda: {'group1': 0, 'group2': 0, 'group3': 0})
    for dog_id, info in dogs_going_today.items():
        for g in info['groups']:
            # MINIMAL FIX: Only count groups 1, 2, 3 toward capacity
            if g in [1, 2, 3]:
                driver_loads[info['driver']][f'group{g}'] += info['num_dogs']
    
    capacity_data = []
    overloaded_drivers = []
    
    for driver, capacity in driver_capacities.items():
        load = driver_loads.get(driver, {'group1': 0, 'group2': 0, 'group3': 0})
        
        # Calculate available spots for each group
        group1_available = capacity['group1'] - load['group1']
        group2_available = capacity['group2'] - load['group2'] 
        group3_available = capacity['group3'] - load['group3']
        
        # Format: "available1, available2, available3"
        availability_display = f"{group1_available}, {group2_available}, {group3_available}"
        
        capacity_data.append({
            'Driver': driver,
            'Available Spots (G1, G2, G3)': availability_display
        })
        
        # Track overloaded drivers
        if group1_available < 0 or group2_available < 0 or group3_available < 0:
            overloaded_drivers.append({
                'Driver': driver,
                'Group 1': f"{load['group1']}/{capacity['group1']} ({group1_available})",
                'Group 2': f"{load['group2']}/{capacity['group2']} ({group2_available})",
                'Group 3': f"{load['group3']}/{capacity['group3']} ({group3_available})"
            })
    
    # Sort by driver name alphabetically
    capacity_data.sort(key=lambda x: x['Driver'])
    
    capacity_df = pd.DataFrame(capacity_data)
    st.dataframe(capacity_df, use_container_width=True)
    
    # Highlight overloaded drivers
    if overloaded_drivers:
        st.error("🚨 Overloaded drivers detected!")
        overloaded_df = pd.DataFrame(overloaded_drivers)
        st.dataframe(overloaded_df, use_container_width=True)
    else:
        st.success("✅ No drivers are overloaded!")

# Smart cache info
with st.expander("💾 Smart Cache Strategy Details"):
    st.write("**🧠 Intelligent Caching Based on Real-World Usage:**")
    st.write("")
    st.write("📅 **Distance Matrix: 1 WEEK cache**")
    st.write("   • Changes rarely (weekly route updates)")
    st.write("   • Heaviest computation (biggest performance gain)")
    st.write("   • Only rebuild when routes actually change")
    st.write("")
    st.write("⏱️ **Daily Assignments: 2 MINUTE cache**")
    st.write("   • Changes frequently throughout the day")
    st.write("   • Balance between performance and freshness")
    st.write("   • Use 'REFRESH ALL DATA' for immediate updates")
    st.write("")
    st.write("⚡ **Driver Callouts: 1 MINUTE cache**")
    st.write("   • Most dynamic data (last-minute callouts)")
    st.write("   • Near real-time updates")
    st.write("   • Critical for accurate reassignments")
    st.write("")
    st.write("🎯 **Result: Distance matrix stays fast, operational data stays fresh!**")
    st.write("")
    st.write("💡 **Pro Tip:** Use 'REFRESH ALL DATA' button when you make changes in Google Sheets!")

# Quick stats
with st.expander("📋 Quick Data Summary"):
    st.write(f"**Dogs in system:** {len(dogs_going_today)}")
    st.write(f"**Active drivers:** {len(driver_capacities)}")
    st.write(f"**Distance matrix size:** {len(distance_matrix)} x {len(matrix_df.columns)-1}")
    st.write(f"**Total driver capacity:** {sum(sum(cap.values()) for cap in driver_capacities.values())}")
    st.write(f"**Current callouts:** {callout_count}")
