import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import folium
from streamlit_folium import st_folium
import requests
import time

st.set_page_config(page_title="Dog Reassignment System", layout="wide")
st.title("🐶 Dog Reassignment System")

MAX_DOMINO_DEPTH = 5
EVICTION_DISTANCE_LIMIT = 0.75
FRINGE_DISTANCE_LIMIT = 0.5
REASSIGNMENT_THRESHOLD = 1.5
PLACEMENT_GOAL_DISTANCE = 0.5
MAX_REASSIGNMENT_DISTANCE = 5.0  # Maximum distance for any reassignment

url_map = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
url_matrix = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"

url_geocodes = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQf3D748U4x_SwynVVCR68kuQSB-kKFx80vOZylLm6bJ0CntSUDL1FjPnwXQqQsTRKWanorkConCtnK/pub?output=csv"

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
    else:  # num_dogs > 1 and num_groups > 1
        return 4  # Lowest priority - hardest to place

@st.cache_data
def load_csv(url):
    return pd.read_csv(url, dtype=str)

@st.cache_data
def load_geocodes():
    """Load geocodes from Google Sheets by Dog ID"""
    # Check if geocodes are disabled
    if st.session_state.get('disable_geocodes', False):
        st.info("🔄 Geocodes disabled - using address geocoding only")
        return {}
    
    try:
        # Load from your geocodes Google Sheet tab
        geocodes_df = pd.read_csv(url_geocodes, dtype=str)
        geocodes_dict = {}
        
        for _, row in geocodes_df.iterrows():
            dog_id = str(row.get('Dog ID', '')).strip()
            try:
                lat_val = row.get('LATITUDE', '')  # Updated column name
                lon_val = row.get('LONGITUDE', '')
                
                # Skip if values are empty, NaN, or invalid
                if pd.isna(lat_val) or pd.isna(lon_val) or lat_val == '' or lon_val == '':
                    continue
                
                lat = float(lat_val)
                lon = float(lon_val)
                
                # Skip if coordinates are invalid (0,0 or NaN)
                if pd.isna(lat) or pd.isna(lon) or (lat == 0 and lon == 0):
                    continue
                    
                if dog_id:
                    geocodes_dict[dog_id] = {'lat': lat, 'lon': lon}
                    
            except (ValueError, TypeError):
                continue
        
        st.success(f"📍 Loaded {len(geocodes_dict)} valid geocoded locations from Google Sheets")
        return geocodes_dict
        
    except Exception as e:
        st.warning(f"⚠️ Could not load geocodes: {e}")
        st.info("🔄 Using fallback address geocoding (slower but works)")
        st.info("💡 Try the 'DISABLE GEOCODES' button above for a temporary fix")
        return {}  # Return empty dict to use address geocoding fallback

def get_coordinates_for_dog(dog_id, dog_info, geocodes_dict):
    """Get coordinates for a dog, using geocodes lookup first, then fallback to address geocoding"""
    # Try Dog ID lookup first (fast and reliable)
    if dog_id in geocodes_dict:
        coords = geocodes_dict[dog_id]
        lat, lon = coords['lat'], coords['lon']
        
        # Double-check for valid coordinates
        if pd.notna(lat) and pd.notna(lon) and lat != 0 and lon != 0:
            return lat, lon, "cached"
    
    # Fallback to address geocoding for missing Dog IDs
    address = dog_info.get('address', '')
    if address and address.strip():
        lat, lon, was_geocoded = geocode_address_with_cache(address, {})
        if lat is not None and lon is not None and pd.notna(lat) and pd.notna(lon):
            return lat, lon, "geocoded" if was_geocoded else "failed"
    
    return None, None, "no_location"

def save_to_address_cache(address, lat, lon, cache_dict):
    """Save new coordinates to cache (in memory for now)"""
    cache_dict[address.lower().strip()] = {'lat': lat, 'lon': lon}
    return cache_dict

@st.cache_data
def geocode_address_with_cache(address, _cache_dict=None):
    """Geocode an address using cache first, then Nominatim if needed (fallback only)"""
    if not address or address.strip() == "":
        return None, None, False  # lat, lon, was_geocoded
    
    clean_address = address.strip()
    
    try:
        # Use Nominatim (OpenStreetMap's free geocoding service)
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            'q': clean_address,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }
        
        headers = {
            'User-Agent': 'DogReassignmentSystem/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return lat, lon, True  # True = was geocoded
        
        return None, None, False
        
    except Exception as e:
        return None, None, False

def get_driver_color_name(driver_name):
    """Get a folium-compatible color name for each driver"""
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
        'gray', 'black', 'lightgray'
    ]
    
    # Use hash of driver name to get consistent color
    driver_hash = hash(driver_name) % len(colors)
    return colors[driver_hash]

def create_assignment_map(dogs_going_today, reassignments, geocodes_dict):
    """Create an interactive map showing dog assignments"""
    
    # Calculate current driver loads
    driver_loads = defaultdict(lambda: {'group1': 0, 'group2': 0, 'group3': 0})
    for dog_id, info in dogs_going_today.items():
        for g in info['groups']:
            driver_loads[info['driver']][f'group{g}'] += info['num_dogs']
    
    # Find center point from actual dog locations
    valid_coords = []
    for dog_id, dog_info in dogs_going_today.items():
        lat, lon, source = get_coordinates_for_dog(dog_id, dog_info, geocodes_dict)
        if lat is not None and lon is not None and pd.notna(lat) and pd.notna(lon):
            valid_coords.append([lat, lon])
    
    # Calculate center point
    if valid_coords:
        center_lat = sum(coord[0] for coord in valid_coords) / len(valid_coords)
        center_lon = sum(coord[1] for coord in valid_coords) / len(valid_coords)
    else:
        center_lat, center_lon = 42.3601, -71.0589  # Default to Boston area
    
    # Create map centered on actual data
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Get unique drivers and create color mapping
    all_drivers = set(info['driver'] for info in dogs_going_today.values())
    driver_colors = {driver: get_driver_color_name(driver) for driver in all_drivers}
    
    # Track reassigned dogs
    reassigned_dog_ids = set(r['Dog ID'] for r in reassignments) if reassignments else set()
    
    # Add markers for each dog
    from_cache = 0
    newly_geocoded = 0
    failed_geocode = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_dogs = len(dogs_going_today)
    
    for idx, (dog_id, dog_info) in enumerate(dogs_going_today.items()):
        # Update progress
        progress = (idx + 1) / total_dogs
        progress_bar.progress(progress)
        status_text.text(f"Mapping dogs... {idx + 1}/{total_dogs} (Cached: {from_cache}, Geocoded: {newly_geocoded})")
        
        # Get coordinates using Dog ID lookup first
        lat, lon, source = get_coordinates_for_dog(dog_id, dog_info, geocodes_dict)
        
        # Final validation before creating marker
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            failed_geocode += 1
            continue
        
        if source == "cached":
            from_cache += 1
        elif source == "geocoded":
            newly_geocoded += 1
            # Rate limiting for new geocoding
            time.sleep(0.1)
        
        # Determine marker style
        driver = dog_info['driver']
        driver_color = driver_colors.get(driver, 'gray')
        
        # Get driver capacity info
        driver_capacity = driver_capacities.get(driver, {'group1': 9, 'group2': 9, 'group3': 9})
        driver_load = driver_loads.get(driver, {'group1': 0, 'group2': 0, 'group3': 0})
        
        # Calculate available capacity
        capacity_info = []
        for group_num in [1, 2, 3]:
            current = driver_load[f'group{group_num}']
            total = driver_capacity[f'group{group_num}']
            available = total - current
            capacity_info.append(f"Group {group_num}: {current}/{total} ({available} available)")
        
        # Different icon for reassigned dogs
        if dog_id in reassigned_dog_ids:
            icon = folium.Icon(
                color=driver_color, 
                icon='refresh'
            )
        else:
            icon = folium.Icon(
                color=driver_color,
                icon='info-sign'
            )
        
        # Create popup text with capacity info
        groups_str = '&'.join(map(str, dog_info['groups']))
        assignment_status = "REASSIGNED" if dog_id in reassigned_dog_ids else "Original"
        
        popup_text = f"""
        <div style="font-family: Arial; font-size: 12px;">
        <b>Dog ID:</b> {dog_id}<br>
        <b>Name:</b> {dog_info.get('dog_name', 'Unknown')}<br>
        <b>Driver:</b> <span style="color: {driver_color}; font-weight: bold;">{driver}</span><br>
        <b>Groups:</b> {groups_str}<br>
        <b>Dogs at stop:</b> {dog_info['num_dogs']}<br>
        <b>Status:</b> {assignment_status}<br>
        <b>Address:</b> {dog_info.get('address', 'Unknown')}<br>
        <hr>
        <b style="color: #0066cc;">Driver Capacity:</b><br>
        {'<br>'.join(capacity_info)}
        </div>
        """
        
        # Create tooltip with driver:groups and capacity summary
        total_current = sum(driver_load.values())
        total_capacity = sum(driver_capacity.values())
        tooltip_text = f"{driver}:{groups_str} ({total_current}/{total_capacity})"
        
        # Add marker to map
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=350),
            tooltip=tooltip_text,
            icon=icon
        ).add_to(m)
    
    progress_bar.progress(1.0)
    status_text.text(f"🗺️ Map complete! Cached: {from_cache}, Geocoded: {newly_geocoded}, Failed: {failed_geocode}")
    
    return m, driver_loads

def generate_radius_steps(start=0.25, max_radius=2.2, step=0.25):
    current = start
    while current <= max_radius:
        yield round(current, 3)
        current += step

def is_adjacent(g1, g2):
    return abs(g1 - g2) == 1

def parse_cap(val):
    try:
        return 9 if val in ["", "X", "NAN"] else int(val)
    except:
        return 9

def get_groups(group_str):
    group_str = group_str.replace("LM", "")
    return sorted(set(int(g) for g in re.findall(r'\d', group_str)))

# Load and clean
map_df = load_csv(url_map)
matrix_df = load_csv(url_matrix)

map_df["Dog ID"] = map_df["Dog ID"].astype(str).str.strip()
map_df["Name"] = map_df["Name"].astype(str).str.strip()
map_df["Group"] = map_df["Group"].astype(str).str.strip()

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
            'address': row.get("Address", ""),
            'dog_name': row.get("Dog Name", "")
        }

# Driver capacities + callouts (from Google Sheets)
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

# Distance matrix
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

# REASSIGNMENT DEBUG CENTER
st.subheader("🔍 REASSIGNMENT DEBUG CENTER")
with st.expander("🐛 Click to Debug What's Broken"):
    
    if st.button("🕵️ Run Full Diagnostic"):
        st.write("### **Step 1: Callout Detection Test**")
        
        # Check if we're reading callouts correctly
        callout_count = 0
        callout_details = []
        
        for driver, callouts in driver_callouts.items():
            called_out_groups = []
            if callouts.get('group1', False): called_out_groups.append("Group 1")
            if callouts.get('group2', False): called_out_groups.append("Group 2") 
            if callouts.get('group3', False): called_out_groups.append("Group 3")
            
            if called_out_groups:
                callout_count += 1
                callout_details.append(f"✅ {driver}: {', '.join(called_out_groups)}")
            else:
                # Show drivers with no callouts for verification
                callout_details.append(f"⚪ {driver}: No callouts")
        
        st.write(f"**Found {callout_count} drivers with callouts:**")
        for detail in callout_details[:10]:  # Show first 10
            st.write(f"  {detail}")
        
        if callout_count == 0:
            st.error("🚨 **PROBLEM FOUND**: No callouts detected!")
            st.write("**Possible causes:**")
            st.write("• Column names changed in Google Sheet (should be 'Group 1', 'Group 2', 'Group 3')")
            st.write("• No 'X' values in callout columns")
            st.write("• Driver names column issue")
            
            # Show raw data sample
            st.write("**Raw Google Sheet Sample:**")
            sample_cols = ['Driver', 'Group 1', 'Group 2', 'Group 3']
            available_cols = [col for col in sample_cols if col in map_df.columns]
            if available_cols:
                st.dataframe(map_df[available_cols].head(3))
            else:
                st.error(f"❌ Expected columns {sample_cols} not found!")
                st.write(f"Available columns: {list(map_df.columns)}")
        
        st.write("---")
        st.write("### **Step 2: Dogs Affected Test**")
        
        # Find dogs that should be reassigned
        dogs_needing_reassignment = []
        for dog_id, info in dogs_going_today.items():
            driver = info["driver"]
            if driver not in driver_callouts:
                continue
            
            affected_groups = []
            for g in info["groups"]:
                if driver_callouts[driver].get(f"group{g}", False):
                    affected_groups.append(g)
            
            if affected_groups:
                dogs_needing_reassignment.append({
                    'dog_id': dog_id,
                    'driver': driver,
                    'groups': info['groups'],
                    'affected_groups': affected_groups
                })
        
        st.write(f"**Found {len(dogs_needing_reassignment)} dogs needing reassignment:**")
        if dogs_needing_reassignment:
            for dog in dogs_needing_reassignment[:5]:  # Show first 5
                st.write(f"  🐕 {dog['dog_id']}: {dog['driver']} (Groups {dog['groups']}, affected: {dog['affected_groups']})")
        else:
            st.error("🚨 **PROBLEM FOUND**: No dogs found for reassignment!")
            st.write("**Possible causes:**")
            st.write("• Driver names don't match between main sheet and callouts")
            st.write("• Groups not parsing correctly")
            st.write("• Logic error in affected group detection")
        
        st.write("---")
        st.write("### **Step 3: Distance Matrix Test**")
        
        if dogs_needing_reassignment:
            test_dog = dogs_needing_reassignment[0]
            dog_id = test_dog['dog_id']
            
            distances = distance_matrix.get(dog_id, {})
            st.write(f"**Testing distances for Dog {dog_id}:**")
            
            if not distances:
                st.error(f"🚨 **PROBLEM FOUND**: Dog {dog_id} not found in distance matrix!")
                st.write("**Available dogs in distance matrix:**")
                st.write(list(distance_matrix.keys())[:10])  # Show first 10
            else:
                # Show nearby dogs
                valid_distances = [(other_id, dist) for other_id, dist in distances.items() 
                                 if dist > 0 and dist <= 5.0 and other_id in dogs_going_today]
                valid_distances.sort(key=lambda x: x[1])  # Sort by distance
                
                st.write(f"**Found {len(valid_distances)} dogs within 5 miles:**")
                for other_id, dist in valid_distances[:5]:  # Show closest 5
                    other_driver = dogs_going_today[other_id]['driver']
                    other_groups = dogs_going_today[other_id]['groups']
                    st.write(f"  📍 {other_id}: {dist} miles → {other_driver} (Groups {other_groups})")
        
        st.write("---")
        st.write("### **Step 4: Driver Capacity Test**")
        
        st.write(f"**Found {len(driver_capacities)} drivers with capacity data:**")
        for driver, capacity in list(driver_capacities.items())[:5]:  # Show first 5
            st.write(f"  👨‍💼 {driver}: G1={capacity['group1']}, G2={capacity['group2']}, G3={capacity['group3']}")
        
        # Check for capacity issues
        missing_capacity_drivers = set(info['driver'] for info in dogs_going_today.values()) - set(driver_capacities.keys())
        if missing_capacity_drivers:
            st.warning(f"⚠️ **Missing capacity data for:** {list(missing_capacity_drivers)[:5]}")
        
        st.write("---")
        st.success("🎯 **Diagnostic complete!** Check above for any red error messages.")

    # Quick manual test buttons
    st.write("**Quick Tests:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Check Sheet Columns"):
            expected = ['Dog ID', 'Name', 'Group', 'Driver', 'Group 1', 'Group 2', 'Group 3']
            available = list(map_df.columns)
            missing = [col for col in expected if col not in available]
            
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                st.success("✅ All expected columns found!")
            
            st.write(f"**Available columns:** {available}")
    
    with col2:
        if st.button("🎯 Test Group Parsing"):
            sample_groups = map_df['Group'].head(5).tolist()
            st.write("**Sample group strings:**")
            for group_str in sample_groups:
                parsed = get_groups(group_str)
                st.write(f"  '{group_str}' → {parsed}")
    
    with col3:
        if st.button("🔗 Check Driver Matching"):
            main_drivers = set(info['driver'] for info in dogs_going_today.values())
            callout_drivers = set(driver_callouts.keys())
            capacity_drivers = set(driver_capacities.keys())
            
            st.write(f"**Main sheet drivers:** {len(main_drivers)}")
            st.write(f"**Callout drivers:** {len(callout_drivers)}")
            st.write(f"**Capacity drivers:** {len(capacity_drivers)}")
            
            missing_in_callouts = main_drivers - callout_drivers
            if missing_in_callouts:
                st.error(f"❌ Missing in callouts: {list(missing_in_callouts)[:3]}")
            else:
                st.success("✅ All drivers found in callouts!")

# Show current callouts from Google Sheets
st.subheader("🚨 Current Driver Callouts (from Google Sheets)")

# Distance limits info
st.info(f"🛡️ **Distance Limits:** Max {MAX_REASSIGNMENT_DISTANCE} miles total | Exact group match: 3 miles | Adjacent group: 1.5 miles")

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

# Debug section for distance matrix
with st.expander("🔍 Debug: Check Distance Matrix Data"):
    if st.button("Check for suspicious distances"):
        suspicious_distances = []
        high_distances = []
        
        for dog_id, distances in distance_matrix.items():
            for other_id, dist in distances.items():
                if dist == 100:  # Exact 100 mile matches are suspicious
                    suspicious_distances.append(f"{dog_id} ↔ {other_id}: {dist} miles")
                elif dist > 10:  # High distances
                    high_distances.append(f"{dog_id} ↔ {other_id}: {dist} miles")
        
        if suspicious_distances:
            st.warning(f"🚨 Found {len(suspicious_distances)} suspicious 100-mile distances:")
            for item in suspicious_distances[:10]:  # Show first 10
                st.write(f"  - {item}")
        
        if high_distances:
            st.info(f"ℹ️ Found {len(high_distances)} distances > 10 miles (showing first 10):")
            for item in high_distances[:10]:
                st.write(f"  - {item}")
        
        if not suspicious_distances and not high_distances:
            st.success("✅ No suspicious distances found!")

# Manual callout override section
st.subheader("🔧 Manual Callout Override (Testing)")
st.info("💡 Use this to test 'what if' scenarios or handle last-minute callouts")

selected_driver = st.selectbox("Driver to call out", ["None"] + sorted(set(info['driver'] for info in dogs_going_today.values())))
selected_groups = st.multiselect("Groups affected", [1, 2, 3], default=[])

# Process reassignments
assignments = []
process_reassignments = False

# Check if manual override is being used
if selected_driver != "None" and selected_groups and st.button("🚀 Run Manual Callout Reassignment"):
    st.subheader("🔄 Processing Manual Callout Reassignment")
    # Apply manual callout override
    for g in selected_groups:
        driver_callouts[selected_driver][f"group{g}"] = True
    process_reassignments = True
    st.warning(f"🔧 Manual override: {selected_driver} calling out Groups {selected_groups}")

# Auto-run reassignment if there are callouts from Google Sheets
elif callout_found:
    st.subheader("🔄 Processing Automatic Reassignments")
    process_reassignments = True

if process_reassignments:
    dogs_to_reassign = []
    for dog_id, info in dogs_going_today.items():
        driver = info["driver"]
        if driver not in driver_callouts:
            continue
        
        affected = []
        for g in info["groups"]:
            if driver_callouts[driver].get(f"group{g}", False):
                affected.append(g)
        
        if affected:
            dogs_to_reassign.append({
                'dog_id': dog_id,
                'original_driver': driver,
                'original_groups': info['groups'],
                'dog_info': info
            })

    if dogs_to_reassign:
        # Sort dogs by priority (easiest assignments first)
        dogs_to_reassign.sort(key=get_reassignment_priority)
        st.info(f"🎯 Processing {len(dogs_to_reassign)} dogs in priority order (single dog/group first)")

        driver_loads = defaultdict(lambda: {'group1': 0, 'group2': 0, 'group3': 0})
        for dog_id, info in dogs_going_today.items():
            for g in info['groups']:
                driver_loads[info['driver']][f'group{g}'] += info['num_dogs']

        fringe_moves = []
        domino_evictions = []

        # Main reassignment logic with distance limits
        for dog in dogs_to_reassign:
            dog_id = dog['dog_id']
            dog_groups = dog['original_groups']
            num_dogs = dog['dog_info']['num_dogs']
            distances = distance_matrix.get(dog_id, {})
            best_driver, best_dist = None, float('inf')
            
            for other_id, dist in distances.items():
                # Skip if distance is 0, too far, or not in today's assignments
                if dist == 0 or dist > MAX_REASSIGNMENT_DISTANCE or other_id not in dogs_going_today:
                    continue
                    
                candidate_driver = dogs_going_today[other_id]['driver']
                if candidate_driver == dog['original_driver'] or candidate_driver not in driver_capacities:
                    continue
                
                # Skip if candidate driver is also calling out
                skip_candidate = False
                for g in dog_groups:
                    if driver_callouts[candidate_driver].get(f"group{g}", False):
                        skip_candidate = True
                        break
                if skip_candidate:
                    continue
                
                other_groups = dogs_going_today[other_id]['groups']
                exact = any(g in other_groups for g in dog_groups)
                adjacent = any(is_adjacent(g1, g2) for g1 in dog_groups for g2 in other_groups)
                if not (exact or adjacent):
                    continue
                
                # Apply stricter distance limits for different match types
                weighted = dist if exact else dist * 2
                max_allowed = 3.0 if exact else 1.5  # Exact: 3 miles, Adjacent: 1.5 miles
                
                if weighted > max_allowed or weighted > best_dist:
                    continue
                
                fits = all(driver_loads[candidate_driver][f"group{g}"] + num_dogs <= driver_capacities[candidate_driver][f"group{g}"] for g in dog_groups)
                if fits:
                    best_driver, best_dist = candidate_driver, weighted

            if best_driver:
                for g in dog_groups:
                    driver_loads[best_driver][f"group{g}"] += num_dogs
                dogs_going_today[dog_id]['driver'] = best_driver
                dogs_going_today[dog_id]['assignment'] = f"{best_driver}:{'&'.join(map(str, dog_groups))}"
                assignments.append({
                    "Dog ID": dog_id,
                    "New Assignment": f"{best_driver}:{'&'.join(map(str, dog_groups))}",
                    "Distance": round(best_dist, 3)
                })
            else:
                # Log dogs that couldn't be reassigned within distance limits
                st.warning(f"⚠️ Could not reassign {dog_id} within {MAX_REASSIGNMENT_DISTANCE}-mile limit")

        if assignments:
            result_df = pd.DataFrame(assignments)[["Dog ID", "New Assignment", "Distance"]]
            st.success("✅ Final Reassignments")
            st.dataframe(result_df)
        else:
            st.info("ℹ️ No reassignments could be made with current constraints")
    else:
        st.info("ℹ️ No dogs need reassignment")

# Create and display the map
st.subheader("🗺️ Assignment Map")
st.info("📍 Hover for Driver:Groups (used/capacity) | Click markers for full capacity details | Colors = Different drivers")

# Quick geocodes troubleshooting
with st.expander("🔧 Data Source Status"):
    st.success("🚀 **ALL SHEETS NOW USING PUBLISHED URLs** (Most Reliable!)")
    
    st.write("**📋 Main Assignments Sheet (Published):**")
    st.code(url_map, language="text")
    
    st.write("**📊 Distance Matrix Sheet (Published):**") 
    st.code(url_matrix, language="text")
    
    st.write("**📍 Geocodes Sheet (Published):**")
    st.code(url_geocodes, language="text")
    
    # Individual URL testing
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Test Main Sheet"):
            try:
                test_df = pd.read_csv(url_map, nrows=3)
                st.success("✅ Main sheet works!")
                st.dataframe(test_df.head(2))
            except Exception as e:
                st.error(f"❌ Main sheet failed: {e}")
    
        if st.button("🧪 Test Matrix Sheet"):
            try:
                test_df = pd.read_csv(url_matrix, nrows=3) 
                st.success("✅ Matrix sheet works!")
                st.dataframe(test_df.head(2))
            except Exception as e:
                st.error(f"❌ Matrix sheet failed: {e}")
    
    with col2:
        if st.button("🧪 Test Geocodes Sheet"):
            try:
                # Show the actual URL being tested
                st.write(f"**Testing URL:** {url_geocodes}")
                test_df = pd.read_csv(url_geocodes, nrows=5)
                st.success("✅ Geocodes sheet works!")
                st.dataframe(test_df.head(3))
                st.info(f"Columns: {list(test_df.columns)}")
            except Exception as e:
                st.error(f"❌ Geocodes failed: {e}")
                st.write("**Current URL being used:**")
                st.code(url_geocodes)
                
                st.write("**Let's try alternative approaches:**")
                
                # Alternative 1: Try with gid parameter
                alt_url1 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQf3D748U4x_SwynVVCR68kuQSB-kKFx80vOZylLm6bJ0CntSUDL1FjPnwXQqQsTRKWanorkConCtnK/pub?gid=0&single=true&output=csv"
                
                # Alternative 2: Try the original export format
                alt_url2 = "https://docs.google.com/spreadsheets/d/1SuvioVrk1aRwXQpPdYxNnzvyCsb4Opquq-9WSiASvi0/export?format=csv&gid=0"
                
                st.write("**Alternative 1 (with gid):**")
                st.code(alt_url1)
                
                st.write("**Alternative 2 (export format):**")
                st.code(alt_url2)
                
                if st.button("🧪 Test Alternative 1"):
                    try:
                        test_alt1 = pd.read_csv(alt_url1, nrows=3)
                        st.success("✅ Alternative 1 works!")
                        st.info("Copy this URL to replace url_geocodes in the code")
                    except Exception as e2:
                        st.error(f"❌ Alternative 1 failed: {e2}")
                
                if st.button("🧪 Test Alternative 2"):
                    try:
                        test_alt2 = pd.read_csv(alt_url2, nrows=3)
                        st.success("✅ Alternative 2 works!")
                        st.info("Copy this URL to replace url_geocodes in the code")
                    except Exception as e3:
                        st.error(f"❌ Alternative 2 failed: {e3}")
    
    # Temporary disable option
    st.write("---")
    st.write("**🔧 Temporary Fix While We Debug:**")
    if st.button("⚡ DISABLE GEOCODES (Use Address Lookup Instead)"):
        st.session_state['disable_geocodes'] = True
        st.success("✅ Geocodes disabled! Maps will use address lookup (slower but works)")
        st.info("Maps will take 2-3 minutes but will work reliably")
    
    if st.button("🔄 RE-ENABLE GEOCODES"):
        st.session_state['disable_geocodes'] = False
        st.success("✅ Geocodes re-enabled!")
    
    # Show current status
    geocodes_disabled = st.session_state.get('disable_geocodes', False)
    if geocodes_disabled:
        st.warning("🔄 Currently using address geocoding (geocodes disabled)")
    else:
        st.info("📍 Currently trying to use geocodes (fast mode)")
    
    st.info("💡 **Benefits**: No permission issues, faster loading, bulletproof reliability!")

if st.button("🗺️ Generate Interactive Map"):
    # Load geocodes from Google Sheets
    st.write("🔄 Step 1: Loading geocodes...")
    geocodes_dict = load_geocodes()
    st.write(f"✅ Step 1 complete: {len(geocodes_dict)} geocodes loaded")
    
    st.write("🔄 Step 2: Creating map...")
    with st.spinner("Creating map using geocoded locations..."):
        try:
            # Get the map and driver loads separately
            result = create_assignment_map(dogs_going_today, assignments, geocodes_dict)
            folium_map = result[0]  # The folium map object
            driver_loads_data = result[1]  # The driver loads data
            st.write("✅ Step 2 complete: Map created successfully")
            
            st.write("🔄 Step 3: Displaying map...")
            st_folium(folium_map, width=1000, height=600)
            st.write("✅ Step 3 complete: Map displayed")
            
        except Exception as e:
            st.error(f"❌ Map creation failed: {e}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.error(f"Full traceback: {traceback.format_exc()}")

# Summary statistics
st.subheader("📊 Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Dogs", len(dogs_going_today))
with col2:
    st.metric("Drivers", len(set(info['driver'] for info in dogs_going_today.values())))
with col3:
    st.metric("Reassignments", len(assignments))
with col4:
    callout_count = sum(1 for callouts in driver_callouts.values() if any(callouts.values()))
    st.metric("Drivers Called Out", callout_count)
