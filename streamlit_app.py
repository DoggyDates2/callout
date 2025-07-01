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

url_map = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
url_matrix = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
url_geocodes = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=YOUR_GEOCODES_GID"  # Add your geocodes tab GID here

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
    try:
        # Load from your geocodes Google Sheet tab
        geocodes_df = pd.read_csv(url_geocodes, dtype=str)
        geocodes_dict = {}
        
        for _, row in geocodes_df.iterrows():
            dog_id = str(row.get('Dog ID', '')).strip()
            try:
                lat_val = row.get('LATTITUDE', '')
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
        
        st.info(f"📍 Loaded {len(geocodes_dict)} valid geocoded locations from Google Sheets")
        return geocodes_dict
        
    except Exception as e:
        st.warning(f"⚠️ Could not load geocodes: {e}")
        st.info("💡 Make sure your geocodes tab URL is set correctly")
        return {}

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

def get_driver_color(driver_name):
    """Get a consistent color for each driver"""
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D2B4DE',
        '#AED6F1', '#A3E4D7', '#F9E79F', '#FADBD8', '#D5DBDB'
    ]
    
    # Use hash of driver name to get consistent color
    driver_hash = hash(driver_name) % len(colors)
    return colors[driver_hash]

def create_assignment_map(dogs_going_today, reassignments, geocodes_dict):
    """Create an interactive map showing dog assignments"""
    
    # Create a map centered on a reasonable location (you may want to adjust this)
    m = folium.Map(location=[42.3601, -71.0589], zoom_start=10)  # Boston area - adjust as needed
    
    # Get unique drivers and create color mapping
    all_drivers = set(info['driver'] for info in dogs_going_today.values())
    driver_colors = {driver: get_driver_color(driver) for driver in all_drivers}
    
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
        color = driver_colors.get(driver, '#000000')
        
        # Different icon for reassigned dogs
        if dog_id in reassigned_dog_ids:
            icon = folium.Icon(color='red', icon='refresh', prefix='fa')
            marker_size = 10
        else:
            icon = folium.Icon(color='blue', icon='paw', prefix='fa')  
            marker_size = 8
        
        # Create popup text
        groups_str = '&'.join(map(str, dog_info['groups']))
        assignment_status = "REASSIGNED" if dog_id in reassigned_dog_ids else "Original"
        
        popup_text = f"""
        <b>Dog ID:</b> {dog_id}<br>
        <b>Name:</b> {dog_info.get('dog_name', 'Unknown')}<br>
        <b>Driver:</b> {driver}<br>
        <b>Groups:</b> {groups_str}<br>
        <b>Dogs at stop:</b> {dog_info['num_dogs']}<br>
        <b>Status:</b> {assignment_status}<br>
        <b>Address:</b> {dog_info.get('address', 'Unknown')}
        """
        
        # Add marker to map
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{dog_info.get('dog_name', dog_id)} - {driver}",
            icon=icon
        ).add_to(m)
    
    progress_bar.progress(1.0)
    status_text.text(f"🗺️ Map complete! Cached: {from_cache}, Geocoded: {newly_geocoded}, Failed: {failed_geocode}")
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Map Legend</h4>
    <p><i class="fa fa-paw" style="color:blue"></i> Original Assignment</p>
    <p><i class="fa fa-refresh" style="color:red"></i> Reassigned Dog</p>
    <p><b>Colors:</b> Different drivers</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

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

# Show current callouts from Google Sheets
st.subheader("🚨 Current Driver Callouts (from Google Sheets)")
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

        # Main reassignment logic (same as before)
        for dog in dogs_to_reassign:
            dog_id = dog['dog_id']
            dog_groups = dog['original_groups']
            num_dogs = dog['dog_info']['num_dogs']
            distances = distance_matrix.get(dog_id, {})
            best_driver, best_dist = None, float('inf')
            
            for other_id, dist in distances.items():
                if dist == 0 or other_id not in dogs_going_today:
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
                
                weighted = dist if exact else dist * 2
                if weighted > best_dist:
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
st.info("📍 Blue paw = Original assignment | 🔄 Red refresh = Reassigned dog | Colors = Different drivers")

if st.button("🗺️ Generate Interactive Map"):
    # Load geocodes from Google Sheets
    geocodes_dict = load_geocodes()
    
    with st.spinner("Creating map using geocoded locations..."):
        assignment_map = create_assignment_map(dogs_going_today, assignments, geocodes_dict)
        st_folium(assignment_map, width=1000, height=600)

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
