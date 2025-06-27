import pandas as pd
import numpy as np
import streamlit as st

def load_google_sheet_data():
    """Load data from Google Sheets using correct URLs"""
    try:
        # Distance Matrix - CORRECT URL with right spreadsheet ID and gid
        matrix_url = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=2146002137"
        distance_matrix = pd.read_csv(matrix_url, index_col=0)
        
        # Map Data - CORRECT URL  
        map_url = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        map_data = pd.read_csv(map_url)
        
        # Driver Counts - CORRECT URL
        driver_url = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=1359695250"
        driver_data = pd.read_csv(driver_url)
        
        return distance_matrix, map_data, driver_data
        
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        return None, None, None

def parse_assignment(assignment_str):
    """Parse assignment string like 'Drew:1&2' into driver and groups"""
    if pd.isna(assignment_str) or 'XX' in str(assignment_str):
        return None, []
    
    try:
        parts = str(assignment_str).split(':')
        if len(parts) != 2:
            return None, []
        
        driver = parts[0].strip()
        groups_str = parts[1].strip()
        
        # Parse groups (handle &, numbers, etc.)
        groups = []
        if '&' in groups_str:
            group_parts = groups_str.split('&')
            for part in group_parts:
                if part.isdigit():
                    groups.append(int(part))
        else:
            if groups_str.isdigit():
                groups.append(int(groups_str))
        
        return driver, groups
    except:
        return None, []

def find_callout_drivers(driver_data):
    """Find drivers who called out (have X in their group columns)"""
    callouts = {}
    
    for _, row in driver_data.iterrows():
        driver = row['Driver']
        callout_groups = []
        
        if str(row['Group 1']).upper() == 'X':
            callout_groups.append(1)
        if str(row['Group 2']).upper() == 'X':
            callout_groups.append(2)
        if str(row['Group 3']).upper() == 'X':
            callout_groups.append(3)
        
        if callout_groups:
            callouts[driver] = callout_groups
    
    return callouts

def diagnostic_test():
    """Comprehensive diagnostic of Dog 777"""
    st.title("🔬 Diagnostic: Why Dog 777 Can't Find Matches")
    
    # Load data
    distance_matrix, map_data, driver_data = load_google_sheet_data()
    
    if distance_matrix is None or map_data is None or driver_data is None:
        st.error("Could not load data")
        return
    
    # Test Dog 777 specifically
    dog_id = 777
    test_dog = map_data[map_data['Dog ID'] == dog_id].iloc[0]
    
    st.subheader(f"🐕 Testing Dog {dog_id}: {test_dog['Dog Name']}")
    st.write(f"Assignment: {test_dog['Today']}")
    
    original_driver, original_groups = parse_assignment(test_dog['Today'])
    st.write(f"Parsed - Driver: {original_driver}, Groups: {original_groups}")
    
    # Step 1: Check if dog exists in matrix
    st.subheader("🔍 Step 1: Matrix Lookup")
    
    # Show what's actually in the matrix index
    st.write("**Matrix index details:**")
    st.write(f"Index type: {type(distance_matrix.index)}")
    st.write(f"Index dtype: {distance_matrix.index.dtype}")
    st.write(f"Total items in index: {len(distance_matrix.index)}")
    
    # Show sample of actual index values
    st.write("**First 20 index values:**")
    sample_index = list(distance_matrix.index[:20])
    for i, idx_val in enumerate(sample_index):
        st.write(f"{i}: {repr(idx_val)} (type: {type(idx_val)})")
    
    # Look for 777 in different areas of the index
    st.write("**Looking for 777 in different parts of index:**")
    
    # Check if 777 exists anywhere in the index
    index_list = list(distance_matrix.index)
    
    # Check for 777 as different types
    found_777_locations = []
    for i, idx_val in enumerate(index_list):
        if str(idx_val) == "777" or idx_val == 777 or idx_val == 777.0:
            found_777_locations.append((i, idx_val, type(idx_val)))
    
    if found_777_locations:
        st.success("✅ Found 777 in matrix!")
        for pos, val, typ in found_777_locations:
            st.write(f"Position {pos}: {repr(val)} (type: {typ})")
    else:
        st.error("❌ 777 not found anywhere in matrix index")
        
        # Check nearby values
        st.write("**Checking for values near 777:**")
        near_777 = []
        for idx_val in index_list:
            try:
                num_val = float(str(idx_val))
                if 775 <= num_val <= 780:
                    near_777.append((idx_val, num_val))
            except:
                continue
        
        if near_777:
            st.write("Values near 777:")
            for original, numeric in near_777:
                st.write(f"- {repr(original)} = {numeric}")
        else:
            st.write("No values found near 777")
    
    # Try the matrix lookup with different formats anyway
    matrix_formats = [777, "777", 777.0, "777.0", int(777), float(777)]
    found_format = None
    
    st.write("**Testing different lookup formats:**")
    for fmt in matrix_formats:
        try:
            test_lookup = fmt in distance_matrix.index
            st.write(f"- {repr(fmt)} (type: {type(fmt)}): {test_lookup}")
            if test_lookup and found_format is None:
                found_format = fmt
        except Exception as e:
            st.write(f"- {repr(fmt)}: Error - {e}")
    
    if found_format is None:
        st.error(f"❌ Could not find Dog 777 with any format!")
        st.write("**This means there's a fundamental mismatch between:**")
        st.write("1. How the matrix index is stored")
        st.write("2. How we're trying to look it up")
        return
    
    # Step 2: Get distances
    st.subheader("📏 Step 2: Distance Analysis")
    
    dog_distances = distance_matrix.loc[found_format]
    
    # Find all non-zero distances
    valid_distances = []
    for other_dog, distance in dog_distances.items():
        try:
            dist_float = float(str(distance))
            if dist_float > 0 and dist_float <= 5.0:  # Within 5 miles
                valid_distances.append((other_dog, dist_float))
        except:
            continue
    
    valid_distances.sort(key=lambda x: x[1])
    
    st.write(f"Found {len(valid_distances)} dogs within 5 miles")
    
    if len(valid_distances) > 0:
        st.write("**Closest 10 dogs:**")
        for other_dog, dist in valid_distances[:10]:
            st.write(f"- Dog {other_dog}: {dist:.3f} miles")
    else:
        st.error("❌ No dogs found within 5 miles!")
        return
    
    # Step 3: Check if nearby dogs are going today
    st.subheader("🗓️ Step 3: Going Today Check")
    
    dogs_going_today = map_data[map_data['Today'].notna() & (map_data['Today'] != '')]
    going_today_ids = dogs_going_today['Dog ID'].astype(int).tolist()
    
    st.write(f"Total dogs going today: {len(going_today_ids)}")
    
    nearby_going_today = []
    for other_dog, dist in valid_distances[:20]:  # Check top 20
        try:
            other_dog_int = int(float(str(other_dog)))
            if other_dog_int in going_today_ids:
                nearby_going_today.append((other_dog_int, dist))
        except:
            continue
    
    st.write(f"Nearby dogs also going today: {len(nearby_going_today)}")
    
    if len(nearby_going_today) > 0:
        st.write("**Closest dogs going today:**")
        for other_dog_id, dist in nearby_going_today[:5]:
            nearby_dog_info = dogs_going_today[dogs_going_today['Dog ID'] == other_dog_id]
            if not nearby_dog_info.empty:
                nearby_assignment = nearby_dog_info.iloc[0]['Today']
                st.write(f"- Dog {other_dog_id}: {dist:.3f} miles - {nearby_assignment}")
    else:
        st.error("❌ No nearby dogs are going today!")
        return
    
    # Step 4: Check driver compatibility  
    st.subheader("👥 Step 4: Driver Compatibility")
    
    compatible_matches = []
    for other_dog_id, dist in nearby_going_today[:10]:
        nearby_dog_info = dogs_going_today[dogs_going_today['Dog ID'] == other_dog_id]
        if nearby_dog_info.empty:
            continue
            
        nearby_assignment = nearby_dog_info.iloc[0]['Today']
        candidate_driver, candidate_groups = parse_assignment(nearby_assignment)
        
        # Must have different driver
        if not candidate_driver or candidate_driver == original_driver:
            continue
        
        # Calculate compatibility
        compatibility = 0
        for target_group in original_groups:
            for candidate_group in candidate_groups:
                if target_group == candidate_group:
                    compatibility += 1.0
                elif abs(target_group - candidate_group) == 1:
                    compatibility += 0.5
        
        if compatibility > 0:
            compatible_matches.append({
                'dog_id': other_dog_id,
                'distance': dist,
                'driver': candidate_driver,
                'groups': candidate_groups,
                'compatibility': compatibility,
                'assignment': nearby_assignment
            })
    
    st.write(f"Compatible matches found: {len(compatible_matches)}")
    
    if len(compatible_matches) > 0:
        st.success("✅ MATCHES FOUND! The logic should work!")
        
        best_match = max(compatible_matches, key=lambda x: x['compatibility'])
        st.write("**Best match:**")
        st.write(f"- Dog {best_match['dog_id']}: {best_match['driver']} ({best_match['distance']:.3f} miles)")
        st.write(f"- Groups: {best_match['groups']} (compatibility: {best_match['compatibility']})")
        st.write(f"- Assignment: {best_match['assignment']}")
        
        st.subheader("🎯 Recommended Fix")
        st.success(f"Dog {dog_id} should be reassigned to: **{best_match['driver']}:3**")
        
    else:
        st.error("❌ No compatible matches found")
        st.write("**Possible issues:**")
        st.write("- All nearby dogs have same driver (Drew)")
        st.write("- No group compatibility")
        st.write("- Assignment parsing issues")

if __name__ == "__main__":
    diagnostic_test()
