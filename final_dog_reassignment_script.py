import pandas as pd
import numpy as np
import streamlit as st

def load_google_sheet_data():
    """Load data from Google Sheets using correct URLs"""
    try:
        # Distance Matrix - CORRECT URL
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

def get_dogs_within_distance(dog_id, distance_matrix, max_distance, dogs_going_today):
    """Find dogs within specified distance of target dog with debugging - FIXED"""
    # Convert float to int then to string for matrix lookup
    dog_id_int = int(float(dog_id))  # Handle 777.0 -> 777
    dog_id_str = str(dog_id_int)     # Convert to string for matrix
    
    nearby_dogs = []
    
    st.write(f"  🔍 Looking for nearby dogs to Dog {dog_id} (converted to {dog_id_str})...")
    
    # Check if dog exists in matrix
    if dog_id_str not in distance_matrix.index:
        st.write(f"  ❌ Dog {dog_id_str} not found in distance matrix index")
        st.write(f"  📋 Matrix index sample: {list(distance_matrix.index[:10])}")
        return []
    
    st.write(f"  ✅ Dog {dog_id_str} found in distance matrix")
    
    # Get the row for this dog
    dog_row = distance_matrix.loc[dog_id_str]
    
    distance_count = 0
    valid_distance_count = 0
    in_map_count = 0
    
    # Check all other dogs
    for other_dog_id_str in distance_matrix.columns:
        if other_dog_id_str == dog_id_str:
            continue
            
        distance_count += 1
        
        try:
            distance = dog_row[other_dog_id_str]
            
            if pd.notna(distance) and distance > 0 and distance <= max_distance:
                valid_distance_count += 1
                
                # Convert to int and check if in map data
                try:
                    other_dog_id_int = int(other_dog_id_str)
                    
                    # Convert map data Dog IDs to int for comparison
                    map_dog_ids = dogs_going_today['Dog ID'].astype(int)
                    
                    if other_dog_id_int in map_dog_ids.values:
                        in_map_count += 1
                        nearby_dogs.append((other_dog_id_int, distance))
                        
                except ValueError:
                    continue
        except:
            continue
    
    st.write(f"  📊 Checked {distance_count} distances")
    st.write(f"  📏 Found {valid_distance_count} within {max_distance} miles")  
    st.write(f"  🗺️ Found {in_map_count} also going today")
    st.write(f"  ✅ Final nearby dogs: {len(nearby_dogs)}")
    
    return sorted(nearby_dogs, key=lambda x: x[1])

def find_best_reassignment(dog_to_move, current_assignments, distance_matrix):
    """Find the best driver to reassign a dog to with full debugging"""
    dog_id = dog_to_move['Dog ID']
    original_driver, original_groups = parse_assignment(dog_to_move['Today'])
    
    st.write(f"🎯 **Finding reassignment for Dog {dog_id}: {dog_to_move['Dog Name']}**")
    st.write(f"  Original: {original_driver}, Groups: {original_groups}")
    
    if not original_groups:
        st.write("  ❌ No groups found in assignment")
        return None
    
    # Use generous distance threshold
    max_distance = 3.0
    st.write(f"  🔍 Searching within {max_distance} miles...")
    
    # Get nearby dogs
    nearby_dogs = get_dogs_within_distance(dog_id, distance_matrix, max_distance, current_assignments)
    
    if not nearby_dogs:
        st.write("  ❌ No nearby dogs found")
        return None
    
    st.write(f"  ✅ Found {len(nearby_dogs)} nearby dogs")
    
    # Test matching logic
    candidates = []
    checked_count = 0
    
    for nearby_dog_id, distance in nearby_dogs[:20]:  # Check top 20 nearest
        checked_count += 1
        
        # Find the nearby dog's data
        nearby_dog = current_assignments[current_assignments['Dog ID'] == nearby_dog_id]
        if nearby_dog.empty:
            continue
        
        nearby_dog = nearby_dog.iloc[0]
        candidate_driver, candidate_groups = parse_assignment(nearby_dog['Today'])
        
        # Must have a different driver
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
            candidates.append({
                'driver': candidate_driver,
                'groups': candidate_groups,
                'distance': distance,
                'compatibility': compatibility,
                'nearby_dog_id': nearby_dog_id,
                'nearby_dog_name': nearby_dog['Dog Name']
            })
    
    st.write(f"  🔍 Checked {checked_count} candidates")
    st.write(f"  ✅ Found {len(candidates)} valid matches")
    
    if candidates:
        # Sort by compatibility then distance
        candidates.sort(key=lambda x: (-x['compatibility'], x['distance']))
        best = candidates[0]
        
        st.write(f"  🏆 **Best match**: {best['driver']} via Dog {best['nearby_dog_id']} ({best['distance']:.3f} miles, compatibility: {best['compatibility']})")
        
        return best
    else:
        st.write("  ❌ No compatible matches found")
        return None

def format_groups(groups):
    """Format groups list back to string format"""
    if len(groups) == 1:
        return str(groups[0])
    else:
        return '&'.join(map(str, sorted(groups)))

def reassign_dogs():
    """Main reassignment function with full debugging"""
    st.title("🐕 Dog Reassignment System - Debug Version")
    
    # Load data
    distance_matrix, map_data, driver_data = load_google_sheet_data()
    
    if distance_matrix is None or map_data is None or driver_data is None:
        st.error("Could not load data from Google Sheets.")
        return
    
    # Display data status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Distance Matrix", f"{len(distance_matrix)} x {len(distance_matrix.columns)}")
    with col2:
        st.metric("Dogs in Map", len(map_data))
    with col3:
        st.metric("Drivers", len(driver_data))
    
    # Check for automatic callouts
    auto_callouts = find_callout_drivers(driver_data)
    
    if auto_callouts:
        st.subheader("🚨 Automatic Callouts Detected:")
        for driver, groups in auto_callouts.items():
            st.error(f"**{driver}** called out from Groups: {', '.join(map(str, groups))}")
    
    if st.button("🔄 Process Drew's Callout", type="primary"):
        if 'Drew' not in auto_callouts:
            st.warning("Drew not found in callouts")
            return
        
        callout_groups = auto_callouts['Drew']
        
        # Find affected dogs
        current_assignments = map_data.copy()
        current_assignments['New Assignment'] = ''
        
        callout_pattern = 'Drew:'
        affected_dogs = current_assignments[current_assignments['Today'].str.contains(callout_pattern, na=False)]
        
        st.subheader(f"Dogs affected by Drew's callout:")
        st.dataframe(affected_dogs[['Dog Name', 'Dog ID', 'Today', 'Number of dogs']])
        
        if affected_dogs.empty:
            st.error("No affected dogs found!")
            return
        
        # Process reassignments with full debugging
        st.subheader("🔄 Processing Reassignments:")
        
        all_reassignments = []
        
        # Test with first 3 dogs to start
        test_dogs = affected_dogs.head(3)
        
        for _, dog in test_dogs.iterrows():
            original_driver, original_groups = parse_assignment(dog['Today'])
            
            # Check if affected by callout
            affected_groups = set(original_groups) & set(callout_groups)
            if not affected_groups:
                st.write(f"⏭️ Skipping Dog {dog['Dog ID']} - not affected by callout groups")
                continue
            
            # Find reassignment
            with st.expander(f"🔍 Debug Dog {dog['Dog ID']}: {dog['Dog Name']}", expanded=True):
                reassignment = find_best_reassignment(dog, current_assignments, distance_matrix)
                
                if reassignment:
                    new_assignment = f"{reassignment['driver']}:{format_groups(list(affected_groups))}"
                    
                    all_reassignments.append({
                        'Dog Name': dog['Dog Name'],
                        'Dog ID': dog['Dog ID'],
                        'Original Assignment': dog['Today'],
                        'New Assignment': new_assignment,
                        'Distance': f"{reassignment['distance']:.3f} miles",
                        'Via Dog': f"{reassignment['nearby_dog_id']} ({reassignment['nearby_dog_name']})"
                    })
                else:
                    all_reassignments.append({
                        'Dog Name': dog['Dog Name'],
                        'Dog ID': dog['Dog ID'],
                        'Original Assignment': dog['Today'],
                        'New Assignment': 'NO SOLUTION',
                        'Distance': 'N/A',
                        'Via Dog': 'N/A'
                    })
        
        # Display results
        st.subheader("📋 Test Results (First 3 Dogs):")
        if all_reassignments:
            results_df = pd.DataFrame(all_reassignments)
            st.dataframe(results_df, use_container_width=True)
            
            successful = len([r for r in all_reassignments if r['New Assignment'] != 'NO SOLUTION'])
            st.success(f"✅ Successfully matched {successful}/{len(all_reassignments)} test dogs!")
            
            if successful > 0:
                st.info("🎉 The logic is working! Ready to process all 20 dogs.")
        else:
            st.error("No reassignments processed")

if __name__ == "__main__":
    reassign_dogs()
