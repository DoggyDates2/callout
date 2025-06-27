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

def find_dog_in_matrix(dog_id, distance_matrix):
    """Find dog in matrix trying all possible data type combinations"""
    # Convert input to clean integer
    dog_id_clean = int(float(dog_id))
    
    # Try all possible formats in the matrix index
    possible_formats = [
        dog_id_clean,          # 777
        str(dog_id_clean),     # "777"
        float(dog_id_clean),   # 777.0
        f"{dog_id_clean}.0"    # "777.0"
    ]
    
    for format_attempt in possible_formats:
        if format_attempt in distance_matrix.index:
            return format_attempt
    
    return None

def get_dogs_within_distance(dog_id, distance_matrix, max_distance, dogs_going_today):
    """Find dogs within specified distance - HANDLES TEXT DISTANCES"""
    
    # Find the dog in matrix using flexible lookup
    matrix_dog_id = find_dog_in_matrix(dog_id, distance_matrix)
    
    if matrix_dog_id is None:
        return []
    
    # Get distances for this dog
    dog_distances = distance_matrix.loc[matrix_dog_id]
    
    nearby_dogs = []
    
    for other_matrix_id in distance_matrix.columns:
        # Skip self
        if str(other_matrix_id) == str(matrix_dog_id):
            continue
        
        try:
            distance_raw = dog_distances[other_matrix_id]
            
            # Convert distance to float (handles both number and text)
            if pd.isna(distance_raw):
                continue
            
            # Convert to float regardless of whether it's text or number
            distance = float(str(distance_raw))
            
            # Check if valid distance
            if distance > 0 and distance <= max_distance:
                # Convert other dog ID to integer for map lookup
                other_dog_id_clean = int(float(str(other_matrix_id)))
                
                # Check if this dog is in the going today list
                map_dog_ids = dogs_going_today['Dog ID'].astype(int)
                if other_dog_id_clean in map_dog_ids.values:
                    nearby_dogs.append((other_dog_id_clean, distance))
        
        except (ValueError, TypeError, KeyError):
            continue
    
    return sorted(nearby_dogs, key=lambda x: x[1])

def find_best_reassignment(dog_to_move, current_assignments, distance_matrix):
    """Find the best driver to reassign a dog to"""
    dog_id = dog_to_move['Dog ID']
    original_driver, original_groups = parse_assignment(dog_to_move['Today'])
    
    if not original_groups:
        return None
    
    # Get nearby dogs within 3 miles
    nearby_dogs = get_dogs_within_distance(dog_id, distance_matrix, 3.0, current_assignments)
    
    if not nearby_dogs:
        return None
    
    candidates = []
    
    # Check each nearby dog
    for nearby_dog_id, distance in nearby_dogs:
        # Find the nearby dog's data
        nearby_dog = current_assignments[current_assignments['Dog ID'] == nearby_dog_id]
        if nearby_dog.empty:
            continue
        
        nearby_dog = nearby_dog.iloc[0]
        candidate_driver, candidate_groups = parse_assignment(nearby_dog['Today'])
        
        # Must have a different driver than the callout driver
        if not candidate_driver or candidate_driver == original_driver:
            continue
        
        # Calculate group compatibility
        compatibility = 0
        for target_group in original_groups:
            for candidate_group in candidate_groups:
                if target_group == candidate_group:
                    compatibility += 1.0  # Same group
                elif abs(target_group - candidate_group) == 1:
                    compatibility += 0.5  # Adjacent group
        
        if compatibility > 0:
            candidates.append({
                'driver': candidate_driver,
                'groups': candidate_groups,
                'distance': distance,
                'compatibility': compatibility,
                'nearby_dog_id': nearby_dog_id
            })
    
    if candidates:
        # Sort by compatibility (desc) then distance (asc)
        candidates.sort(key=lambda x: (-x['compatibility'], x['distance']))
        return candidates[0]
    
    return None

def format_groups(groups):
    """Format groups list back to string format"""
    if len(groups) == 1:
        return str(groups[0])
    else:
        return '&'.join(map(str, sorted(groups)))

def reassign_dogs():
    """Main reassignment function"""
    st.title("🐕 Dog Reassignment System - Final Version")
    
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
    
    # Process Drew's callout automatically
    if 'Drew' in auto_callouts:
        if st.button("🔄 Process All Drew's Dogs", type="primary"):
            callout_groups = auto_callouts['Drew']
            
            # Find affected dogs
            current_assignments = map_data.copy()
            current_assignments['New Assignment'] = ''
            
            affected_dogs = current_assignments[current_assignments['Today'].str.contains('Drew:', na=False)]
            
            st.subheader(f"Dogs affected by Drew's callout ({len(affected_dogs)} dogs):")
            st.dataframe(affected_dogs[['Dog Name', 'Dog ID', 'Today', 'Number of dogs']])
            
            if affected_dogs.empty:
                st.error("No affected dogs found!")
                return
            
            # Process all reassignments
            all_reassignments = []
            progress_bar = st.progress(0)
            
            for i, (_, dog) in enumerate(affected_dogs.iterrows()):
                progress_bar.progress((i + 1) / len(affected_dogs))
                
                original_driver, original_groups = parse_assignment(dog['Today'])
                
                # Check if affected by callout
                affected_groups = set(original_groups) & set(callout_groups)
                if not affected_groups:
                    continue
                
                # Find reassignment
                reassignment = find_best_reassignment(dog, current_assignments, distance_matrix)
                
                if reassignment:
                    new_assignment = f"{reassignment['driver']}:{format_groups(list(affected_groups))}"
                    current_assignments.loc[current_assignments['Dog ID'] == dog['Dog ID'], 'New Assignment'] = new_assignment
                    
                    all_reassignments.append({
                        'Dog Name': dog['Dog Name'],
                        'Dog ID': dog['Dog ID'],
                        'Original Assignment': dog['Today'],
                        'New Assignment': new_assignment,
                        'Distance': f"{reassignment['distance']:.3f} miles",
                        'New Driver': reassignment['driver'],
                        'Compatibility': reassignment['compatibility']
                    })
                else:
                    all_reassignments.append({
                        'Dog Name': dog['Dog Name'],
                        'Dog ID': dog['Dog ID'],
                        'Original Assignment': dog['Today'],
                        'New Assignment': 'NO SOLUTION',
                        'Distance': 'N/A',
                        'New Driver': 'N/A',
                        'Compatibility': 0
                    })
            
            progress_bar.progress(1.0)
            
            # Display results
            st.subheader("📋 Final Reassignment Results:")
            if all_reassignments:
                results_df = pd.DataFrame(all_reassignments)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary stats
                successful = len([r for r in all_reassignments if r['New Assignment'] != 'NO SOLUTION'])
                failed = len(all_reassignments) - successful
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Dogs Affected", len(all_reassignments))
                with col2:
                    st.metric("Successfully Reassigned", successful, delta=f"+{successful}")
                with col3:
                    st.metric("No Solution Found", failed, delta=f"-{failed}" if failed > 0 else "0")
                
                if successful > 0:
                    st.success(f"🎉 Successfully reassigned {successful} out of {len(all_reassignments)} dogs!")
                    
                    # Download CSV
                    csv = current_assignments.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Updated Assignments (CSV)",
                        data=csv,
                        file_name="drew_reassignments.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("❌ No successful reassignments found")
            else:
                st.info("No reassignments processed")
    
    # Manual testing option
    st.subheader("Manual Testing")
    test_dog_id = st.number_input("Test with specific Dog ID:", value=777, step=1)
    
    if st.button("🧪 Test Single Dog"):
        test_dog = map_data[map_data['Dog ID'] == test_dog_id]
        if not test_dog.empty:
            test_dog = test_dog.iloc[0]
            st.write(f"Testing: {test_dog['Dog Name']} - {test_dog['Today']}")
            
            reassignment = find_best_reassignment(test_dog, map_data, distance_matrix)
            if reassignment:
                st.success(f"✅ Match found: {reassignment['driver']} ({reassignment['distance']:.3f} miles)")
            else:
                st.error("❌ No match found")
        else:
            st.error(f"Dog {test_dog_id} not found")

if __name__ == "__main__":
    reassign_dogs()
