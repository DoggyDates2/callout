import pandas as pd
import numpy as np
from collections import defaultdict
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
    """Find dogs within specified distance of target dog - FIXED VERSION"""
    # Convert dog_id to string for matrix lookup
    dog_id_str = str(dog_id)
    
    # Check if dog exists in matrix (try both string and int formats)
    if dog_id_str not in distance_matrix.index:
        if dog_id not in distance_matrix.index:
            return []
    
    nearby_dogs = []
    
    # Iterate through all columns in the distance matrix
    for other_dog_id_str in distance_matrix.columns:
        if other_dog_id_str == dog_id_str:
            continue
        
        try:
            # Get distance using proper indexing
            distance = distance_matrix.loc[dog_id_str, other_dog_id_str]
            
            # Convert to float if it's not already
            if pd.notna(distance):
                distance = float(distance)
                
                # Check if distance is valid and within threshold
                if 0 < distance <= max_distance:
                    # Convert other_dog_id back to int for checking against map data
                    try:
                        other_dog_id_int = int(other_dog_id_str)
                        
                        # Check if this dog is going today
                        if other_dog_id_int in dogs_going_today['Dog ID'].values:
                            nearby_dogs.append((other_dog_id_int, distance))
                    except ValueError:
                        # Skip if can't convert to int
                        continue
        except (KeyError, ValueError, TypeError):
            continue
    
    return sorted(nearby_dogs, key=lambda x: x[1])  # Sort by distance

def calculate_group_compatibility_weight(target_groups, candidate_groups):
    """Calculate compatibility weight between target and candidate groups"""
    weight = 0
    
    for target_group in target_groups:
        for candidate_group in candidate_groups:
            if target_group == candidate_group:
                weight += 1.0  # Same group = full weight
            elif abs(target_group - candidate_group) == 1:
                weight += 0.5  # Adjacent group = half weight
    
    return weight

def find_best_reassignment_simplified(dog_to_move, current_assignments, distance_matrix):
    """Simplified reassignment logic with better debugging"""
    dog_id = dog_to_move['Dog ID']
    original_driver, original_groups = parse_assignment(dog_to_move['Today'])
    
    if not original_groups:
        return None
    
    # More generous distance thresholds
    max_distance = 2.0  # Increased from 0.5/0.25
    
    # Get nearby dogs
    nearby_dogs = get_dogs_within_distance(dog_id, distance_matrix, max_distance, current_assignments)
    
    if not nearby_dogs:
        return None
    
    candidates = []
    
    for nearby_dog_id, distance in nearby_dogs:
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
        compatibility = calculate_group_compatibility_weight(original_groups, candidate_groups)
        if compatibility == 0:
            continue
        
        candidates.append({
            'driver': candidate_driver,
            'groups': candidate_groups,
            'distance': distance,
            'compatibility': compatibility,
            'nearby_dog_id': nearby_dog_id
        })
    
    if candidates:
        # Sort by: compatibility (desc), distance (asc)
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
    """Main reassignment function with simplified logic"""
    st.title("🐕 Dog Reassignment System - Fixed Version")
    
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
    
    # Manual testing
    st.subheader("Manual Testing")
    available_drivers = sorted([d for d in map_data['Today'].str.split(':').str[0].dropna().unique() if d])
    callout_driver = st.selectbox("Select driver who called out:", ["None"] + available_drivers)
    callout_groups = st.multiselect("Select groups:", [1, 2, 3])
    
    if st.button("🔄 Process Reassignments", type="primary"):
        # Combine automatic and manual callouts
        all_callouts = auto_callouts.copy()
        
        if callout_driver != "None" and callout_groups:
            all_callouts[callout_driver] = callout_groups
        
        if not all_callouts:
            st.warning("No callouts selected")
            return
        
        # Process reassignments
        current_assignments = map_data.copy()
        current_assignments['New Assignment'] = ''
        all_reassignments = []
        
        for callout_driver_name, callout_groups_list in all_callouts.items():
            # Find affected dogs
            affected_dogs = current_assignments[
                current_assignments['Today'].str.contains(f'{callout_driver_name}:', na=False)
            ]
            
            st.subheader(f"Dogs affected by {callout_driver_name}'s callout:")
            if not affected_dogs.empty:
                st.dataframe(affected_dogs[['Dog Name', 'Dog ID', 'Today', 'Number of dogs']])
                
                # Process each dog
                with st.spinner(f"Finding reassignments..."):
                    for _, dog in affected_dogs.iterrows():
                        original_driver, original_groups = parse_assignment(dog['Today'])
                        
                        # Check if affected by callout
                        affected_groups = set(original_groups) & set(callout_groups_list)
                        if not affected_groups:
                            continue
                        
                        # Find reassignment
                        reassignment = find_best_reassignment_simplified(dog, current_assignments, distance_matrix)
                        
                        if reassignment:
                            new_assignment = f"{reassignment['driver']}:{format_groups(list(affected_groups))}"
                            current_assignments.loc[current_assignments['Dog ID'] == dog['Dog ID'], 'New Assignment'] = new_assignment
                            
                            all_reassignments.append({
                                'Dog Name': dog['Dog Name'],
                                'Dog ID': dog['Dog ID'],
                                'Original Assignment': dog['Today'],
                                'New Assignment': new_assignment,
                                'Distance': f"{reassignment['distance']:.3f} miles",
                                'Compatibility': reassignment['compatibility'],
                                'Via Dog': reassignment['nearby_dog_id']
                            })
                        else:
                            all_reassignments.append({
                                'Dog Name': dog['Dog Name'],
                                'Dog ID': dog['Dog ID'],
                                'Original Assignment': dog['Today'],
                                'New Assignment': 'NO SOLUTION',
                                'Distance': 'N/A',
                                'Compatibility': 0,
                                'Via Dog': 'N/A'
                            })
            else:
                st.info(f"No dogs assigned to {callout_driver_name}")
        
        # Display results
        st.subheader("📋 Reassignment Results:")
        if all_reassignments:
            results_df = pd.DataFrame(all_reassignments)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary
            successful = len([r for r in all_reassignments if r['New Assignment'] != 'NO SOLUTION'])
            failed = len(all_reassignments) - successful
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Dogs", len(all_reassignments))
            with col2:
                st.metric("Successfully Reassigned", successful, delta=f"+{successful}")
            with col3:
                st.metric("No Solution Found", failed, delta=f"-{failed}" if failed > 0 else "0")
            
            # Download option
            if successful > 0:
                csv = current_assignments.to_csv(index=False)
                st.download_button(
                    label="📥 Download Updated Assignments",
                    data=csv,
                    file_name="updated_dog_assignments.csv",
                    mime="text/csv"
                )
        else:
            st.info("No reassignments needed.")

if __name__ == "__main__":
    reassign_dogs()
