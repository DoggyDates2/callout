import pandas as pd
import numpy as np
from collections import defaultdict
import streamlit as st

def load_google_sheet_data():
    """Load data from Google Sheets using public CSV links"""
    try:
        # Distance Matrix (separate spreadsheet)
        matrix_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSh-t6fIfgsli9D79KZeXM_-V5fO3zam6T_Bcp94d-IoRucxWusl6vbtT-WSaqFimHw7ABd76YGcKGV/pub?gid=0&single=true&output=csv"
        distance_matrix = pd.read_csv(matrix_url, index_col=0)
        
        # Map Sheet (gid=0 - first tab)
        map_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_9o2CJOgNpwtLPaqZkYBYVHtNJo5D-0qfeRqtdW-9yYV9cp5TMOvI5YTR8Xp3GcGhOU25mGBTHEdF/pub?gid=0&single=true&output=csv"
        map_data = pd.read_csv(map_url)
        
        # Driver Counts (same spreadsheet as Map, but different tab - gid=1359695250)
        driver_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_9o2CJOgNpwtLPaqZkYBYVHtNJo5D-0qfeRqtdW-9yYV9cp5TMOvI5YTR8Xp3GcGhOU25mGBTHEdF/pub?gid=1359695250&single=true&output=csv"
        driver_data = pd.read_csv(driver_url)
        
        return distance_matrix, map_data, driver_data
        
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        st.info("Make sure your Google Sheets are set to 'Anyone with the link can view'")
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
    
    # Get column names - they might be different than expected
    cols = driver_data.columns.tolist()
    
    for _, row in driver_data.iterrows():
        # Column A should be driver name - find the right column
        driver_col = cols[0]  # First column
        driver = row[driver_col]
        
        callout_groups = []
        
        # Look for columns D, E, F (indices 3, 4, 5) or by name
        if len(cols) > 3 and (row[cols[3]] == 'X' or str(row[cols[3]]).upper() == 'X'):
            callout_groups.append(1)
        if len(cols) > 4 and (row[cols[4]] == 'X' or str(row[cols[4]]).upper() == 'X'):
            callout_groups.append(2)
        if len(cols) > 5 and (row[cols[5]] == 'X' or str(row[cols[5]]).upper() == 'X'):
            callout_groups.append(3)
        
        if callout_groups:
            callouts[driver] = callout_groups
    
    return callouts

def load_driver_capacities(driver_data):
    """Load driver capacity data into a usable format"""
    capacities = {}
    
    if driver_data is None or driver_data.empty:
        return capacities
    
    cols = driver_data.columns.tolist()
    
    for _, row in driver_data.iterrows():
        driver = row[cols[0]]  # First column is driver name
        
        driver_caps = {}
        # Columns D, E, F are indices 3, 4, 5
        if len(cols) > 3 and pd.notna(row[cols[3]]) and str(row[cols[3]]).upper() != 'X':
            try:
                driver_caps['Group 1'] = int(row[cols[3]])
            except:
                pass
        
        if len(cols) > 4 and pd.notna(row[cols[4]]) and str(row[cols[4]]).upper() != 'X':
            try:
                driver_caps['Group 2'] = int(row[cols[4]])
            except:
                pass
        
        if len(cols) > 5 and pd.notna(row[cols[5]]) and str(row[cols[5]]).upper() != 'X':
            try:
                driver_caps['Group 3'] = int(row[cols[5]])
            except:
                pass
        
        if driver_caps:
            capacities[driver] = driver_caps
    
    return capacities

def get_dogs_going_today(map_data):
    """Get list of dogs going today based on map data"""
    # Dogs going today are those with non-empty assignments in 'Today' column
    today_dogs = map_data[map_data['Today'].notna() & (map_data['Today'] != '')]
    return today_dogs

def get_dogs_within_distance(dog_id, distance_matrix, max_distance, dogs_going_today):
    """Find dogs within specified distance of target dog"""
    if str(dog_id) not in distance_matrix.columns:
        return []
    
    nearby_dogs = []
    for other_dog_id in distance_matrix.columns:
        if other_dog_id == str(dog_id):
            continue
        
        try:
            distance = distance_matrix.loc[str(dog_id), str(other_dog_id)]
            if 0 < distance <= max_distance:
                # Check if this dog is going today
                if int(other_dog_id) in dogs_going_today['Dog ID'].values:
                    nearby_dogs.append((int(other_dog_id), distance))
        except:
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

def count_current_dogs_in_groups(driver, groups, current_assignments):
    """Count how many dogs a driver currently has in specified groups"""
    count = 0
    for _, row in current_assignments.iterrows():
        assigned_driver, assigned_groups = parse_assignment(row['Today'])
        if assigned_driver == driver:
            # Count overlap between assigned groups and target groups
            overlap = set(assigned_groups) & set(groups)
            if overlap:
                # Multiply by number of dogs at this location
                num_dogs = row['Number of dogs'] if pd.notna(row['Number of dogs']) else 1
                count += num_dogs
    
    return count

def find_best_reassignment(dog_to_move, current_assignments, distance_matrix, driver_capacities):
    """Find the best driver to reassign a dog to"""
    dog_id = dog_to_move['Dog ID']
    original_driver, original_groups = parse_assignment(dog_to_move['Today'])
    num_dogs = dog_to_move['Number of dogs'] if pd.notna(dog_to_move['Number of dogs']) else 1
    
    dogs_going_today = current_assignments
    
    # Start with base distance thresholds
    group_1_threshold = 0.5
    group_2_threshold = 0.25
    max_iterations = 5
    
    for iteration in range(max_iterations):
        # Adjust thresholds based on iteration
        current_g1_threshold = group_1_threshold + (iteration * 0.5)
        current_g2_threshold = group_2_threshold + (iteration * 0.25)
        
        # Get nearby dogs
        max_threshold = max(current_g1_threshold, current_g2_threshold)
        nearby_dogs = get_dogs_within_distance(dog_id, distance_matrix, max_threshold, dogs_going_today)
        
        candidates = []
        
        for nearby_dog_id, distance in nearby_dogs:
            nearby_dog = dogs_going_today[dogs_going_today['Dog ID'] == nearby_dog_id]
            if nearby_dog.empty:
                continue
            
            nearby_dog = nearby_dog.iloc[0]
            candidate_driver, candidate_groups = parse_assignment(nearby_dog['Today'])
            
            if not candidate_driver or candidate_driver == original_driver:
                continue
            
            # Check distance threshold based on groups
            threshold_met = False
            for group in original_groups:
                if group == 1 and distance <= current_g1_threshold:
                    threshold_met = True
                elif group == 2 and distance <= current_g2_threshold:
                    threshold_met = True
                elif group == 3 and distance <= current_g1_threshold:  # Group 3 uses same threshold as Group 1
                    threshold_met = True
            
            if not threshold_met:
                continue
            
            # Calculate compatibility weight
            compatibility = calculate_group_compatibility_weight(original_groups, candidate_groups)
            if compatibility == 0:
                continue
            
            # Count current dogs for this driver in relevant groups
            current_dog_count = count_current_dogs_in_groups(candidate_driver, original_groups, current_assignments)
            
            # Check capacity constraints
            capacity_ok = True
            if driver_capacities and candidate_driver in driver_capacities:
                for group in original_groups:
                    max_capacity = driver_capacities[candidate_driver].get(f'Group {group}', float('inf'))
                    if current_dog_count + num_dogs > max_capacity:
                        capacity_ok = False
                        break
            
            if capacity_ok:
                candidates.append({
                    'driver': candidate_driver,
                    'groups': candidate_groups,
                    'distance': distance,
                    'compatibility': compatibility,
                    'current_count': current_dog_count,
                    'nearby_dog_id': nearby_dog_id
                })
        
        if candidates:
            # Sort by: compatibility (desc), current count (asc), distance (asc)
            candidates.sort(key=lambda x: (-x['compatibility'], x['current_count'], x['distance']))
            return candidates[0]
    
    return None

def handle_capacity_overflow(driver, groups, current_assignments, distance_matrix, driver_capacities, max_iterations=5):
    """Handle when a driver exceeds capacity by moving their dogs to other drivers"""
    moved_dogs = []
    
    for iteration in range(max_iterations):
        # Check if still over capacity
        current_count = count_current_dogs_in_groups(driver, groups, current_assignments)
        
        over_capacity = False
        if driver_capacities and driver in driver_capacities:
            for group in groups:
                max_capacity = driver_capacities[driver].get(f'Group {group}', float('inf'))
                if current_count > max_capacity:
                    over_capacity = True
                    break
        
        if not over_capacity:
            break
        
        # Find a dog to move from this driver
        driver_dogs = current_assignments[
            current_assignments['Today'].str.contains(f'{driver}:', na=False)
        ]
        
        if driver_dogs.empty:
            break
        
        # Try to move the first dog that can be moved
        for _, dog in driver_dogs.iterrows():
            if dog['Dog ID'] in [d['dog_id'] for d in moved_dogs]:
                continue  # Already moved this dog
            
            reassignment = find_best_reassignment(dog, current_assignments, distance_matrix, driver_capacities)
            if reassignment:
                # Update assignment
                new_assignment = f"{reassignment['driver']}:{format_groups(groups)}"
                current_assignments.loc[current_assignments['Dog ID'] == dog['Dog ID'], 'New Assignment'] = new_assignment
                
                moved_dogs.append({
                    'dog_id': dog['Dog ID'],
                    'old_driver': driver,
                    'new_driver': reassignment['driver'],
                    'groups': groups
                })
                break
    
    return moved_dogs

def format_groups(groups):
    """Format groups list back to string format (e.g., [1,2] -> '1&2')"""
    if len(groups) == 1:
        return str(groups[0])
    else:
        return '&'.join(map(str, sorted(groups)))

def reassign_dogs():
    """Main reassignment function"""
    st.title("🐕 Dog Reassignment System")
    
    # Load data
    distance_matrix, map_data, driver_data = load_google_sheet_data()
    
    if distance_matrix is None or map_data is None:
        st.error("Could not load data from Google Sheets. Please check the URLs and sheet permissions.")
        return
    
    if driver_data is None:
        st.warning("Could not load driver data. Manual callout selection will be used.")
    
    # Display data status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Distance Matrix", f"{len(distance_matrix)} x {len(distance_matrix.columns)}")
    with col2:
        st.metric("Dogs in Map", len(map_data))
    with col3:
        st.metric("Drivers", len(driver_data) if driver_data is not None else "N/A")
    
    # Check for automatic callouts first
    auto_callouts = find_callout_drivers(driver_data) if driver_data is not None else {}
    
    if auto_callouts:
        st.subheader("🚨 Automatic Callouts Detected from Driver Sheet:")
        for driver, groups in auto_callouts.items():
            st.error(f"**{driver}** called out from Groups: {', '.join(map(str, groups))}")
    
    # Manual callout simulation for testing
    st.subheader("Additional Manual Callout Simulation (for testing)")
    callout_driver = st.selectbox("Select additional driver who called out:", ["None"] + sorted(map_data['Today'].str.split(':').str[0].dropna().unique()))
    callout_groups = st.multiselect("Select groups they called out from:", [1, 2, 3])
    
    if st.button("🔄 Process Reassignments", type="primary"):
        # Combine automatic and manual callouts
        all_callouts = auto_callouts.copy()
        
        if callout_driver != "None" and callout_groups:
            if callout_driver in all_callouts:
                # Merge with existing callout
                all_callouts[callout_driver] = list(set(all_callouts[callout_driver] + callout_groups))
            else:
                all_callouts[callout_driver] = callout_groups
        
        if not all_callouts:
            st.warning("No callouts detected or selected")
            return
        
        st.subheader("Processing Callouts:")
        for driver, groups in all_callouts.items():
            st.write(f"• **{driver}**: Groups {groups}")
        
        # Load driver capacities
        driver_capacities = load_driver_capacities(driver_data)
        
        # Get current assignments
        current_assignments = map_data.copy()
        current_assignments['New Assignment'] = ''
        
        all_reassignments = []
        
        # Process each driver's callout
        for callout_driver, callout_groups in all_callouts.items():
            # Find dogs assigned to the calling-out driver
            callout_pattern = f'{callout_driver}:'
            affected_dogs = current_assignments[
                current_assignments['Today'].str.contains(callout_pattern, na=False)
            ]
            
            st.subheader(f"Dogs affected by {callout_driver}'s callout:")
            if not affected_dogs.empty:
                st.dataframe(affected_dogs[['Dog Name', 'Dog ID', 'Today', 'Number of dogs']])
            else:
                st.info(f"No dogs currently assigned to {callout_driver}")
                continue
            
            # Process reassignments for this driver
            reassignments = []
            
            with st.spinner(f"Finding reassignments for {callout_driver}'s dogs..."):
                for _, dog in affected_dogs.iterrows():
                    original_driver, original_groups = parse_assignment(dog['Today'])
                    
                    # Check if this dog is affected by the callout
                    affected_groups = set(original_groups) & set(callout_groups)
                    if not affected_groups:
                        continue
                    
                    # Find best reassignment
                    reassignment = find_best_reassignment(dog, current_assignments, distance_matrix, driver_capacities)
                    
                    if reassignment:
                        new_assignment = f"{reassignment['driver']}:{format_groups(list(affected_groups))}"
                        current_assignments.loc[current_assignments['Dog ID'] == dog['Dog ID'], 'New Assignment'] = new_assignment
                        
                        reassignments.append({
                            'Dog Name': dog['Dog Name'],
                            'Dog ID': dog['Dog ID'],
                            'Original Assignment': dog['Today'],
                            'New Assignment': new_assignment,
                            'Distance to Nearby Dog': f"{reassignment['distance']:.2f} miles",
                            'Reason': f"Moved to {reassignment['driver']} (nearest dog: {reassignment['nearby_dog_id']})"
                        })
                    else:
                        reassignments.append({
                            'Dog Name': dog['Dog Name'],
                            'Dog ID': dog['Dog ID'],
                            'Original Assignment': dog['Today'],
                            'New Assignment': 'NO SOLUTION FOUND',
                            'Distance to Nearby Dog': 'N/A',
                            'Reason': 'No suitable driver found within distance thresholds'
                        })
            
            all_reassignments.extend(reassignments)
        
        # Display results
        st.subheader("📋 Final Reassignment Results:")
        if all_reassignments:
            results_df = pd.DataFrame(all_reassignments)
            st.dataframe(results_df, use_container_width=True)
            
            # Show updated assignments
            st.subheader("📊 Updated Map Data (Column K - New Assignment):")
            display_data = current_assignments[current_assignments['New Assignment'] != ''][
                ['Dog Name', 'Dog ID', 'Today', 'New Assignment']
            ]
            if not display_data.empty:
                st.dataframe(display_data, use_container_width=True)
            else:
                st.info("No new assignments were made.")
            
            # Option to download results
            csv = current_assignments.to_csv(index=False)
            st.download_button(
                label="📥 Download Updated Assignments as CSV",
                data=csv,
                file_name="updated_dog_assignments.csv",
                mime="text/csv"
            )
            
            # Summary stats
            successful_reassignments = len([r for r in all_reassignments if r['New Assignment'] != 'NO SOLUTION FOUND'])
            failed_reassignments = len(all_reassignments) - successful_reassignments
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Dogs Affected", len(all_reassignments))
            with col2:
                st.metric("Successfully Reassigned", successful_reassignments)
            with col3:
                st.metric("No Solution Found", failed_reassignments)
                
        else:
            st.info("No dogs needed reassignment based on the detected callouts.")

# Streamlit app entry point
if __name__ == "__main__":
    reassign_dogs()
