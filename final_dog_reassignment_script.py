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
    """Find dogs within specified distance - FIXED for integer matrix"""
    
    # Convert to integer for matrix lookup (matrix expects int64)
    dog_id_int = int(float(dog_id))
    
    # Check if dog exists in matrix
    if dog_id_int not in distance_matrix.index:
        return []
    
    # Get distances for this dog (this will be a pandas Series)
    dog_distances = distance_matrix.loc[dog_id_int]
    
    nearby_dogs = []
    
    # Iterate through all distances
    for other_dog_id_int, distance in dog_distances.items():
        # Skip self
        if other_dog_id_int == dog_id_int:
            continue
        
        try:
            # Convert distance to float
            distance_float = float(distance)
            
            # Check if valid distance
            if distance_float > 0 and distance_float <= max_distance:
                # Check if this dog is in the going today list
                # Convert map Dog IDs to int for comparison
                map_dog_ids = dogs_going_today['Dog ID'].astype(int)
                if other_dog_id_int in map_dog_ids.values:
                    nearby_dogs.append((other_dog_id_int, distance_float))
        
        except (ValueError, TypeError):
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
    st.title("🐕 Dog Reassignment System - Working Version")
    
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
    
    # Quick test first
    st.subheader("🧪 Quick Test")
    if st.button("Test Dog 777", type="secondary"):
        test_dog = map_data[map_data['Dog ID'] == 777].iloc[0]
        st.write(f"Testing: {test_dog['Dog Name']} - {test_dog['Today']}")
        
        # Test matrix lookup
        dog_id_int = int(float(777))
        if dog_id_int in distance_matrix.index:
            st.success("✅ Dog 777 found in matrix!")
            
            # Get nearby dogs
            nearby = get_dogs_within_distance(777, distance_matrix, 3.0, map_data)
            st.write(f"Found {len(nearby)} nearby dogs within 3 miles")
            
            if nearby:
                st.write("Sample nearby dogs:")
                for dog_id, dist in nearby[:5]:
                    nearby_info = map_data[map_data['Dog ID'] == dog_id]
                    if not nearby_info.empty:
                        assignment = nearby_info.iloc[0]['Today']
                        st.write(f"- Dog {dog_id}: {dist:.3f} miles - {assignment}")
                
                # Test full reassignment
                reassignment = find_best_reassignment(test_dog, map_data, distance_matrix)
                if reassignment:
                    st.success(f"✅ MATCH FOUND: {reassignment['driver']} ({reassignment['distance']:.3f} miles)")
                else:
                    st.error("❌ No compatible match found")
        else:
            st.error("❌ Dog 777 not found in matrix")
    
    # Process Drew's callout
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
            status_text = st.empty()
            
            for i, (_, dog) in enumerate(affected_dogs.iterrows()):
                status_text.text(f"Processing dog {i+1}/{len(affected_dogs)}: {dog['Dog Name']}")
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
                        'Via Dog': reassignment['nearby_dog_id']
                    })
                else:
                    all_reassignments.append({
                        'Dog Name': dog['Dog Name'],
                        'Dog ID': dog['Dog ID'],
                        'Original Assignment': dog['Today'],
                        'New Assignment': 'NO SOLUTION',
                        'Distance': 'N/A',
                        'New Driver': 'N/A',
                        'Via Dog': 'N/A'
                    })
            
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
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
                    st.metric("No Solution Found", failed)
                
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
                    st.error("❌ No successful reassignments found - may need to adjust distance thresholds")
            else:
                st.info("No reassignments processed")

if __name__ == "__main__":
    reassign_dogs()
