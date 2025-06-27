import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import requests
from io import StringIO
import re

class DogReassignmentSystem:
    def __init__(self):
        self.distance_matrix = {}
        self.dogs_going_today = {}
        self.driver_capacities = {}
        self.driver_callouts = {}
        self.reassignments = []
        
    def load_distance_matrix(self, source):
        """Load distance matrix from file upload or Google Sheets URL"""
        try:
            if isinstance(source, str):  # URL
                # Convert Google Sheets URL to CSV export
                if 'edit' in source:
                    sheet_id = source.split('/d/')[1].split('/')[0]
                    gid = source.split('gid=')[1].split('#')[0] if 'gid=' in source else '0'
                    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
                
                response = requests.get(csv_url)
                df = pd.read_csv(StringIO(response.text))
                st.info("📡 Loaded from Google Sheets URL")
            else:  # File upload
                df = pd.read_csv(source)
                st.info("📁 Loaded from uploaded file")
            
            st.info(f"🔍 Distance Matrix Shape: {df.shape}")
            
            # YOUR EXACT FORMAT: First column has row Dog IDs, other columns have column Dog IDs
            # Get column Dog IDs (skip first column which is ":" or similar)
            column_dog_ids = []
            for col in df.columns[1:]:  # Skip first column
                if pd.notna(col) and str(col).strip() != '':
                    clean_id = str(col).strip()
                    column_dog_ids.append(clean_id)
            
            st.info(f"🔍 Found {len(column_dog_ids)} column Dog IDs")
            
            # Process each row to build distance matrix
            rows_processed = 0
            for idx, row in df.iterrows():
                # Get row Dog ID from first column (whatever it's named)
                row_dog_id = row.iloc[0]  # First column by position
                
                if pd.isna(row_dog_id) or str(row_dog_id).strip() == '':
                    continue
                    
                row_dog_id = str(row_dog_id).strip()
                
                # Skip if this doesn't look like a Dog ID
                if not row_dog_id or 'x' not in row_dog_id.lower():
                    continue
                
                # Initialize this dog's distances
                self.distance_matrix[row_dog_id] = {}
                
                # Get distances for each column Dog ID
                for col_idx, col_dog_id in enumerate(column_dog_ids):
                    try:
                        # Get distance value from corresponding column (skip first column)
                        distance_val = row.iloc[col_idx + 1]
                        
                        if pd.isna(distance_val):
                            distance = 0.0
                        else:
                            distance = float(str(distance_val).strip())
                    except (ValueError, TypeError):
                        distance = 0.0
                    
                    self.distance_matrix[row_dog_id][col_dog_id] = distance
                
                rows_processed += 1
                
                # Progress for large files
                if rows_processed % 200 == 0:
                    st.write(f"  Processed {rows_processed} dogs...")
            
            st.success(f"✅ Loaded distance matrix for {len(self.distance_matrix)} dogs")
            
            # Show sample
            if len(self.distance_matrix) > 0:
                first_dog = list(self.distance_matrix.keys())[0]
                sample_distances = list(self.distance_matrix[first_dog].items())[:3]
                st.info(f"📏 Sample distances for {first_dog}: {sample_distances}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading distance matrix: {e}")
            import traceback
            st.error(f"🔍 Full traceback: {traceback.format_exc()}")
            return False
    
    def load_dog_assignments(self, source):
        """Load dog assignments from file upload or Google Sheets URL"""
        try:
            if isinstance(source, str):  # URL
                # Convert Google Sheets URL to CSV export  
                if 'edit' in source:
                    sheet_id = source.split('/d/')[1].split('/')[0]
                    gid = source.split('gid=')[1].split('#')[0] if 'gid=' in source else '267803750'
                    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
                
                response = requests.get(csv_url)
                df = pd.read_csv(StringIO(response.text))
                st.info("📡 Loaded from Google Sheets URL")
            else:  # File upload
                df = pd.read_csv(source)
                st.info("📁 Loaded from uploaded file")
            
            st.info(f"🔍 Map Sheet Shape: {df.shape}")
            st.info(f"🔍 Map Sheet Columns: {list(df.columns)}")
            
            assignments_found = 0
            for _, row in df.iterrows():
                # Get Dog ID - exact column name from your file
                dog_id = row.get('Dog ID')
                if pd.isna(dog_id) or str(dog_id).strip() == '':
                    continue
                dog_id = str(dog_id).strip()
                
                # Get Today assignment - exact column name from your file
                assignment = row.get('Today')
                if pd.isna(assignment) or str(assignment).strip() == '':
                    continue
                assignment = str(assignment).strip()
                
                # Skip if no valid assignment or if it's not a driver assignment
                if not assignment or ':' not in assignment:
                    continue
                
                # Get number of dogs (exact column name from your file)
                num_dogs = 1
                if 'Number of dogs' in row and not pd.isna(row['Number of dogs']):
                    try:
                        num_dogs = int(float(row['Number of dogs']))
                    except (ValueError, TypeError):
                        num_dogs = 1
                
                # Store assignment
                self.dogs_going_today[dog_id] = {
                    'assignment': assignment,
                    'num_dogs': num_dogs,
                    'address': str(row.get('Address', '')),
                    'dog_name': str(row.get('Dog Name', ''))
                }
                assignments_found += 1
            
            st.success(f"✅ Loaded assignments for {assignments_found} dogs going today")
            
            # Show sample assignments
            if assignments_found > 0:
                sample_assignments = []
                for dog_id, info in list(self.dogs_going_today.items())[:5]:
                    sample_assignments.append(f"{dog_id}:{info['assignment']}")
                st.info(f"📋 Sample assignments: {sample_assignments}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading dog assignments: {e}")
            return False
    
    def load_driver_capacities(self, source):
        """Load driver capacities from file upload or Google Sheets URL"""
        try:
            if isinstance(source, str):  # URL
                # Convert Google Sheets URL to CSV export
                if 'edit' in source:
                    sheet_id = source.split('/d/')[1].split('/')[0]
                    gid = source.split('gid=')[1].split('#')[0] if 'gid=' in source else '1359695250'
                    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
                
                response = requests.get(csv_url)
                df = pd.read_csv(StringIO(response.text))
                st.info("📡 Loaded from Google Sheets URL")
            else:  # File upload
                df = pd.read_csv(source)
                st.info("📁 Loaded from uploaded file")
            
            st.info(f"🔍 Driver Sheet Shape: {df.shape}")
            st.info(f"🔍 Driver Sheet Columns: {list(df.columns)}")
            
            drivers_loaded = 0
            callout_drivers = []
            
            for _, row in df.iterrows():
                # Get driver name - exact column name from your file
                driver = row.get('Driver')
                if pd.isna(driver) or str(driver).strip() == '':
                    continue
                driver = str(driver).strip()
                
                # Get group assignments/callouts - exact column names from your file
                group1 = str(row.get('Group 1', '')).strip().upper()
                group2 = str(row.get('Group 2', '')).strip().upper()
                group3 = str(row.get('Group 3', '')).strip().upper()
                
                # Track callouts (X means calling out)
                self.driver_callouts[driver] = {
                    'group1': group1 == 'X',
                    'group2': group2 == 'X',
                    'group3': group3 == 'X'
                }
                
                # Parse capacities (X = called out, number = capacity, empty = default 9)
                def parse_capacity(val):
                    if val == 'X' or val == '':
                        return 9  # Default capacity when called out or empty
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return 9
                
                self.driver_capacities[driver] = {
                    'group1': parse_capacity(group1),
                    'group2': parse_capacity(group2), 
                    'group3': parse_capacity(group3)
                }
                
                drivers_loaded += 1
                
                # Track callouts for display
                if any([group1 == 'X', group2 == 'X', group3 == 'X']):
                    callout_details = []
                    if group1 == 'X': callout_details.append("Group 1")
                    if group2 == 'X': callout_details.append("Group 2")
                    if group3 == 'X': callout_details.append("Group 3")
                    callout_drivers.append(f"{driver} ({', '.join(callout_details)})")
            
            # Show results
            if callout_drivers:
                st.warning(f"🚨 Drivers calling out: {', '.join(callout_drivers)}")
            else:
                st.info("ℹ️ No drivers calling out today")
            
            st.success(f"✅ Loaded capacities for {drivers_loaded} drivers")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading driver capacities: {e}")
            return False
    
    def parse_group_assignment(self, assignment):
        """Parse group assignment like 'Andy:1&2' into driver and groups"""
        if ':' not in assignment:
            return None, []
        
        parts = assignment.split(':')
        driver = parts[0].strip()
        group_part = parts[1].strip()
        
        # Handle special cases like "XX" - ignore them
        if 'XX' in group_part.upper():
            return None, []
        
        # Parse groups (1, 2, 3, 1&2, 2&3, 1&2&3, etc.)
        groups = []
        if '&' in group_part:
            for g in group_part.split('&'):
                try:
                    group_num = int(g.strip())
                    if 1 <= group_num <= 3:
                        groups.append(group_num)
                except ValueError:
                    continue
        else:
            # Handle single group or extract numbers
            import re
            numbers = re.findall(r'\d+', group_part)
            for num_str in numbers:
                try:
                    group_num = int(num_str)
                    if 1 <= group_num <= 3:
                        groups.append(group_num)
                except ValueError:
                    continue
        
        return driver, sorted(list(set(groups)))  # Remove duplicates and sort
    
    def get_dogs_to_reassign(self):
        """Get list of dogs that need to be reassigned due to driver callouts"""
        dogs_to_reassign = []
        
        for dog_id, dog_info in self.dogs_going_today.items():
            assignment = dog_info['assignment']
            driver, groups = self.parse_group_assignment(assignment)
            
            if not driver or not groups:
                continue
            
            # Check if this driver is calling out for any of the needed groups
            if driver in self.driver_callouts:
                callout_info = self.driver_callouts[driver]
                needs_reassignment = False
                affected_groups = []
                
                for group in groups:
                    if group == 1 and callout_info['group1']:
                        needs_reassignment = True
                        affected_groups.append(1)
                    elif group == 2 and callout_info['group2']:
                        needs_reassignment = True
                        affected_groups.append(2)
                    elif group == 3 and callout_info['group3']:
                        needs_reassignment = True
                        affected_groups.append(3)
                
                if needs_reassignment:
                    dogs_to_reassign.append({
                        'dog_id': dog_id,
                        'original_driver': driver,
                        'original_groups': groups,
                        'affected_groups': affected_groups,
                        'dog_info': dog_info
                    })
        
        return dogs_to_reassign
    
    def get_nearby_dogs(self, dog_id, max_distance=3.0):
        """Get dogs within max_distance miles of the given dog"""
        if dog_id not in self.distance_matrix:
            return []
        
        nearby = []
        for other_dog_id, distance in self.distance_matrix[dog_id].items():
            if distance > 0 and distance <= max_distance:
                nearby.append({
                    'dog_id': other_dog_id,
                    'distance': distance
                })
        
        return sorted(nearby, key=lambda x: x['distance'])
    
    def get_current_driver_loads(self):
        """Calculate current number of dogs assigned to each driver by group"""
        driver_loads = {}
        
        for dog_id, dog_info in self.dogs_going_today.items():
            assignment = dog_info['assignment']
            driver, groups = self.parse_group_assignment(assignment)
            
            if not driver or not groups:
                continue
            
            if driver not in driver_loads:
                driver_loads[driver] = {'group1': 0, 'group2': 0, 'group3': 0}
            
            num_dogs = dog_info['num_dogs']
            for group in groups:
                if group == 1:
                    driver_loads[driver]['group1'] += num_dogs
                elif group == 2:
                    driver_loads[driver]['group2'] += num_dogs
                elif group == 3:
                    driver_loads[driver]['group3'] += num_dogs
        
        return driver_loads
    
    def find_best_reassignment(self, dog_to_reassign, iteration=0):
        """Find the best driver to reassign a dog to"""
        dog_id = dog_to_reassign['dog_id']
        original_groups = dog_to_reassign['original_groups']
        num_dogs = dog_to_reassign['dog_info']['num_dogs']
        dog_name = dog_to_reassign['dog_info']['dog_name']
        
        # Get nearby dogs
        nearby_dogs = self.get_nearby_dogs(dog_id, max_distance=3.0)
        current_loads = self.get_current_driver_loads()
        
        candidates = []
        
        # Adjust thresholds by iteration
        same_group_threshold = 0.5 + (iteration * 0.5)
        adjacent_group_threshold = 0.25 + (iteration * 0.25)
        
        for nearby in nearby_dogs:
            nearby_dog_id = nearby['dog_id']
            distance = nearby['distance']
            
            if nearby_dog_id not in self.dogs_going_today:
                continue
            
            nearby_assignment = self.dogs_going_today[nearby_dog_id]['assignment']
            nearby_driver, nearby_groups = self.parse_group_assignment(nearby_assignment)
            
            if not nearby_driver or not nearby_groups:
                continue
            
            # Skip if it's the same driver that called out
            if nearby_driver == dog_to_reassign['original_driver']:
                continue
            
            # Skip if this driver is also calling out for the needed groups
            if nearby_driver in self.driver_callouts:
                callout = self.driver_callouts[nearby_driver]
                if any([
                    1 in original_groups and callout['group1'],
                    2 in original_groups and callout['group2'],
                    3 in original_groups and callout['group3']
                ]):
                    continue
            
            # Check for group compatibility
            score = 0
            compatible_groups = []
            
            for group in original_groups:
                # Same group match (higher score, stricter distance)
                if group in nearby_groups and distance <= same_group_threshold:
                    score += 10
                    compatible_groups.append(group)
                # Adjacent group match (lower score, even stricter distance)
                elif self.is_adjacent_group(group, nearby_groups) and distance <= adjacent_group_threshold:
                    score += 5
                    compatible_groups.append(group)
            
            # Must match ALL original groups
            if len(compatible_groups) != len(original_groups):
                continue
            
            # Check capacity constraints
            driver_capacity = self.driver_capacities.get(nearby_driver, {'group1': 9, 'group2': 9, 'group3': 9})
            current_load = current_loads.get(nearby_driver, {'group1': 0, 'group2': 0, 'group3': 0})
            
            capacity_ok = True
            for group in original_groups:
                current = current_load.get(f'group{group}', 0)
                capacity = driver_capacity.get(f'group{group}', 9)
                if current + num_dogs > capacity:
                    capacity_ok = False
                    break
            
            if not capacity_ok:
                continue
            
            # Calculate final score
            load_penalty = sum(current_load.values())
            final_score = score - (distance * 2) - (load_penalty * 0.1)
            
            candidates.append({
                'driver': nearby_driver,
                'groups': original_groups,
                'distance': distance,
                'score': final_score,
                'current_load': dict(current_load)
            })
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[0] if candidates else None
    
    def is_adjacent_group(self, group, nearby_groups):
        """Check if any nearby group is adjacent to the target group"""
        adjacent_map = {
            1: [2],
            2: [1, 3], 
            3: [2]
        }
        
        adjacent_to_group = adjacent_map.get(group, [])
        return any(adj_group in nearby_groups for adj_group in adjacent_to_group)
    
    def reassign_dogs(self, max_iterations=5):
        """Main reassignment logic with domino effect handling"""
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            st.info("✅ No dogs need reassignment.")
            return []
        
        st.info(f"🔄 Found {len(dogs_to_reassign)} dogs that need reassignment")
        
        reassignments = []
        iteration = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while dogs_to_reassign and iteration < max_iterations:
            status_text.text(f"Processing iteration {iteration + 1}/{max_iterations}...")
            progress_bar.progress((iteration + 1) / max_iterations)
            
            successful_reassignments = 0
            
            for dog_data in dogs_to_reassign[:]:  # Copy to avoid modification during iteration
                best_match = self.find_best_reassignment(dog_data, iteration)
                
                if best_match:
                    dog_id = dog_data['dog_id']
                    new_driver = best_match['driver']
                    new_groups = best_match['groups']
                    
                    # Create new assignment string
                    if len(new_groups) == 1:
                        new_assignment = f"{new_driver}:{new_groups[0]}"
                    else:
                        new_assignment = f"{new_driver}:{'&'.join(map(str, sorted(new_groups)))}"
                    
                    # Update the assignment
                    self.dogs_going_today[dog_id]['assignment'] = new_assignment
                    
                    reassignments.append({
                        'dog_id': dog_id,
                        'dog_name': dog_data['dog_info']['dog_name'],
                        'from_driver': dog_data['original_driver'],
                        'to_driver': new_driver,
                        'from_groups': dog_data['original_groups'],
                        'to_groups': new_groups,
                        'distance': best_match['distance'],
                        'reason': f"Driver {dog_data['original_driver']} called out"
                    })
                    
                    dogs_to_reassign.remove(dog_data)
                    successful_reassignments += 1
                    
                    st.write(f"  ✅ Moved {dog_data['dog_info']['dog_name']} → {new_assignment}")
            
            if successful_reassignments == 0:
                status_text.text(f"No more reassignments possible after iteration {iteration + 1}")
                break
            
            iteration += 1
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Report results
        if reassignments:
            st.success(f"✅ Successfully reassigned {len(reassignments)} dogs!")
        
        if dogs_to_reassign:
            st.warning(f"⚠️ {len(dogs_to_reassign)} dogs could not be reassigned:")
            for dog_data in dogs_to_reassign:
                dog_name = dog_data['dog_info']['dog_name']
                dog_id = dog_data['dog_id']
                groups = '&'.join(map(str, dog_data['original_groups']))
                original_driver = dog_data['original_driver']
                st.write(f"  - {dog_name} (ID: {dog_id}) from {original_driver}:{groups}")
        
        return reassignments

def main():
    st.set_page_config(
        page_title="Dog Reassignment System",
        page_icon="🐕",
        layout="wide"
    )
    
    st.title("🐕 Dog Reassignment System")
    st.markdown("Choose: Upload CSV files OR use Google Sheets URLs (updates daily)")
    
    # Choose input method
    input_method = st.sidebar.radio(
        "📊 Data Source:",
        ["Google Sheets URLs (Daily Updates)", "Upload CSV Files (Testing)"],
        help="URLs update automatically, file uploads are for testing"
    )
    
    if input_method == "Google Sheets URLs (Daily Updates)":
        st.sidebar.header("📋 Google Sheets URLs")
        st.sidebar.info("💡 Set these URLs once - they'll pull fresh data daily!")
        
        # Initialize session state for URLs if not exists
        if 'distance_url' not in st.session_state:
            st.session_state.distance_url = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/edit?gid=2146002137#gid=2146002137"
        if 'map_url' not in st.session_state:
            st.session_state.map_url = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/edit?gid=267803750#gid=267803750"
        if 'driver_url' not in st.session_state:
            st.session_state.driver_url = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/edit?gid=1359695250#gid=1359695250"
        
        distance_source = st.sidebar.text_input(
            "Distance Matrix URL:",
            value=st.session_state.distance_url,
            key="distance_input",
            help="⚡ Set once - URL stays the same, data updates daily"
        )
        
        map_source = st.sidebar.text_input(
            "Map Sheet URL:",
            value=st.session_state.map_url,
            key="map_input",
            help="⚡ Set once - URL stays the same, data updates daily"
        )
        
        driver_source = st.sidebar.text_input(
            "Driver Counts URL:",
            value=st.session_state.driver_url,
            key="driver_input",
            help="⚡ Set once - URL stays the same, data updates daily"
        )
        
        # Update session state when URLs change
        st.session_state.distance_url = distance_source
        st.session_state.map_url = map_source
        st.session_state.driver_url = driver_source
        
        # Quick setup button
        if st.sidebar.button("💾 Save URLs (Remember for next time)"):
            st.sidebar.success("✅ URLs saved! They'll be remembered next time you visit.")
        
        sources_ready = all([distance_source, map_source, driver_source])
        
    else:  # File upload method
        st.sidebar.header("📁 Upload CSV Files")
        
        distance_source = st.sidebar.file_uploader(
            "Distance Matrix CSV",
            type=['csv'],
            help="Upload the Matrix Matrix 2.csv file"
        )
        
        map_source = st.sidebar.file_uploader(
            "Dog Assignments CSV", 
            type=['csv'],
            help="Upload the New districts Map 8.csv file"
        )
        
        driver_source = st.sidebar.file_uploader(
            "Driver Counts CSV",
            type=['csv'], 
            help="Upload the New districts Driver Counts 3.csv file"
        )
        
        sources_ready = all([distance_source, map_source, driver_source])
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Run Reassignment", type="primary"):
            if not sources_ready:
                if input_method == "Google Sheets URLs (Daily Updates)":
                    st.error("❌ Please provide all three Google Sheet URLs")
                else:
                    st.error("❌ Please upload all three CSV files")
                return
            
            # Initialize system
            system = DogReassignmentSystem()
            
            # Load data
            st.subheader("📊 Loading Data")
            
            with st.spinner("Loading distance matrix..."):
                if not system.load_distance_matrix(distance_source):
                    return
            
            with st.spinner("Loading dog assignments..."):
                if not system.load_dog_assignments(map_source):
                    return
            
            with st.spinner("Loading driver capacities..."):
                if not system.load_driver_capacities(driver_source):
                    return
            
            # Data validation
            st.subheader("🔍 Data Validation")
            matrix_ids = set(system.distance_matrix.keys())
            assignment_ids = set(system.dogs_going_today.keys())
            
            matching_ids = matrix_ids.intersection(assignment_ids)
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Matrix Dogs", len(matrix_ids))
            with col_b:
                st.metric("Assignment Dogs", len(assignment_ids))
            with col_c:
                st.metric("Matching Dogs", len(matching_ids))
            
            if len(matching_ids) == 0:
                st.error("❌ NO MATCHING DOG IDs! Check that Dog IDs match between sheets.")
                return
            
            # Show sample matches
            if len(matching_ids) > 0:
                st.success(f"✅ Found {len(matching_ids)} matching Dog IDs")
                sample_matches = list(matching_ids)[:10]
                st.info(f"Sample matching IDs: {sample_matches}")
            
            # Perform reassignments
            st.subheader("🔄 Processing Reassignments")
            reassignments = system.reassign_dogs()
            
            # Display results
            st.subheader("📋 Results")
            
            if reassignments:
                st.success(f"✅ Successfully processed {len(reassignments)} reassignments!")
                
                # Create results dataframe
                results_df = pd.DataFrame([
                    {
                        'Dog ID': r['dog_id'],
                        'Dog Name': r['dog_name'],
                        'From': f"{r['from_driver']}:{'&'.join(map(str, r['from_groups']))}",
                        'To': f"{r['to_driver']}:{'&'.join(map(str, r['to_groups']))}",
                        'Distance (mi)': f"{r['distance']:.2f}",
                        'Reason': r['reason']
                    }
                    for r in reassignments
                ])
                
                st.dataframe(results_df, use_container_width=True)
                
                # Column updates for copy/paste
                st.subheader("📝 Updates for Today Column")
                st.info("Copy these values into the 'Today' column of your Map sheet:")
                
                for r in reassignments:
                    to_groups_str = '&'.join(map(str, r['to_groups']))
                    st.code(f"Dog ID {r['dog_id']} ({r['dog_name']}): {r['to_driver']}:{to_groups_str}")
                
            else:
                st.info("ℹ️ No reassignments needed - all drivers are available!")
    
    with col2:
        st.subheader("ℹ️ How it works")
        
        if input_method == "Google Sheets URLs (Daily Updates)":
            st.markdown("""
            **🔄 DAILY USE MODE**
            
            ✅ **Set URLs once, use forever**
            ✅ **Gets fresh data automatically** 
            ✅ **No daily URL updates needed**
            
            **Setup (one time):**
            1. 📋 Paste your 3 Google Sheets URLs
            2. 💾 Click "Save URLs" 
            3. ✅ Done forever!
            
            **Daily use:**
            1. 📝 Update your Google Sheets (mark "X" for callouts)
            2. 🚀 Click "Run Reassignment" 
            3. ✅ Gets latest data automatically!
            
            **The URLs never change - only the data inside your sheets changes!**
            """)
        else:
            st.markdown("""
            **📁 TESTING MODE**
            
            Upload the 3 CSV files:
            1. Distance Matrix CSV
            2. Map Sheet CSV  
            3. Driver Counts CSV
            
            **Good for:**
            - 🧪 Testing changes
            - 🔍 Debugging issues
            - 📊 One-time analysis
            
            **For daily use, switch to Google Sheets URLs!**
            """)

if __name__ == "__main__":
    main()
