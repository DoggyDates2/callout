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
        
    def load_distance_matrix(self, csv_url):
        """Load distance matrix from Google Sheets CSV export"""
        try:
            # Convert Google Sheets edit URL to CSV export URL
            if 'edit' in csv_url:
                sheet_id = csv_url.split('/d/')[1].split('/')[0]
                gid = csv_url.split('gid=')[1].split('#')[0] if 'gid=' in csv_url else '0'
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            
            response = requests.get(csv_url)
            df = pd.read_csv(StringIO(response.text), dtype=str)  # Force all data as strings
            
            # Process the matrix - first row and first column are Dog IDs
            dog_ids = df.iloc[0, 1:].tolist()  # Skip first cell, get dog IDs from first row
            
            # Clean and convert dog IDs from header row
            clean_dog_ids = []
            for dog_id in dog_ids:
                if pd.isna(dog_id) or str(dog_id).strip() == '':
                    continue
                # Keep as string, just strip whitespace - DON'T convert to float
                clean_id = str(dog_id).strip()
                clean_dog_ids.append(clean_id)
            
            # Process each row
            for i, row_dog_id in enumerate(df.iloc[1:, 0]):  # Skip header row
                if pd.isna(row_dog_id) or str(row_dog_id).strip() == '':
                    continue
                
                # Keep row dog ID as string - DON'T convert to float
                row_dog_id = str(row_dog_id).strip()
                self.distance_matrix[row_dog_id] = {}
                
                # Process distances for this row
                for j, col_dog_id in enumerate(clean_dog_ids):
                    distance_val = df.iloc[i + 1, j + 1]  # +1 to account for header
                    
                    # Only convert DISTANCE values to float, not Dog IDs
                    if pd.isna(distance_val) or str(distance_val).strip() == '':
                        distance = 0.0
                    else:
                        distance_str = str(distance_val).strip()
                        # Check if this looks like a number (distance) vs Dog ID
                        try:
                            # Try to convert to float - this should be a distance value
                            distance = float(distance_str)
                        except (ValueError, TypeError):
                            # If conversion fails, it might be a Dog ID or invalid data
                            # Set distance to 0 for invalid entries
                            distance = 0.0
                    
                    self.distance_matrix[row_dog_id][col_dog_id] = distance
            
            # Debug output
            matrix_ids = list(self.distance_matrix.keys())[:10]  # Show first 10
            st.success(f"✅ Loaded distance matrix for {len(self.distance_matrix)} dogs")
            st.info(f"🔍 Matrix Dog IDs (first 10): {matrix_ids}")
            
            # Show a sample of the distance data
            if len(self.distance_matrix) > 0:
                first_dog = list(self.distance_matrix.keys())[0]
                sample_distances = list(self.distance_matrix[first_dog].items())[:5]
                st.info(f"📏 Sample distances for {first_dog}: {sample_distances}")
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading distance matrix: {e}")
            import traceback
            st.error(f"🔍 Full error: {traceback.format_exc()}")
            return False
    
    def load_dog_assignments(self, csv_url):
        """Load dog assignments from Map sheet"""
        try:
            if 'edit' in csv_url:
                sheet_id = csv_url.split('/d/')[1].split('/')[0]
                gid = csv_url.split('gid=')[1].split('#')[0] if 'gid=' in csv_url else '267803750'
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            
            response = requests.get(csv_url)
            df = pd.read_csv(StringIO(response.text), dtype=str)  # Force all data as strings
            
            # Debug: Show column names
            st.info(f"🔍 Map sheet columns: {list(df.columns)}")
            
            # Process dog assignments
            assignments_found = []
            for _, row in df.iterrows():
                # Get Dog ID - try different possible column names
                dog_id = None
                for col_name in ['Dog ID', 'DogID', 'Dog_ID', 'dog_id']:
                    if col_name in row and not pd.isna(row.get(col_name)):
                        dog_id = str(row[col_name]).strip()
                        break
                
                # Get Today assignment - try different possible column names  
                assignment = None
                for col_name in ['Today', 'Group', 'Assignment']:
                    if col_name in row and not pd.isna(row.get(col_name)):
                        assignment = str(row[col_name]).strip()
                        break
                
                if not dog_id or not assignment or dog_id == '' or assignment == '':
                    continue
                
                # Get number of dogs
                num_dogs = 1
                for col_name in ['Number of dogs', 'Num Dogs', 'Count']:
                    if col_name in row and not pd.isna(row.get(col_name)):
                        try:
                            num_dogs = int(float(str(row[col_name]).strip()))
                        except (ValueError, TypeError):
                            num_dogs = 1
                        break
                
                if assignment and ':' in assignment:
                    self.dogs_going_today[dog_id] = {
                        'assignment': assignment,
                        'num_dogs': num_dogs,
                        'address': row.get('Address', ''),
                        'dog_name': row.get('Dog Name', '')
                    }
                    assignments_found.append(f"{dog_id}:{assignment}")
            
            # Debug output
            assignment_ids = list(self.dogs_going_today.keys())[:10]  # Show first 10
            st.success(f"✅ Loaded assignments for {len(self.dogs_going_today)} dogs going today")
            st.info(f"🔍 Assignment Dog IDs (first 10): {assignment_ids}")
            
            if len(assignments_found) > 0:
                st.info(f"📋 Sample assignments: {assignments_found[:5]}")
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading dog assignments: {e}")
            return False
    
    def load_driver_capacities(self, csv_url):
        """Load driver capacities and callouts from Driver Counts sheet"""
        try:
            if 'edit' in csv_url:
                sheet_id = csv_url.split('/d/')[1].split('/')[0]
                gid = csv_url.split('gid=')[1].split('#')[0] if 'gid=' in csv_url else '1359695250'
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            
            response = requests.get(csv_url)
            df = pd.read_csv(StringIO(response.text), dtype=str)  # Force all data as strings
            
            # Debug: Show column names
            st.info(f"🔍 Driver sheet columns: {list(df.columns)}")
            
            drivers_loaded = []
            callout_drivers = []
            
            for _, row in df.iterrows():
                if pd.isna(row.get('Driver')) or str(row.get('Driver')).strip() == '':
                    continue
                
                driver = str(row['Driver']).strip()
                if not driver:
                    continue
                
                # Check for callouts (X in group columns)
                group1 = str(row.get('Group 1', '')).strip().upper()
                group2 = str(row.get('Group 2', '')).strip().upper()
                group3 = str(row.get('Group 3', '')).strip().upper()
                
                self.driver_callouts[driver] = {
                    'group1': group1 == 'X',
                    'group2': group2 == 'X',
                    'group3': group3 == 'X'
                }
                
                # Parse capacities (default to 9 if not specified or X)
                try:
                    cap1 = 9 if group1 == 'X' or group1 == '' else int(group1)
                except (ValueError, TypeError):
                    cap1 = 9
                
                try:
                    cap2 = 9 if group2 == 'X' or group2 == '' else int(group2)
                except (ValueError, TypeError):
                    cap2 = 9
                
                try:
                    cap3 = 9 if group3 == 'X' or group3 == '' else int(group3)
                except (ValueError, TypeError):
                    cap3 = 9
                
                self.driver_capacities[driver] = {
                    'group1': cap1,
                    'group2': cap2,
                    'group3': cap3
                }
                
                drivers_loaded.append(driver)
                
                # Track callouts
                if any([group1 == 'X', group2 == 'X', group3 == 'X']):
                    callout_details = []
                    if group1 == 'X': callout_details.append("Group 1")
                    if group2 == 'X': callout_details.append("Group 2") 
                    if group3 == 'X': callout_details.append("Group 3")
                    callout_drivers.append(f"{driver} ({', '.join(callout_details)})")
            
            # Debug output
            if callout_drivers:
                st.warning(f"🚨 Drivers calling out: {', '.join(callout_drivers)}")
            else:
                st.info("ℹ️ No drivers calling out today")
            
            st.success(f"✅ Loaded capacities for {len(self.driver_capacities)} drivers")
            st.info(f"🔍 Drivers found: {drivers_loaded[:10]}")  # Show first 10
            
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
        if 'XX' in group_part:
            return None, []
        
        # Parse groups (1, 2, 3, 1&2, 2&3, 1&2&3, etc.)
        groups = []
        if '&' in group_part:
            for g in group_part.split('&'):
                if g.isdigit() and 1 <= int(g) <= 3:
                    groups.append(int(g))
        else:
            # Handle single group or special codes like "2LM2"
            clean_group = re.findall(r'[123]', group_part)
            for g in clean_group:
                if g.isdigit() and 1 <= int(g) <= 3:
                    groups.append(int(g))
        
        return driver, sorted(list(set(groups)))  # Remove duplicates and sort
    
    def get_dogs_to_reassign(self):
        """Get list of dogs that need to be reassigned due to driver callouts"""
        dogs_to_reassign = []
        
        for dog_id, dog_info in self.dogs_going_today.items():
            assignment = dog_info['assignment']
            driver, groups = self.parse_group_assignment(assignment)
            
            if not driver or not groups:
                continue
            
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
    
    def get_nearby_dogs(self, dog_id, max_distance=3.0):  # Back to original
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
        debug_info = []
        
        # CORRECTED: Different thresholds for same vs adjacent groups
        same_group_threshold = 0.5 + (iteration * 0.5)  # Same group: 0.5 miles base
        adjacent_group_threshold = 0.25 + (iteration * 0.25)  # Adjacent group: 0.25 miles base (CLOSER required)
        
        debug_info.append(f"🔍 {dog_name} ({dog_id}) Groups {original_groups}:")
        debug_info.append(f"   Same group threshold: {same_group_threshold:.2f} miles")
        debug_info.append(f"   Adjacent group threshold: {adjacent_group_threshold:.2f} miles")
        
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
                    debug_info.append(f"   ❌ {nearby_driver} also calling out")
                    continue
            
            # CORRECTED WEIGHTED MATCHING LOGIC
            score = 0
            compatible_groups = []
            match_type = None
            distance_threshold_used = 0
            
            for group in original_groups:
                # Check for SAME GROUP matches first (full weight, 0.5 mile threshold)
                if group in nearby_groups:
                    if distance <= same_group_threshold:
                        score += 10  # Full weight for same group
                        compatible_groups.append(group)
                        match_type = "same"
                        distance_threshold_used = same_group_threshold
                        debug_info.append(f"   ✅ {nearby_driver} SAME group {group} match at {distance:.2f}mi (threshold: {same_group_threshold:.2f})")
                    else:
                        debug_info.append(f"   ❌ {nearby_driver} same group {group} too far: {distance:.2f}mi > {same_group_threshold:.2f}mi")
                
                # Check for ADJACENT GROUP matches (half weight, 0.25 mile threshold - CLOSER required)
                elif self.is_adjacent_group(group, nearby_groups):
                    if distance <= adjacent_group_threshold:
                        score += 5  # Half weight for adjacent group
                        compatible_groups.append(group)
                        match_type = "adjacent"
                        distance_threshold_used = adjacent_group_threshold
                        debug_info.append(f"   🟡 {nearby_driver} ADJACENT group match (need {group}, has {nearby_groups}) at {distance:.2f}mi (threshold: {adjacent_group_threshold:.2f})")
                    else:
                        debug_info.append(f"   ❌ {nearby_driver} adjacent group too far: {distance:.2f}mi > {adjacent_group_threshold:.2f}mi")
            
            if not compatible_groups:
                debug_info.append(f"   ❌ {nearby_driver} no viable matches (has {nearby_groups}, need {original_groups})")
                continue
            
            # Must match ALL original groups for this to work
            if len(compatible_groups) != len(original_groups):
                debug_info.append(f"   ❌ {nearby_driver} partial match only ({len(compatible_groups)}/{len(original_groups)} groups)")
                continue
            
            # Check capacity constraints
            driver_capacity = self.driver_capacities.get(nearby_driver, {'group1': 9, 'group2': 9, 'group3': 9})
            current_load = current_loads.get(nearby_driver, {'group1': 0, 'group2': 0, 'group3': 0})
            
            capacity_ok = True
            capacity_details = []
            for group in original_groups:  # Use original groups for capacity check
                current = current_load.get(f'group{group}', 0)
                capacity = driver_capacity.get(f'group{group}', 9)
                if current + num_dogs > capacity:
                    capacity_ok = False
                    capacity_details.append(f"Group{group}: {current}+{num_dogs}>{capacity}")
                else:
                    capacity_details.append(f"Group{group}: {current}+{num_dogs}<={capacity}")
            
            if not capacity_ok:
                debug_info.append(f"   ❌ {nearby_driver} over capacity: {', '.join(capacity_details)}")
                continue
            
            # Calculate final score
            load_penalty = sum(current_load.values())
            final_score = score - (distance * 2) - (load_penalty * 0.1)
            
            candidates.append({
                'driver': nearby_driver,
                'groups': original_groups,  # Keep original groups
                'distance': distance,
                'score': final_score,
                'current_load': dict(current_load),
                'match_type': match_type
            })
            
            debug_info.append(f"   ✅ {nearby_driver} VIABLE: {match_type} match, score={final_score:.1f}, {distance:.2f}mi")
        
        # Store debug info for failed assignments
        if not candidates:
            debug_info.append(f"   ❌ NO VIABLE CANDIDATES for {dog_name}")
            if not hasattr(self, 'debug_failures'):
                self.debug_failures = []
            self.debug_failures.extend(debug_info)
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if candidates:
            best = candidates[0]
            debug_info.append(f"   🎯 SELECTED: {best['driver']} ({best['match_type']} match, score: {best['score']:.1f})")
            
        return candidates[0] if candidates else None
    
    def is_adjacent_group(self, group, nearby_groups):
        """Check if any nearby group is adjacent to the target group"""
        # Group 1 is adjacent to Group 2
        # Group 2 is adjacent to Group 1 and Group 3  
        # Group 3 is adjacent to Group 2
        adjacent_map = {
            1: [2],
            2: [1, 3], 
            3: [2]
        }
        
        adjacent_to_group = adjacent_map.get(group, [])
        return any(adj_group in nearby_groups for adj_group in adjacent_to_group)
    
    def reassign_dogs(self, max_iterations=5):  # Back to original
        """Main reassignment logic with domino effect handling"""
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            st.info("✅ No dogs need reassignment.")
            return []
        
        st.info(f"🔄 Found {len(dogs_to_reassign)} dogs that need reassignment")
        
        # Initialize debug tracking
        self.debug_failures = []
        
        reassignments = []
        iteration = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while dogs_to_reassign and iteration < max_iterations:
            status_text.text(f"Processing iteration {iteration + 1}/{max_iterations}...")
            progress_bar.progress((iteration + 1) / max_iterations)
            
            successful_reassignments = 0
            
            # Try to reassign each dog - continue even if some fail
            for dog_data in dogs_to_reassign[:]:  # Copy list to avoid modification during iteration
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
            
            # Stop if no progress made in this iteration
            if successful_reassignments == 0:
                status_text.text(f"No more reassignments possible after iteration {iteration + 1}")
                break
            
            iteration += 1
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Report results - partial success is still success!
        if reassignments:
            st.success(f"✅ Successfully reassigned {len(reassignments)} dogs!")
        
        if dogs_to_reassign:
            st.warning(f"⚠️ {len(dogs_to_reassign)} dogs could not be reassigned (but {len(reassignments)} were successfully moved):")
            
            # Show which dogs failed with their details
            for dog_data in dogs_to_reassign:
                dog_name = dog_data['dog_info']['dog_name']
                dog_id = dog_data['dog_id']
                groups = '&'.join(map(str, dog_data['original_groups']))
                original_driver = dog_data['original_driver']
                st.write(f"  - {dog_name} (ID: {dog_id}) from {original_driver}:{groups}")
            
            # Show detailed debugging for failures
            if hasattr(self, 'debug_failures') and self.debug_failures:
                with st.expander("🔍 Debug Info for Failed Dogs (Click to expand)"):
                    for debug_line in self.debug_failures:
                        st.text(debug_line)
            
            # Provide helpful context
            st.info(f"""
            ℹ️ **Results Summary:**
            - ✅ **{len(reassignments)} dogs successfully reassigned**
            - ⚠️ **{len(dogs_to_reassign)} dogs need manual assignment**
            
            Common reasons dogs can't be auto-reassigned:
            - No nearby drivers with matching groups and available capacity
            - Dog is in a remote location with limited nearby options
            - All nearby drivers are at capacity for that group
            """)
        
        return reassignments

def main():
    st.set_page_config(
        page_title="Dog Reassignment System",
        page_icon="🐕",
        layout="wide"
    )
    
    st.title("🐕 Dog Reassignment System")
    st.markdown("Automatically reassign dogs when drivers call out")
    
    # Sidebar for inputs
    st.sidebar.header("📋 Configuration")
    
    # Default URLs
    default_distance_url = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/edit?gid=2146002137#gid=2146002137"
    default_map_url = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/edit?gid=267803750#gid=267803750"
    default_driver_url = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/edit?gid=1359695250#gid=1359695250"
    
    distance_matrix_url = st.sidebar.text_input(
        "Distance Matrix URL:",
        value=default_distance_url,
        help="URL to the Google Sheet with the distance matrix"
    )
    
    map_sheet_url = st.sidebar.text_input(
        "Map Sheet URL:",
        value=default_map_url,
        help="URL to the Google Sheet with dog assignments"
    )
    
    driver_counts_url = st.sidebar.text_input(
        "Driver Counts URL:",
        value=default_driver_url,
        help="URL to the Google Sheet with driver capacities"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Run Reassignment", type="primary"):
            if not all([distance_matrix_url, map_sheet_url, driver_counts_url]):
                st.error("❌ Please provide all three Google Sheet URLs")
                return
            
            # Initialize system
            system = DogReassignmentSystem()
            
            # Load data
            st.subheader("📊 Loading Data")
            with st.spinner("Loading distance matrix..."):
                if not system.load_distance_matrix(distance_matrix_url):
                    return
            
            with st.spinner("Loading dog assignments..."):
                if not system.load_dog_assignments(map_sheet_url):
                    return
            
            with st.spinner("Loading driver capacities..."):
                if not system.load_driver_capacities(driver_counts_url):
                    return
            
            # Data validation section
            st.subheader("🔍 Data Validation")
            matrix_ids = set(system.distance_matrix.keys())
            assignment_ids = set(system.dogs_going_today.keys())
            
            # Show overlap
            matching_ids = matrix_ids.intersection(assignment_ids)
            matrix_only = matrix_ids - assignment_ids
            assignment_only = assignment_ids - matrix_ids
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Matrix Dogs", len(matrix_ids))
                if len(matrix_only) > 0:
                    st.write(f"Matrix only (first 5): {list(matrix_only)[:5]}")
            
            with col_b:
                st.metric("Assignment Dogs", len(assignment_ids))
                if len(assignment_only) > 0:
                    st.write(f"Assignments only (first 5): {list(assignment_only)[:5]}")
            
            with col_c:
                st.metric("Matching Dogs", len(matching_ids))
                if len(matching_ids) > 0:
                    st.write(f"Matching (first 5): {list(matching_ids)[:5]}")
            
            if len(matching_ids) == 0:
                st.error("❌ NO MATCHING DOG IDs FOUND! Check that Dog IDs are in plain text format in both sheets.")
                st.info("💡 Try converting Dog ID columns to 'Plain text' format in Google Sheets")
                return
            
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
                
                # Column K updates
                st.subheader("📝 Column K Updates")
                st.info("Copy these values into Column K of your Map sheet:")
                
                for r in reassignments:
                    to_groups_str = '&'.join(map(str, r['to_groups']))
                    st.code(f"Dog ID {r['dog_id']} ({r['dog_name']}): {r['to_driver']}:{to_groups_str}")
                
            else:
                st.info("ℹ️ No reassignments needed - all drivers are available!")
    
    with col2:
        st.subheader("ℹ️ How it works")
        st.markdown("""
        **This system automatically:**
        
        1. 🔍 **Detects callouts** - Finds drivers marked with "X"
        
        2. 📍 **Uses distance matrix** - Finds nearby dogs for optimal placement
        
        3. 🎯 **Maintains groups** - Keeps dogs in compatible groups (1→1, 2→2, etc.)
        
        4. 📊 **Respects capacity** - Ensures drivers don't exceed limits
        
        5. 🔄 **Handles domino effects** - Multiple iterations if needed
        
        6. 📝 **Provides updates** - Clear Column K suggestions
        
        **Distance thresholds:**
        - Group 1: 0.5 miles
        - Group 2: 0.25 miles  
        - Group 3: 0.5 miles
        
        **Expands search** if no matches found initially.
        """)

if __name__ == "__main__":
    main()
