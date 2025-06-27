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
            df = pd.read_csv(StringIO(response.text))
            
            # Process the matrix - first row and first column are Dog IDs
            dog_ids = df.iloc[0, 1:].tolist()  # Skip first cell, get dog IDs from first row
            
            for i, row_dog_id in enumerate(df.iloc[1:, 0]):  # Skip header row
                if pd.isna(row_dog_id):
                    continue
                row_dog_id = str(int(float(row_dog_id)))
                self.distance_matrix[row_dog_id] = {}
                
                for j, col_dog_id in enumerate(dog_ids):
                    if pd.isna(col_dog_id):
                        continue
                    col_dog_id = str(int(float(col_dog_id)))
                    distance = df.iloc[i + 1, j + 1]  # +1 to account for header
                    if pd.isna(distance):
                        distance = 0
                    self.distance_matrix[row_dog_id][col_dog_id] = float(distance)
            
            st.success(f"✅ Loaded distance matrix for {len(self.distance_matrix)} dogs")
            return True
        except Exception as e:
            st.error(f"❌ Error loading distance matrix: {e}")
            return False
    
    def load_dog_assignments(self, csv_url):
        """Load dog assignments from Map sheet"""
        try:
            if 'edit' in csv_url:
                sheet_id = csv_url.split('/d/')[1].split('/')[0]
                gid = csv_url.split('gid=')[1].split('#')[0] if 'gid=' in csv_url else '267803750'
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            
            response = requests.get(csv_url)
            df = pd.read_csv(StringIO(response.text))
            
            # Process dog assignments
            for _, row in df.iterrows():
                if pd.isna(row.get('Dog ID')) or pd.isna(row.get('Today')):
                    continue
                
                dog_id = str(int(float(row['Dog ID'])))
                assignment = str(row['Today']) if not pd.isna(row['Today']) else None
                num_dogs = int(row['Number of dogs']) if not pd.isna(row['Number of dogs']) else 1
                
                if assignment and ':' in assignment:
                    self.dogs_going_today[dog_id] = {
                        'assignment': assignment,
                        'num_dogs': num_dogs,
                        'address': row.get('Address', ''),
                        'dog_name': row.get('Dog Name', '')
                    }
            
            st.success(f"✅ Loaded assignments for {len(self.dogs_going_today)} dogs going today")
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
            df = pd.read_csv(StringIO(response.text))
            
            for _, row in df.iterrows():
                if pd.isna(row.get('Driver')):
                    continue
                
                driver = str(row['Driver']).strip()
                if not driver:
                    continue
                
                # Check for callouts (X in group columns)
                group1 = str(row.get('Group 1', '')).strip()
                group2 = str(row.get('Group 2', '')).strip()
                group3 = str(row.get('Group 3', '')).strip()
                
                self.driver_callouts[driver] = {
                    'group1': group1.upper() == 'X',
                    'group2': group2.upper() == 'X',
                    'group3': group3.upper() == 'X'
                }
                
                # Parse capacities (default to 9 if not specified or X)
                cap1 = 9 if group1.upper() == 'X' else (int(group1) if group1.isdigit() else 9)
                cap2 = 9 if group2.upper() == 'X' else (int(group2) if group2.isdigit() else 9)
                cap3 = 9 if group3.upper() == 'X' else (int(group3) if group3.isdigit() else 9)
                
                self.driver_capacities[driver] = {
                    'group1': cap1,
                    'group2': cap2,
                    'group3': cap3
                }
            
            # Find drivers calling out
            callout_drivers = [d for d, c in self.driver_callouts.items() if any(c.values())]
            if callout_drivers:
                st.warning(f"🚨 Drivers calling out: {', '.join(callout_drivers)}")
            else:
                st.info("ℹ️ No drivers calling out today")
            
            st.success(f"✅ Loaded capacities for {len(self.driver_capacities)} drivers")
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
        
        # Get nearby dogs
        nearby_dogs = self.get_nearby_dogs(dog_id)
        current_loads = self.get_current_driver_loads()
        
        candidates = []
        
        # Set distance thresholds based on groups
        base_threshold = 0.5 if 1 in original_groups else 0.25 if 2 in original_groups else 0.5
        threshold = base_threshold + (iteration * 0.5)  # Expand with iterations
        
        for nearby in nearby_dogs:
            nearby_dog_id = nearby['dog_id']
            distance = nearby['distance']
            
            if distance > threshold:
                continue
            
            if nearby_dog_id not in self.dogs_going_today:
                continue
            
            nearby_assignment = self.dogs_going_today[nearby_dog_id]['assignment']
            nearby_driver, nearby_groups = self.parse_group_assignment(nearby_assignment)
            
            if not nearby_driver or not nearby_groups:
                continue
            
            # Skip if it's the same driver that called out
            if nearby_driver == dog_to_reassign['original_driver']:
                continue
            
            # Skip if this driver is also calling out
            if nearby_driver in self.driver_callouts:
                callout = self.driver_callouts[nearby_driver]
                if any([
                    1 in original_groups and callout['group1'],
                    2 in original_groups and callout['group2'],
                    3 in original_groups and callout['group3']
                ]):
                    continue
            
            # Calculate compatibility score
            score = 0
            compatible_groups = []
            
            for group in original_groups:
                if group in nearby_groups:
                    # Same group - full weight
                    score += 10
                    compatible_groups.append(group)
                elif abs(group - max(nearby_groups)) == 1 or abs(group - min(nearby_groups)) == 1:
                    # Adjacent group - half weight
                    score += 5
                    # For multi-group dogs, we can assign to adjacent groups
                    if len(original_groups) > 1:
                        compatible_groups.append(group)
            
            if not compatible_groups:
                continue
            
            # Check capacity constraints
            driver_capacity = self.driver_capacities.get(nearby_driver, {'group1': 9, 'group2': 9, 'group3': 9})
            current_load = current_loads.get(nearby_driver, {'group1': 0, 'group2': 0, 'group3': 0})
            
            capacity_ok = True
            for group in compatible_groups:
                current = current_load.get(f'group{group}', 0)
                capacity = driver_capacity.get(f'group{group}', 9)
                if current + num_dogs > capacity:
                    capacity_ok = False
                    break
            
            if capacity_ok:
                # Prefer drivers with fewer dogs
                load_penalty = sum(current_load.values())
                final_score = score - (distance * 2) - (load_penalty * 0.1)
                
                candidates.append({
                    'driver': nearby_driver,
                    'groups': compatible_groups,
                    'distance': distance,
                    'score': final_score,
                    'current_load': dict(current_load)
                })
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0] if candidates else None
    
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
            
            if successful_reassignments == 0:
                break
            
            iteration += 1
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        if dogs_to_reassign:
            st.warning(f"⚠️ {len(dogs_to_reassign)} dogs could not be reassigned:")
            for dog_data in dogs_to_reassign:
                st.write(f"  - {dog_data['dog_info']['dog_name']} (ID: {dog_data['dog_id']})")
        
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
