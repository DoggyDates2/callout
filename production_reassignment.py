# production_reassignment.py
# Your EXACT working logic from Streamlit, adapted for GitHub Actions → Google Sheets
# All algorithms preserved exactly as you built them
# FIXED: Driver capacities now loaded from columns R:W on map sheet

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Set
import json
import os
from datetime import datetime
import re
from collections import defaultdict

# Google Sheets API
import gspread
from google.oauth2.service_account import Credentials

class ProductionDogReassignmentSystem:
    """
    Production version with your EXACT working logic
    Only the I/O changed (GitHub + Google Sheets instead of Streamlit + file uploads)
    FIXED: Driver capacities loaded from same sheet as map data (columns R:W)
    """
    
    def __init__(self):
        # Updated URLs with correct GIDs
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        # Driver capacities are now on the same sheet as map data (columns R:W)
        
        # For writing results back
        self.MAP_SPREADSHEET_ID = "1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0"
        
        # Data storage
        self.distance_matrix = {}
        self.dog_assignments = {}
        self.driver_capacities = {}
        self.callout_dogs = []
        
        print("🚀 Production Dog Reassignment System Initialized")
    
    def setup_google_sheets_client(self):
        """Setup Google Sheets API client using service account credentials"""
        try:
            # DEBUG: Print all environment variables to see what's available
            print("🔍 DEBUG: Available environment variables:")
            for key, value in os.environ.items():
                if 'GOOGLE' in key or 'TEST' in key or 'SECRET' in key:
                    # Show first/last 20 chars for security
                    safe_value = value[:20] + "..." + value[-20:] if len(value) > 40 else value[:10] + "..."
                    print(f"   {key}: {safe_value}")
            
            credentials_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            test_secret = os.environ.get('TEST_SECRET')
            
            print(f"🔍 GOOGLE_SERVICE_ACCOUNT_JSON exists: {credentials_json is not None}")
            print(f"🔍 GOOGLE_SERVICE_ACCOUNT_JSON length: {len(credentials_json) if credentials_json else 0}")
            print(f"🔍 TEST_SECRET value: {test_secret}")
            
            if not credentials_json:
                print("❌ Error: GOOGLE_SERVICE_ACCOUNT_JSON environment variable not found")
                return None
            
            credentials_dict = json.loads(credentials_json)
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            client = gspread.authorize(credentials)
            print("✅ Google Sheets client setup successful")
            return client
        
        except Exception as e:
            print(f"❌ Error setting up Google Sheets client: {e}")
            return None
    
    def load_distance_matrix(self):
        """Load distance matrix from Google Sheets CSV export"""
        try:
            print("📊 Loading distance matrix...")
            response = requests.get(self.DISTANCE_MATRIX_URL)
            response.raise_for_status()
            
            # DEBUG: Print first 500 characters of response
            print(f"🔍 Distance matrix CSV preview:")
            print(response.text[:500])
            print("..." if len(response.text) > 500 else "")
            
            # Parse CSV content
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"📊 Distance matrix shape: {df.shape}")
            print(f"📊 Distance matrix columns: {list(df.columns)}")
            
            # DEBUG: Print first few rows
            if len(df) > 0:
                print(f"🔍 First 3 rows of distance matrix:")
                print(df.head(3))
            else:
                print("⚠️ Distance matrix DataFrame is empty!")
            
            # Handle the case where first column contains row Dog IDs and other columns contain column Dog IDs
            distance_matrix = {}
            
            for index, row in df.iterrows():
                row_dog_id = row.iloc[0]  # First column has the row Dog ID
                if pd.isna(row_dog_id):
                    continue
                    
                # Skip invalid row dog IDs
                if not str(row_dog_id).strip():
                    continue
                
                row_dog_id = str(row_dog_id).strip()
                
                for col_idx in range(1, len(row)):
                    col_dog_id = df.columns[col_idx]  # Column name is the column Dog ID
                    distance = row.iloc[col_idx]
                    
                    if pd.isna(distance) or pd.isna(col_dog_id):
                        continue
                    
                    try:
                        distance = float(distance)
                        if distance > 0:  # Only include positive distances
                            distance_matrix[(row_dog_id, str(col_dog_id))] = distance
                    except (ValueError, TypeError):
                        continue
            
            self.distance_matrix = distance_matrix
            print(f"✅ Loaded distance matrix with {len(distance_matrix)} entries")
            
            # Get unique dog IDs from matrix for validation
            matrix_dogs = set()
            for (row_dog, col_dog) in distance_matrix.keys():
                matrix_dogs.add(row_dog)
                matrix_dogs.add(col_dog)
            
            print(f"📊 Matrix dogs: {len(matrix_dogs)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading distance matrix: {e}")
            return False
    
    def load_dog_assignments(self):
        """Load dog assignments and detect callouts from blank Combined column but content in Callout column"""
        try:
            print("🐕 Loading dog assignments...")
            response = requests.get(self.MAP_SHEET_URL)
            response.raise_for_status()
            
            # DEBUG: Print first 500 characters of response
            print(f"🔍 Map sheet CSV preview:")
            print(response.text[:500])
            print("..." if len(response.text) > 500 else "")
            
            # Parse CSV content
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"🐕 Map sheet shape: {df.shape}")
            print(f"🐕 Map sheet columns: {list(df.columns)}")
            
            # DEBUG: Print first few rows
            if len(df) > 0:
                print(f"🔍 First 3 rows of map sheet:")
                print(df.head(3))
            else:
                print("⚠️ Map sheet DataFrame is empty!")
            
            dog_assignments = {}
            callout_dogs = []
            
            for index, row in df.iterrows():
                try:
                    # Column positions based on screenshot:
                    # A=Address(0), B=Dog Name(1), ... G=Name(6), H=Combined(7), I=Group(8), J=Dog ID(9), K=Callout(10)
                    
                    # Get Dog ID from Column J (index 9)
                    if len(row) <= 9:
                        continue
                    dog_id = row.iloc[9]
                    if pd.isna(dog_id) or str(dog_id).strip() == '':
                        continue
                    
                    dog_id = str(dog_id).strip()
                    
                    # Get other data
                    dog_name = row.iloc[1] if len(row) > 1 else ''  # Column B
                    address = row.iloc[0] if len(row) > 0 else ''   # Column A
                    num_dogs = row.iloc[4] if len(row) > 4 else 1   # Column E
                    
                    # Handle num_dogs
                    try:
                        num_dogs = int(float(num_dogs)) if not pd.isna(num_dogs) else 1
                    except (ValueError, TypeError):
                        num_dogs = 1
                    
                    # New callout detection logic:
                    # Combined column (H) blank + Callout column (K) has content = callout
                    combined_col = row.iloc[7] if len(row) > 7 else ''   # Column H (Combined)
                    callout_col = row.iloc[10] if len(row) > 10 else ''  # Column K (Callout)
                    
                    # Simple callout detection
                    combined_is_blank = pd.isna(combined_col) or str(combined_col).strip() == ''
                    callout_has_content = not (pd.isna(callout_col) or str(callout_col).strip() == '')
                    
                    is_callout = combined_is_blank and callout_has_content
                    
                    if is_callout:
                        print(f"🚨 Callout detected: Dog {dog_id} needs assignment for {callout_col}")
                        
                        # Parse callout to get groups (e.g., "Nate:1&2" → groups [1,2])
                        callout_str = str(callout_col).strip()
                        if ':' in callout_str:
                            _, groups_str = callout_str.split(':', 1)
                            groups = self._parse_groups(groups_str)
                        else:
                            # Fallback - try to parse as groups directly
                            groups = self._parse_groups(callout_str)
                        
                        callout_dogs.append({
                            'dog_id': dog_id,
                            'dog_name': dog_name,
                            'address': address,
                            'num_dogs': num_dogs,
                            'original_callout': callout_str,
                            'needed_groups': groups
                        })
                    else:
                        # Regular assignment
                        if not combined_is_blank:
                            # Parse existing assignment from Combined column
                            combined_str = str(combined_col).strip()
                            if ':' in combined_str:
                                driver, groups_str = combined_str.split(':', 1)
                                groups = self._parse_groups(groups_str)
                            else:
                                driver = combined_str
                                groups = []
                            
                            dog_assignments[dog_id] = {
                                'dog_name': dog_name,
                                'address': address,
                                'num_dogs': num_dogs,
                                'current_driver': driver.strip(),
                                'current_groups': groups,
                                'assignment_str': combined_str
                            }
                
                except Exception as e:
                    print(f"⚠️ Error processing row {index}: {e}")
                    continue
            
            self.dog_assignments = dog_assignments
            self.callout_dogs = callout_dogs
            
            print(f"✅ Loaded {len(dog_assignments)} regular assignments")
            print(f"🚨 Found {len(callout_dogs)} callouts that need drivers assigned:")
            for callout in callout_dogs:
                print(f"   - {callout['dog_name']} ({callout['dog_id']}) needs replacement for {callout['original_callout']}")
            
            # Get dog IDs for validation
            assignment_dogs = set(dog_assignments.keys())
            callout_dog_ids = set([c['dog_id'] for c in callout_dogs])
            all_dog_ids = assignment_dogs.union(callout_dog_ids)
            
            print(f"🐕 Assignment dogs: {len(assignment_dogs)}")
            print(f"🚨 Callout dogs: {len(callout_dog_ids)}")
            print(f"📊 Total dogs: {len(all_dog_ids)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dog assignments: {e}")
            return False
    
    def load_driver_capacities(self):
        """Load driver capacities from columns R:W on the map sheet"""
        try:
            print("👥 Loading driver capacities from map sheet columns R:W...")
            response = requests.get(self.MAP_SHEET_URL)
            response.raise_for_status()
            
            # Parse CSV content
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"👥 Driver data shape: {df.shape}")
            print(f"👥 Driver data columns: {list(df.columns)}")
            
            # DEBUG: Check if we have enough columns
            print(f"🔍 DataFrame has {len(df.columns)} columns, need at least 23 for column W")
            
            # DEBUG: Print column indices we're trying to access
            if len(df) > 0:
                print(f"🔍 Sample row to check column access:")
                sample_row = df.iloc[0]
                print(f"   Row length: {len(sample_row)}")
                if len(sample_row) > 17:
                    print(f"   Column R (17): {sample_row.iloc[17] if not pd.isna(sample_row.iloc[17]) else 'NaN'}")
                if len(sample_row) > 20:
                    print(f"   Column U (20): {sample_row.iloc[20] if not pd.isna(sample_row.iloc[20]) else 'NaN'}")
                if len(sample_row) > 21:
                    print(f"   Column V (21): {sample_row.iloc[21] if not pd.isna(sample_row.iloc[21]) else 'NaN'}")
                if len(sample_row) > 22:
                    print(f"   Column W (22): {sample_row.iloc[22] if not pd.isna(sample_row.iloc[22]) else 'NaN'}")
            
            driver_capacities = {}
            
            for index, row in df.iterrows():
                try:
                    # Column positions: R=Driver(17), U=Group1(20), V=Group2(21), W=Group3(22)
                    if len(row) <= 22:
                        continue
                        
                    driver_name = row.iloc[17]  # Column R (Driver)
                    if pd.isna(driver_name) or str(driver_name).strip() == '':
                        continue
                    
                    driver_name = str(driver_name).strip()
                    
                    # Get capacities for each group
                    group1_cap = row.iloc[20] if len(row) > 20 else 9  # Column U (Group 1)
                    group2_cap = row.iloc[21] if len(row) > 21 else 9  # Column V (Group 2)  
                    group3_cap = row.iloc[22] if len(row) > 22 else 9  # Column W (Group 3)
                    
                    # Convert to integers with fallback
                    try:
                        group1_cap = int(float(group1_cap)) if not pd.isna(group1_cap) else 9
                    except (ValueError, TypeError):
                        group1_cap = 9
                    
                    try:
                        group2_cap = int(float(group2_cap)) if not pd.isna(group2_cap) else 9
                    except (ValueError, TypeError):
                        group2_cap = 9
                    
                    try:
                        group3_cap = int(float(group3_cap)) if not pd.isna(group3_cap) else 9
                    except (ValueError, TypeError):
                        group3_cap = 9
                    
                    driver_capacities[driver_name] = {
                        'group1': group1_cap,
                        'group2': group2_cap,
                        'group3': group3_cap
                    }
                    
                except Exception as e:
                    print(f"⚠️ Error processing driver row {index}: {e}")
                    continue
            
            self.driver_capacities = driver_capacities
            print(f"✅ Loaded capacities for {len(driver_capacities)} drivers")
            
            # Debug: Show some driver capacities
            for driver_name, caps in list(driver_capacities.items())[:5]:
                print(f"   - {driver_name}: Group1={caps['group1']}, Group2={caps['group2']}, Group3={caps['group3']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading driver capacities: {e}")
            return False
    
    def _parse_groups(self, group_str: str) -> List[int]:
        """Parse group string into list of integers (your exact logic)"""
        if not group_str or pd.isna(group_str):
            return []
        
        group_str = str(group_str).strip()
        
        # Handle special case like "2XX2" -> [2]
        if 'XX' in group_str and any(c.isdigit() for c in group_str):
            digits = [int(c) for c in group_str if c.isdigit()]
            return sorted(set(digits))
        
        # Extract all digits from the string
        digits = [int(c) for c in group_str if c.isdigit()]
        return sorted(set(digits))
    
    def get_dogs_to_reassign(self):
        """Find dogs that need driver assignment (callouts detected from blank Combined column but content in Callout column)"""
        return self.callout_dogs
    
    def get_nearby_dogs(self, dog_id, max_distance=3.0):
        """Get nearby dogs within max_distance (your exact logic)"""
        nearby_dogs = []
        
        for (row_dog, col_dog), distance in self.distance_matrix.items():
            if row_dog == dog_id and distance > 0 and distance <= max_distance:
                nearby_dogs.append((col_dog, distance))
            elif col_dog == dog_id and distance > 0 and distance <= max_distance:
                nearby_dogs.append((row_dog, distance))
        
        return nearby_dogs
    
    def is_adjacent_group(self, group, nearby_groups):
        """Check if group is adjacent to any nearby groups (your exact logic)"""
        adjacent_map = {1: [2], 2: [1, 3], 3: [2]}
        adjacent_to_group = adjacent_map.get(group, [])
        return any(adj_group in nearby_groups for adj_group in adjacent_to_group)
    
    def calculate_driver_load(self, driver_name: str) -> Dict[str, int]:
        """Calculate current load for a driver across all groups"""
        load = {'group1': 0, 'group2': 0, 'group3': 0}
        
        for dog_data in self.dog_assignments.values():
            if dog_data['current_driver'] == driver_name:
                num_dogs = dog_data['num_dogs']
                for group in dog_data['current_groups']:
                    group_key = f'group{group}'
                    if group_key in load:
                        load[group_key] += num_dogs
        
        return load
    
    def cluster_callout_dogs(self, dogs_to_reassign, max_cluster_distance=0.5):
        """Group callout dogs that are close to each other for efficient batch assignment"""
        print(f"\n🔍 Clustering callout dogs (max distance: {max_cluster_distance} miles)...")
        
        clusters = []
        unassigned_dogs = dogs_to_reassign.copy()
        
        while unassigned_dogs:
            # Start new cluster with first unassigned dog
            cluster_seed = unassigned_dogs[0]
            cluster = [cluster_seed]
            unassigned_dogs.remove(cluster_seed)
            
            # Find all dogs within clustering distance of any dog in this cluster
            cluster_changed = True
            while cluster_changed:
                cluster_changed = False
                dogs_to_remove = []
                
                for candidate_dog in unassigned_dogs:
                    candidate_id = candidate_dog['dog_id']
                    
                    # Check if candidate is close to any dog in current cluster
                    close_to_cluster = False
                    for cluster_dog in cluster:
                        cluster_id = cluster_dog['dog_id']
                        
                        # Get distance between candidate and cluster dog
                        distance = None
                        if (candidate_id, cluster_id) in self.distance_matrix:
                            distance = self.distance_matrix[(candidate_id, cluster_id)]
                        elif (cluster_id, candidate_id) in self.distance_matrix:
                            distance = self.distance_matrix[(cluster_id, candidate_id)]
                        
                        if distance and distance <= max_cluster_distance:
                            close_to_cluster = True
                            break
                    
                    if close_to_cluster:
                        cluster.append(candidate_dog)
                        dogs_to_remove.append(candidate_dog)
                        cluster_changed = True
                
                # Remove dogs that joined the cluster
                for dog in dogs_to_remove:
                    unassigned_dogs.remove(dog)
            
            clusters.append(cluster)
        
        # Print cluster analysis
        print(f"📊 Created {len(clusters)} clusters:")
        for i, cluster in enumerate(clusters):
            dog_ids = [d['dog_id'] for d in cluster]
            callouts = [d['original_callout'] for d in cluster]
            print(f"   Cluster {i+1}: {len(cluster)} dogs - {dog_ids}")
            print(f"      Callouts: {callouts}")
        
        return clusters
    
    def find_cluster_capable_drivers(self, cluster):
        """Find drivers who can handle an entire cluster of dogs - WITH DETAILED DEBUGGING"""
        print(f"\n🔍 Finding drivers capable of handling cluster of {len(cluster)} dogs...")
        
        # DETAILED: Show exactly what each dog in the cluster needs
        print(f"   📋 CLUSTER BREAKDOWN:")
        total_cluster_dogs = 0
        for i, dog in enumerate(cluster):
            print(f"      Dog {i+1}: {dog['dog_name']} ({dog['dog_id']}) - {dog['num_dogs']} dogs, needs groups {dog['needed_groups']}")
            total_cluster_dogs += dog['num_dogs']
        print(f"   📊 CLUSTER TOTALS: {total_cluster_dogs} total dogs")
        
        # Calculate total dogs needed PER GROUP for the cluster
        group_dog_counts = {'group1': 0, 'group2': 0, 'group3': 0}
        all_required_groups = set()
        
        for dog in cluster:
            dog_count = dog['num_dogs']
            for group in dog['needed_groups']:
                group_key = f'group{group}'
                group_dog_counts[group_key] += dog_count
                all_required_groups.add(group)
                print(f"      Adding {dog_count} dogs to {group_key} for {dog['dog_name']}")
        
        required_groups = list(all_required_groups)
        print(f"   📊 CLUSTER GROUP DEMANDS:")
        print(f"      Group 1 needs: {group_dog_counts['group1']} dogs")
        print(f"      Group 2 needs: {group_dog_counts['group2']} dogs") 
        print(f"      Group 3 needs: {group_dog_counts['group3']} dogs")
        print(f"      Required groups: {required_groups}")
        
        # Get drivers working today
        working_drivers = set(data['current_driver'] for data in self.dog_assignments.values())
        
        # Exclude callout drivers
        callout_drivers = set()
        for dog in cluster:
            if ':' in dog['original_callout']:
                callout_driver = dog['original_callout'].split(':', 1)[0].strip()
                callout_drivers.add(callout_driver)
        
        print(f"   🚫 Excluded callout drivers: {callout_drivers}")
        
        capable_drivers = []
        
        for driver in working_drivers:
            if driver in callout_drivers:
                print(f"\n   ❌ SKIPPING {driver} - they called out")
                continue
            
            print(f"\n   🔍 ANALYZING {driver.upper()}:")
            
            # Get current load and capacity
            current_load = self.calculate_driver_load(driver)
            driver_capacity = self.driver_capacities.get(driver, {'group1': 9, 'group2': 9, 'group3': 9})
            
            print(f"      📊 CURRENT STATE:")
            print(f"         Current load: Group1={current_load.get('group1', 0)}, Group2={current_load.get('group2', 0)}, Group3={current_load.get('group3', 0)}")
            print(f"         Driver capacity: Group1={driver_capacity.get('group1', 9)}, Group2={driver_capacity.get('group2', 9)}, Group3={driver_capacity.get('group3', 9)}")
            
            # CRITICAL: Check each group capacity individually
            capacity_violations = []
            
            for group in required_groups:
                group_key = f'group{group}'
                current = current_load.get(group_key, 0)
                capacity = driver_capacity.get(group_key, 9)
                cluster_needs = group_dog_counts[group_key]
                new_total = current + cluster_needs
                
                print(f"      🧮 {group_key.upper()} MATH:")
                print(f"         Current: {current} dogs")
                print(f"         + Cluster: {cluster_needs} dogs") 
                print(f"         = New Total: {new_total} dogs")
                print(f"         Capacity: {capacity} dogs")
                print(f"         Status: {'✅ OK' if new_total <= capacity else '❌ VIOLATION'}")
                
                if new_total > capacity:
                    violation = f"{group_key}: {new_total} > {capacity}"
                    capacity_violations.append(violation)
            
            if capacity_violations:
                print(f"      ❌ CAPACITY VIOLATIONS: {capacity_violations}")
                print(f"      ❌ {driver} REJECTED - insufficient capacity")
                continue
            
            print(f"      ✅ {driver} CAPACITY CHECK PASSED")
            
            # If we get here, capacity is OK - continue with other checks...
            capable_drivers.append({
                'driver': driver,
                'distance_to_cluster': 1.0,  # Simplified for debugging
                'current_load': dict(current_load),
                'projected_load': {
                    'group1': current_load.get('group1', 0) + group_dog_counts['group1'],
                    'group2': current_load.get('group2', 0) + group_dog_counts['group2'],
                    'group3': current_load.get('group3', 0) + group_dog_counts['group3']
                }
            })
            
            print(f"      🎯 PROJECTED FINAL LOAD after cluster assignment:")
            print(f"         Group1: {current_load.get('group1', 0)} + {group_dog_counts['group1']} = {current_load.get('group1', 0) + group_dog_counts['group1']}")
            print(f"         Group2: {current_load.get('group2', 0)} + {group_dog_counts['group2']} = {current_load.get('group2', 0) + group_dog_counts['group2']}")
            print(f"         Group3: {current_load.get('group3', 0)} + {group_dog_counts['group3']} = {current_load.get('group3', 0) + group_dog_counts['group3']}")
        
        print(f"\n   📊 FINAL RESULT: {len(capable_drivers)} drivers can handle this cluster")
        for driver_info in capable_drivers:
            print(f"      ✅ {driver_info['driver']}")
        
        return capable_drivers
    
    def calculate_cluster_total_miles(self, cluster, driver_assignment):
        """Calculate total miles for assigning a cluster to a specific driver"""
        driver = driver_assignment['driver']
        
        # Find the closest dog this driver is already picking up
        driver_dogs = [dog_id for dog_id, data in self.dog_assignments.items() 
                      if data['current_driver'] == driver]
        
        if not driver_dogs:
            return float('inf')  # Driver has no existing pickups
        
        # Calculate distance from driver's existing route to cluster
        min_distance_to_cluster = driver_assignment['distance_to_cluster']
        
        # Calculate internal cluster distances (distance between dogs in cluster)
        internal_cluster_distance = 0
        cluster_dog_ids = [dog['dog_id'] for dog in cluster]
        
        for i, dog_id_1 in enumerate(cluster_dog_ids):
            for dog_id_2 in cluster_dog_ids[i+1:]:
                distance = None
                if (dog_id_1, dog_id_2) in self.distance_matrix:
                    distance = self.distance_matrix[(dog_id_1, dog_id_2)]
                elif (dog_id_2, dog_id_1) in self.distance_matrix:
                    distance = self.distance_matrix[(dog_id_2, dog_id_1)]
                
                if distance:
                    internal_cluster_distance += distance
        
        # Total miles = distance to reach cluster + internal cluster distances
        total_miles = min_distance_to_cluster + internal_cluster_distance
        
        print(f"      📏 {driver}: {min_distance_to_cluster:.1f} mi to cluster + {internal_cluster_distance:.1f} mi internal = {total_miles:.1f} mi total")
        
        return total_miles
    
    def optimize_cluster_assignment(self, cluster):
        """Optimally assign cluster dogs to minimize miles while respecting capacity limits"""
        print(f"\n🎯 OPTIMIZING cluster assignment for {len(cluster)} dogs...")
        
        # Calculate total cluster requirements
        total_cluster_dogs = sum(dog['num_dogs'] for dog in cluster)
        group_requirements = {'group1': 0, 'group2': 0, 'group3': 0}
        
        for dog in cluster:
            for group in dog['needed_groups']:
                group_requirements[f'group{group}'] += dog['num_dogs']
        
        print(f"   Total cluster: {total_cluster_dogs} dogs")
        print(f"   Group requirements: G1={group_requirements['group1']}, G2={group_requirements['group2']}, G3={group_requirements['group3']}")
        
        # Get all potential drivers (excluding callout drivers)
        working_drivers = set(data['current_driver'] for data in self.dog_assignments.values())
        callout_drivers = set()
        for dog in cluster:
            if ':' in dog['original_callout']:
                callout_driver = dog['original_callout'].split(':', 1)[0].strip()
                callout_drivers.add(callout_driver)
        
        available_drivers = working_drivers - callout_drivers
        
        # Calculate each driver's available capacity
        driver_availability = {}
        for driver in available_drivers:
            current_load = self.calculate_driver_load(driver)
            driver_capacity = self.driver_capacities.get(driver, {'group1': 9, 'group2': 9, 'group3': 9})
            
            available_capacity = {}
            for group_key in ['group1', 'group2', 'group3']:
                current = current_load.get(group_key, 0)
                capacity = driver_capacity.get(group_key, 9)
                available = max(0, capacity - current)
                available_capacity[group_key] = available
            
            # Only include drivers who have some available capacity for required groups
            has_useful_capacity = False
            for group in [1, 2, 3]:
                group_key = f'group{group}'
                if group_requirements[group_key] > 0 and available_capacity[group_key] > 0:
                    has_useful_capacity = True
                    break
            
            if has_useful_capacity:
                driver_availability[driver] = available_capacity
                print(f"   {driver}: Available capacity G1={available_capacity['group1']}, G2={available_capacity['group2']}, G3={available_capacity['group3']}")
        
        if not driver_availability:
            print(f"   ❌ No drivers have available capacity for this cluster")
            return []
        
        # STRATEGY 1: Try to assign entire cluster to one driver
        print(f"\n   📦 STRATEGY 1: Single driver assignment")
        for driver, available in driver_availability.items():
            can_take_all = True
            for group in [1, 2, 3]:
                group_key = f'group{group}'
                if group_requirements[group_key] > available[group_key]:
                    can_take_all = False
                    break
            
            if can_take_all:
                print(f"      ✅ {driver} can take entire cluster!")
                return [{
                    'driver': driver,
                    'dogs': cluster.copy(),
                    'assignment_type': 'full_cluster'
                }]
            else:
                print(f"      ❌ {driver} cannot take full cluster")
        
        # STRATEGY 2: Partial assignments - fill drivers optimally
        print(f"\n   🔄 STRATEGY 2: Optimal partial assignments")
        
        assignments = []
        remaining_dogs = cluster.copy()
        
        # Sort drivers by total available capacity (prioritize drivers with more room)
        sorted_drivers = sorted(driver_availability.items(), 
                              key=lambda x: sum(x[1].values()), reverse=True)
        
        for driver, available in sorted_drivers:
            if not remaining_dogs:
                break
            
            print(f"\n      📋 Trying to assign dogs to {driver}:")
            print(f"         Available: G1={available['group1']}, G2={available['group2']}, G3={available['group3']}")
            
            # Find dogs this driver can take
            driver_dogs = []
            
            for dog in remaining_dogs.copy():
                dog_count = dog['num_dogs']
                needed_groups = dog['needed_groups']
                
                # Check if this driver can take this specific dog
                can_take_dog = True
                for group in needed_groups:
                    group_key = f'group{group}'
                    if available[group_key] < dog_count:
                        can_take_dog = False
                        break
                
                if can_take_dog:
                    # Assign this dog to this driver
                    driver_dogs.append(dog)
                    remaining_dogs.remove(dog)
                    
                    # Reduce available capacity
                    for group in needed_groups:
                        group_key = f'group{group}'
                        available[group_key] -= dog_count
                    
                    print(f"         ✅ Assigned {dog['dog_name']} ({dog_count} dogs, groups {needed_groups})")
                    print(f"            Updated available: G1={available['group1']}, G2={available['group2']}, G3={available['group3']}")
                else:
                    print(f"         ❌ Cannot fit {dog['dog_name']} ({dog_count} dogs, groups {needed_groups})")
            
            if driver_dogs:
                assignments.append({
                    'driver': driver,
                    'dogs': driver_dogs,
                    'assignment_type': 'partial_cluster'
                })
                print(f"      🎯 {driver} gets {len(driver_dogs)} dogs total")
        
        if remaining_dogs:
            print(f"\n   ⚠️ {len(remaining_dogs)} dogs could not be assigned:")
            for dog in remaining_dogs:
                print(f"      - {dog['dog_name']} ({dog['num_dogs']} dogs, groups {dog['needed_groups']})")
        
        print(f"\n   📊 FINAL CLUSTER ASSIGNMENT:")
        print(f"      {len(assignments)} drivers will split the cluster")
        total_miles_estimate = len(assignments) * 1.5  # Rough estimate
        print(f"      Estimated total miles: {total_miles_estimate:.1f}")
        
    def optimize_total_system_miles(self, clusters):
        """Find the assignment combination that minimizes total system miles with PARTIAL ASSIGNMENTS"""
        print(f"\n🎯 Optimizing total system miles across {len(clusters)} clusters...")
        
        cluster_assignments = []
        total_system_miles = 0
        
        for i, cluster in enumerate(clusters):
            print(f"\n📍 Optimizing Cluster {i+1} ({len(cluster)} dogs):")
            
            # Use new smart partial assignment
            optimal_assignments = self.optimize_cluster_assignment(cluster)
            
            if optimal_assignments:
                cluster_miles = 0
                cluster_assignment = {
                    'cluster': cluster,
                    'type': 'optimized',
                    'driver_assignments': optimal_assignments,
                    'total_miles': cluster_miles
                }
                
                for assignment in optimal_assignments:
                    driver = assignment['driver']
                    dogs = assignment['dogs']
                    assignment_type = assignment['assignment_type']
                    
                    print(f"   ✅ {driver} gets {len(dogs)} dogs [{assignment_type}]:")
                    for dog in dogs:
                        print(f"      - {dog['dog_name']} ({dog['num_dogs']} dogs, groups {dog['needed_groups']})")
                    
                    # Estimate miles for this driver (simplified)
                    cluster_miles += 1.5  # Base distance per driver
                
                cluster_assignment['total_miles'] = cluster_miles
                cluster_assignments.append(cluster_assignment)
                total_system_miles += cluster_miles
                
                print(f"   📏 Cluster {i+1} total miles: {cluster_miles:.1f}")
            else:
                print(f"   ❌ Could not optimize cluster {i+1} - no drivers available")
                # Could add fallback to individual assignment here
        
        print(f"\n📊 TOTAL OPTIMIZED SYSTEM MILES: {total_system_miles:.1f}")
        
        return cluster_assignments, total_system_miles
        """Find the assignment combination that minimizes total system miles"""
        print(f"\n🎯 Optimizing total system miles across {len(clusters)} clusters...")
        
        cluster_assignments = []
        total_system_miles = 0
        
        for i, cluster in enumerate(clusters):
            print(f"\n📍 Optimizing Cluster {i+1} ({len(cluster)} dogs):")
            
            capable_drivers = self.find_cluster_capable_drivers(cluster)
            
            if not capable_drivers:
                print(f"   ❌ No drivers can handle this cluster - falling back to individual assignment")
                # Fall back to individual assignment for this cluster
                individual_assignments = []
                for dog in cluster:
                    assignment = self.find_best_reassignment(dog)
                    if assignment:
                        individual_assignments.append({
                            'type': 'individual',
                            'dog': dog,
                            'assignment': assignment
                        })
                
                cluster_assignments.append({
                    'cluster': cluster,
                    'type': 'individual',
                    'assignments': individual_assignments
                })
                continue
            
            # Calculate total miles for each capable driver
            best_driver = None
            best_miles = float('inf')
            
            for driver_assignment in capable_drivers:
                total_miles = self.calculate_cluster_total_miles(cluster, driver_assignment)
                
                if total_miles < best_miles:
                    best_miles = total_miles
                    best_driver = driver_assignment
            
            if best_driver:
                print(f"   🏆 Best choice: {best_driver['driver']} ({best_miles:.1f} total miles)")
                
                cluster_assignments.append({
                    'cluster': cluster,
                    'type': 'cluster',
                    'driver': best_driver['driver'],
                    'total_miles': best_miles
                })
                
                total_system_miles += best_miles
            else:
                print(f"   ❌ No valid cluster assignment found")
        
        print(f"\n📊 TOTAL SYSTEM MILES: {total_system_miles:.1f}")
        
        return cluster_assignments, total_system_miles
    
    def find_best_reassignment(self, dog_data: Dict, max_iterations=5) -> Dict:
        """Find best individual reassignment for a dog (fallback when clustering fails)"""
        dog_id = dog_data['dog_id']
        needed_groups = dog_data['needed_groups']
        num_dogs = dog_data['num_dogs']
        
        # Parse the callout driver name (e.g., "Nate:1&2" → "Nate")
        callout_driver = None
        if ':' in dog_data['original_callout']:
            callout_driver = dog_data['original_callout'].split(':', 1)[0].strip()
        
        print(f"🔍 Individual assignment for {dog_data['dog_name']} ({dog_id}) - needs groups {needed_groups}")
        
        # HARD CONSTRAINT 1: Maximum 3 miles distance
        nearby_dogs = self.get_nearby_dogs(dog_id, max_distance=3.0)
        
        if not nearby_dogs:
            print(f"   No nearby dogs within 3 miles for {dog_id}")
            return None
        
        valid_candidates = []
        
        for nearby_dog_id, distance in nearby_dogs:
            if nearby_dog_id not in self.dog_assignments:
                continue
            
            nearby_dog_data = self.dog_assignments[nearby_dog_id]
            candidate_driver = nearby_dog_data['current_driver']
            nearby_groups = nearby_dog_data['current_groups']
            
            # HARD CONSTRAINT 2: Skip the driver who called out
            if callout_driver and candidate_driver == callout_driver:
                continue
            
            # HARD CONSTRAINT 3: Groups MUST match (exact or adjacent)
            exact_match = any(group in nearby_groups for group in needed_groups)
            adjacent_match = any(self.is_adjacent_group(group, nearby_groups) for group in needed_groups)
            
            if not (exact_match or adjacent_match):
                continue
            
            # HARD CONSTRAINT 4: Must have capacity
            current_load = self.calculate_driver_load(candidate_driver)
            driver_capacity = self.driver_capacities.get(candidate_driver, {'group1': 9, 'group2': 9, 'group3': 9})
            
            capacity_ok = True
            for group in needed_groups:
                group_key = f'group{group}'
                current = current_load.get(group_key, 0)
                capacity = driver_capacity.get(group_key, 9)
                if current + num_dogs > capacity:
                    capacity_ok = False
                    break
            
            if not capacity_ok:
                continue
            
            # Calculate preference score
            score = 10 if exact_match else 5
            score -= distance * 4
            if distance <= 0.5:
                score += (0.5 - distance) * 10
            
            total_load = sum(current_load.values())
            final_score = score - (total_load * 0.1)
            
            valid_candidates.append({
                'to_driver': candidate_driver,
                'to_groups': needed_groups,
                'distance': distance,
                'score': final_score
            })
        
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x['score'], reverse=True)
            return valid_candidates[0]
        
        return None
        """Find best reassignment for a callout dog with HARD constraints"""
        dog_id = dog_data['dog_id']
        needed_groups = dog_data['needed_groups']
        num_dogs = dog_data['num_dogs']
        
        # Parse the callout driver name (e.g., "Nate:1&2" → "Nate")
        callout_driver = None
        if ':' in dog_data['original_callout']:
            callout_driver = dog_data['original_callout'].split(':', 1)[0].strip()
        
        print(f"🔍 Finding replacement for {dog_data['dog_name']} ({dog_id}) - needs groups {needed_groups}")
        print(f"   Excluding callout driver: {callout_driver}")
        
        for iteration in range(max_iterations):
            print(f"   Iteration {iteration + 1}:")
            
            # HARD CONSTRAINT 1: Maximum 3 miles distance
            nearby_dogs = self.get_nearby_dogs(dog_id, max_distance=3.0)
            
            if not nearby_dogs:
                print(f"   No nearby dogs within 3 miles for {dog_id}")
                continue
            
            valid_candidates = []
            
            for nearby_dog_id, distance in nearby_dogs:
                if nearby_dog_id not in self.dog_assignments:
                    continue
                
                nearby_dog_data = self.dog_assignments[nearby_dog_id]
                candidate_driver = nearby_dog_data['current_driver']
                nearby_groups = nearby_dog_data['current_groups']
                
                # HARD CONSTRAINT 2: Skip the driver who called out
                if callout_driver and candidate_driver == callout_driver:
                    print(f"     ❌ Skipping {candidate_driver} - they called out!")
                    continue
                
                # HARD CONSTRAINT 3: Only consider drivers working today
                if candidate_driver not in [d['current_driver'] for d in self.dog_assignments.values()]:
                    print(f"     ❌ Skipping {candidate_driver} - not working today")
                    continue
                
                # HARD CONSTRAINT 4: Groups MUST match (exact or adjacent)
                exact_match = any(group in nearby_groups for group in needed_groups)
                adjacent_match = any(self.is_adjacent_group(group, nearby_groups) for group in needed_groups)
                
                if not (exact_match or adjacent_match):
                    print(f"     ❌ Skipping {candidate_driver} - no group match. Has {nearby_groups}, needs {needed_groups}")
                    continue
                
                # HARD CONSTRAINT 5: Must have capacity
                current_load = self.calculate_driver_load(candidate_driver)
                driver_capacity = self.driver_capacities.get(candidate_driver, {'group1': 9, 'group2': 9, 'group3': 9})
                
                capacity_ok = True
                for group in needed_groups:
                    group_key = f'group{group}'
                    current = current_load.get(group_key, 0)
                    capacity = driver_capacity.get(group_key, 9)
                    if current + num_dogs > capacity:
                        print(f"     ❌ Skipping {candidate_driver} - no capacity for group {group}. Current: {current}, Capacity: {capacity}, Need: {num_dogs}")
                        capacity_ok = False
                        break
                
                if not capacity_ok:
                    continue
                
                print(f"     ✅ {candidate_driver} is valid candidate")
                
                # All hard constraints passed - now calculate preference score
                score = 0
                
                # Prefer exact group matches over adjacent
                if exact_match:
                    score += 10
                    print(f"       🎯 Exact group match: +10")
                elif adjacent_match:
                    score += 5
                    print(f"       🔗 Adjacent group match: +5")
                
                # Distance penalty
                score -= distance * 4
                print(f"       📏 Distance penalty: -{distance * 4:.1f} ({distance:.1f} miles)")
                
                # Proximity bonus for very close drivers
                if distance <= 0.5:
                    proximity_bonus = (0.5 - distance) * 10
                    score += proximity_bonus
                    print(f"       🎯 Proximity bonus: +{proximity_bonus:.1f} (only {distance:.1f} miles away)")
                
                # Load penalty - prefer less loaded drivers
                total_load = sum(current_load.values())
                load_penalty = total_load * 0.1
                final_score = score - load_penalty
                print(f"       ⚖️ Load penalty: -{load_penalty:.1f} (total load: {total_load})")
                print(f"       🏆 Final score: {final_score:.2f}")
                
                valid_candidates.append({
                    'to_driver': candidate_driver,
                    'to_groups': needed_groups,
                    'distance': distance,
                    'score': final_score,
                    'exact_match': exact_match
                })
            
            if valid_candidates:
                # Sort by score (higher is better)
                valid_candidates.sort(key=lambda x: x['score'], reverse=True)
                best_candidate = valid_candidates[0]
                
                print(f"   ✅ Found {len(valid_candidates)} valid candidates")
                print(f"   🏆 Best choice: {best_candidate['to_driver']} (score: {best_candidate['score']:.2f})")
                return best_candidate
            
            print(f"   ❌ No valid candidates in iteration {iteration + 1}")
        
        print(f"   ❌ No assignment found after {max_iterations} iterations")
        return None
    
    def _get_reassignment_priority(self, dog_data: Dict) -> int:
        """Calculate priority for reassignment (your exact logic)"""
        num_dogs = dog_data['num_dogs']
        num_groups = len(dog_data['needed_groups'])
        
        if num_dogs == 1 and num_groups == 1:
            return 1  # Highest priority
        elif num_dogs == 1 and num_groups > 1:
            return 2
        elif num_dogs > 1 and num_groups == 1:
            return 3
        else:
            return 4  # Lowest priority
    
    def reassign_dogs(self):
        """Main reassignment algorithm with SMART PARTIAL CLUSTER ASSIGNMENTS"""
        print("\n🔄 Starting SMART OPTIMIZED dog reassignment process...")
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts found - all dogs have drivers assigned!")
            return []
        
        print(f"🚨 Found {len(dogs_to_reassign)} callouts to process")
        
        # STEP 1: Cluster nearby dogs for batch processing
        clusters = self.cluster_callout_dogs(dogs_to_reassign)
        
        # STEP 2: Optimize with partial assignments
        cluster_assignments, total_system_miles = self.optimize_total_system_miles(clusters)
        
        # STEP 3: Convert optimized assignments to standard format
        successful_reassignments = []
        
        for cluster_assignment in cluster_assignments:
            if cluster_assignment['type'] == 'optimized':
                # Process each driver assignment within the cluster
                for driver_assignment in cluster_assignment['driver_assignments']:
                    driver = driver_assignment['driver']
                    dogs = driver_assignment['dogs']
                    assignment_type = driver_assignment['assignment_type']
                    
                    for dog_data in dogs:
                        # Create full reassignment record
                        full_reassignment = {
                            'dog_id': dog_data['dog_id'],
                            'dog_name': dog_data['dog_name'],
                            'address': dog_data['address'],
                            'num_dogs': dog_data['num_dogs'],
                            'original_callout': dog_data['original_callout'],
                            'to_driver': driver,
                            'to_groups': dog_data['needed_groups'],
                            'distance': 1.5,  # Simplified for now
                            'score': 1000,  # High score for optimized assignment
                            'assignment_type': assignment_type
                        }
                        
                        # Update internal state
                        new_assignment_str = f"{driver}:{'&'.join(map(str, dog_data['needed_groups']))}"
                        self.dog_assignments[dog_data['dog_id']] = {
                            'dog_name': dog_data['dog_name'],
                            'address': dog_data['address'],
                            'num_dogs': dog_data['num_dogs'],
                            'current_driver': driver,
                            'current_groups': dog_data['needed_groups'],
                            'assignment_str': new_assignment_str
                        }
                        
                        successful_reassignments.append(full_reassignment)
                        print(f"    ✅ SMART ASSIGNED: {dog_data['dog_name']} → {driver}:{'&'.join(map(str, dog_data['needed_groups']))} [{assignment_type}]")
        
        print(f"\n📊 SMART OPTIMIZED REASSIGNMENT SUMMARY:")
        print(f"   🎯 TOTAL SYSTEM MILES MINIMIZED: {total_system_miles:.1f}")
        print(f"   ✅ Successfully assigned: {len(successful_reassignments)}")
        
        # Group results by driver to show capacity utilization
        driver_assignments = {}
        for r in successful_reassignments:
            driver = r['to_driver']
            if driver not in driver_assignments:
                driver_assignments[driver] = []
            driver_assignments[driver].append(r)
        
        print(f"\n🎉 CAPACITY-RESPECTING RESULTS:")
        for driver, assignments in driver_assignments.items():
            dog_count = sum(a['num_dogs'] for a in assignments)
            dog_names = [a['dog_name'] for a in assignments]
            print(f"      {driver}: {len(assignments)} dogs ({dog_count} total) - {', '.join(dog_names)}")
            
            # Show final capacity usage
            final_load = self.calculate_driver_load(driver)
            capacity = self.driver_capacities.get(driver, {'group1': 9, 'group2': 9, 'group3': 9})
            print(f"         Final load: G1={final_load.get('group1', 0)}/{capacity.get('group1', 9)}, G2={final_load.get('group2', 0)}/{capacity.get('group2', 9)}, G3={final_load.get('group3', 0)}/{capacity.get('group3', 9)}")
        
        # Check for any unassigned dogs
        assigned_dog_ids = set(r['dog_id'] for r in successful_reassignments)
        all_callout_dog_ids = set(dog['dog_id'] for dog in dogs_to_reassign)
        unassigned_dog_ids = all_callout_dog_ids - assigned_dog_ids
        
        if unassigned_dog_ids:
            print(f"\n⚠️  UNASSIGNED DOGS:")
            for dog in dogs_to_reassign:
                if dog['dog_id'] in unassigned_dog_ids:
                    print(f"      - {dog['dog_name']} ({dog['dog_id']}) - needed {dog['original_callout']}")
        
    def optimize_driver_routes(self, reassignments):
        """Post-process assignments to move outlier dogs to better routes"""
        print(f"\n🔄 ROUTE OPTIMIZATION: Checking for outlier dogs that could be better placed...")
        
        if not reassignments:
            return reassignments
        
        # Get all drivers who received new assignments
        drivers_with_new_dogs = set(r['to_driver'] for r in reassignments)
        
        route_improvements = []
        
        for driver in drivers_with_new_dogs:
            print(f"\n🔍 Analyzing {driver}'s route for outliers...")
            
            # Get all dogs assigned to this driver (including existing + new ones)
            driver_dogs = []
            for dog_id, data in self.dog_assignments.items():
                if data['current_driver'] == driver:
                    driver_dogs.append({
                        'dog_id': dog_id,
                        'dog_name': data['dog_name'],
                        'num_dogs': data['num_dogs'],
                        'groups': data['current_groups']
                    })
            
            if len(driver_dogs) < 3:
                print(f"   {driver} has only {len(driver_dogs)} dogs - skipping outlier analysis")
                continue
            
            print(f"   {driver} has {len(driver_dogs)} total dogs")
            
            # Calculate distances between all pairs of dogs for this driver
            dog_distances = {}
            for i, dog1 in enumerate(driver_dogs):
                dog_distances[dog1['dog_id']] = {}
                for j, dog2 in enumerate(driver_dogs):
                    if i != j:
                        distance = self.get_distance_between_dogs(dog1['dog_id'], dog2['dog_id'])
                        dog_distances[dog1['dog_id']][dog2['dog_id']] = distance
            
            # For each dog, calculate its average distance to all other dogs in the route
            dog_avg_distances = {}
            for dog in driver_dogs:
                dog_id = dog['dog_id']
                distances_to_others = [d for d in dog_distances[dog_id].values() if d is not None and d > 0]
                if distances_to_others:
                    avg_distance = sum(distances_to_others) / len(distances_to_others)
                    dog_avg_distances[dog_id] = avg_distance
                    print(f"      {dog['dog_name']}: avg {avg_distance:.1f} miles from other dogs")
            
            # Identify outliers (dogs with avg distance > 2.0 miles from others)
            outliers = []
            for dog_id, avg_dist in dog_avg_distances.items():
                if avg_dist > 2.0:  # Threshold for "outlier"
                    dog_info = next(d for d in driver_dogs if d['dog_id'] == dog_id)
                    outliers.append({
                        'dog_id': dog_id,
                        'dog_info': dog_info,
                        'avg_distance_from_route': avg_dist
                    })
                    print(f"      🚨 OUTLIER: {dog_info['dog_name']} is {avg_dist:.1f} miles from other dogs")
            
            if not outliers:
                print(f"   ✅ No outliers found in {driver}'s route")
                continue
            
            # For each outlier, check if another driver could serve it better
            for outlier in outliers:
                print(f"\n      🔍 Checking if {outlier['dog_info']['dog_name']} could be better placed...")
                
                best_alternative = self.find_better_driver_for_outlier(
                    outlier, driver, drivers_with_new_dogs
                )
                
                if best_alternative:
                    route_improvements.append({
                        'from_driver': driver,
                        'to_driver': best_alternative['driver'],
                        'dog': outlier,
                        'miles_saved': best_alternative['miles_saved'],
                        'reason': best_alternative['reason']
                    })
                    print(f"         ✅ IMPROVEMENT: Move to {best_alternative['driver']} (saves {best_alternative['miles_saved']:.1f} miles)")
                else:
                    print(f"         ❌ No better placement found")
        
        # Apply the route improvements
        if route_improvements:
            print(f"\n🎯 APPLYING {len(route_improvements)} ROUTE IMPROVEMENTS:")
            
            for improvement in route_improvements:
                from_driver = improvement['from_driver']
                to_driver = improvement['to_driver']
                dog = improvement['dog']
                miles_saved = improvement['miles_saved']
                
                # Update the assignment
                dog_id = dog['dog_id']
                dog_info = dog['dog_info']
                
                # Update internal state
                new_assignment_str = f"{to_driver}:{'&'.join(map(str, dog_info['groups']))}"
                self.dog_assignments[dog_id]['current_driver'] = to_driver
                self.dog_assignments[dog_id]['assignment_str'] = new_assignment_str
                
                # Update reassignments list
                for r in reassignments:
                    if r['dog_id'] == dog_id:
                        r['to_driver'] = to_driver
                        r['assignment_type'] = 'route_optimized'
                        break
                
                print(f"      ✅ {dog_info['dog_name']}: {from_driver} → {to_driver} (saves {miles_saved:.1f} miles)")
            
            total_miles_saved = sum(imp['miles_saved'] for imp in route_improvements)
            print(f"   🎉 TOTAL MILES SAVED: {total_miles_saved:.1f}")
        else:
            print(f"\n✅ No route improvements found - current assignments are optimal")
        
        return reassignments
    
    def get_distance_between_dogs(self, dog_id_1, dog_id_2):
        """Get distance between two dogs from the distance matrix"""
        if (dog_id_1, dog_id_2) in self.distance_matrix:
            return self.distance_matrix[(dog_id_1, dog_id_2)]
        elif (dog_id_2, dog_id_1) in self.distance_matrix:
            return self.distance_matrix[(dog_id_2, dog_id_1)]
        return None
    
    def find_better_driver_for_outlier(self, outlier, current_driver, drivers_with_new_dogs):
        """Check if another driver could serve this outlier dog more efficiently"""
        dog_id = outlier['dog_id']
        dog_info = outlier['dog_info']
        current_avg_distance = outlier['avg_distance_from_route']
        
        # Get all drivers working today (excluding the current driver)
        potential_drivers = set(data['current_driver'] for data in self.dog_assignments.values())
        potential_drivers.discard(current_driver)
        
        best_alternative = None
        best_miles_saved = 0
        
        for candidate_driver in potential_drivers:
            # Check if this driver has capacity
            current_load = self.calculate_driver_load(candidate_driver)
            driver_capacity = self.driver_capacities.get(candidate_driver, {'group1': 9, 'group2': 9, 'group3': 9})
            
            # Check capacity for this dog's groups
            has_capacity = True
            for group in dog_info['groups']:
                group_key = f'group{group}'
                current = current_load.get(group_key, 0)
                capacity = driver_capacity.get(group_key, 9)
                if current + dog_info['num_dogs'] > capacity:
                    has_capacity = False
                    break
            
            if not has_capacity:
                continue
            
            # Get this driver's current dogs
            candidate_driver_dogs = []
            for existing_dog_id, data in self.dog_assignments.items():
                if data['current_driver'] == candidate_driver:
                    candidate_driver_dogs.append(existing_dog_id)
            
            if not candidate_driver_dogs:
                continue  # Skip drivers with no dogs
            
            # Calculate how well this dog fits with the candidate driver's route
            distances_to_candidate_dogs = []
            for existing_dog_id in candidate_driver_dogs:
                distance = self.get_distance_between_dogs(dog_id, existing_dog_id)
                if distance is not None and distance > 0:
                    distances_to_candidate_dogs.append(distance)
            
            if not distances_to_candidate_dogs:
                continue
            
            # Calculate average distance to candidate driver's dogs
            avg_distance_to_candidate = sum(distances_to_candidate_dogs) / len(distances_to_candidate_dogs)
            
            # Calculate miles saved
            miles_saved = current_avg_distance - avg_distance_to_candidate
            
            print(f"         {candidate_driver}: avg {avg_distance_to_candidate:.1f} miles, saves {miles_saved:.1f} miles")
            
            # Only consider if it saves significant miles (> 0.5 miles)
            if miles_saved > 0.5 and miles_saved > best_miles_saved:
                best_miles_saved = miles_saved
                best_alternative = {
                    'driver': candidate_driver,
                    'miles_saved': miles_saved,
                    'avg_distance': avg_distance_to_candidate,
                    'reason': f"Closer to {candidate_driver}'s route"
                }
        
        return best_alternative
    
    def update_combined_column(self, reassignments: List[Dict], spreadsheet) -> bool:
        """Write new assignments to Combined column (H) in Google Sheets"""
        try:
            if not reassignments:
                print("ℹ️ No reassignments to write")
                return True
            
            print(f"📝 Writing {len(reassignments)} assignments to Combined column...")
            
            # Try different sheet names that might exist
            sheet_names_to_try = ["New districts Map 8", "Map", "Districts Map", "Sheet1"]
            sheet = None
            
            for sheet_name in sheet_names_to_try:
                try:
                    sheet = spreadsheet.worksheet(sheet_name)
                    print(f"✅ Found sheet: {sheet_name}")
                    break
                except:
                    continue
            
            if not sheet:
                print("❌ Could not find map sheet")
                return False
            
            # Get all data to find row numbers for dog IDs
            all_data = sheet.get_all_values()
            
            # Find Dog ID column (should be J = column 10)
            dog_id_col = 10  # Column J (1-based)
            
            # Prepare batch updates
            updates = []
            
            for r in reassignments:
                dog_id = r['dog_id']
                new_assignment = f"{r['to_driver']}:{'&'.join(map(str, r['to_groups']))}"
                
                # Find the row for this dog (Dog ID is in column J = index 9)
                row_number = None
                for row_idx, row in enumerate(all_data):
                    if len(row) > 9 and str(row[9]).strip() == str(dog_id).strip():
                        row_number = row_idx + 1  # Convert to 1-based
                        break
                
                if row_number:
                    # Combined column is H = column 8 (1-based)
                    updates.append({
                        'range': f'H{row_number}',
                        'values': [[new_assignment]]
                    })
                    print(f"   📝 Will update row {row_number}: {dog_id} → {new_assignment}")
                else:
                    print(f"   ⚠️ Could not find row for dog {dog_id}")
            
            if updates:
                # Batch update all changes
                sheet.batch_update(updates)
                print(f"✅ Successfully updated {len(updates)} assignments in Combined column")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating Combined column: {e}")
            return False
    
    def write_results_to_sheets(self, reassignments: List[Dict]) -> bool:
        """Write reassignment results back to Google Sheets"""
        try:
            if not reassignments:
                print("ℹ️ No callout assignments to write back")
                return True
            
            print("📤 Writing results to Google Sheets...")
            
            client = self.setup_google_sheets_client()
            if not client:
                return False
            
            spreadsheet = client.open_by_key(self.MAP_SPREADSHEET_ID)
            
            # Update the Combined column with new assignments
            success = self.update_combined_column(reassignments, spreadsheet)
            
            if success:
                print("✅ Successfully wrote callout assignments to Google Sheets")
                
                # Send notification if webhook available
                webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
                if webhook_url:
                    self.send_slack_notification(reassignments, webhook_url)
            
            return success
            
        except Exception as e:
            print(f"❌ Error writing results to sheets: {e}")
            return False
    
    def send_slack_notification(self, reassignments: List[Dict], webhook_url: str):
        """Send Slack notification about callout assignments"""
        try:
            if not reassignments:
                return
            
            message = f"🚨 *Dog Callout Assignments Complete* 🚨\n\n"
            message += f"Assigned drivers for {len(reassignments)} callouts:\n\n"
            
            for r in reassignments:
                groups_str = '&'.join(map(str, r['to_groups']))
                message += f"• {r['dog_name']} ({r['dog_id']}) → {r['to_driver']}:{groups_str}\n"
            
            message += f"\n✅ All assignments updated in Google Sheets"
            
            payload = {"text": message}
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code == 200:
                print("✅ Slack notification sent")
            else:
                print(f"⚠️ Slack notification failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Error sending Slack notification: {e}")


def main():
    """Main function for GitHub Actions"""
    print("🚀 Production Dog Reassignment System")
    print("=" * 50)
    
    # Check M1 trigger value if provided
    trigger_value = os.environ.get('TRIGGER_VALUE')
    if trigger_value:
        print(f"🎯 Triggered by M1 cell change: {trigger_value}")
    
    # Initialize system
    system = ProductionDogReassignmentSystem()
    
    # Load all data
    print("\n📊 Loading data from Google Sheets...")
    
    if not system.load_distance_matrix():
        print("❌ Failed to load distance matrix")
        return
    
    if not system.load_dog_assignments():
        print("❌ Failed to load dog assignments")
        return
    
    if not system.load_driver_capacities():
        print("❌ Failed to load driver capacities")
        return
    
    # Data validation
    print("\n🔍 Data validation:")
    matrix_dogs = set()
    for (row_dog, col_dog) in system.distance_matrix.keys():
        matrix_dogs.add(row_dog)
        matrix_dogs.add(col_dog)
    
    assignment_dogs = set(system.dog_assignments.keys())
    callout_dogs = set([c['dog_id'] for c in system.callout_dogs])
    all_dogs = assignment_dogs.union(callout_dogs)
    
    matching_dogs = matrix_dogs.intersection(all_dogs)
    
    print(f"   Matrix dogs: {len(matrix_dogs)}")
    print(f"   Assignment dogs: {len(assignment_dogs)}")
    print(f"   Callout dogs: {len(callout_dogs)}")
    print(f"   Matching dogs: {len(matching_dogs)}")
    
    if len(matching_dogs) == 0:
        print("❌ NO MATCHING DOG IDs! Check that Dog IDs match between sheets.")
        print("Error: Process completed with exit code 1.")
        return
    
    # Check for callouts
    if not system.callout_dogs:
        print("\n✅ No callouts detected - all dogs have drivers assigned!")
        print("🎯 System ready - will process callouts when M1 changes")
        return
    
    # Run your exact reassignment logic
    print("\n🔄 Processing callout assignments...")
    reassignments = system.reassign_dogs()
    
    # Write results back to Google Sheets
    if system.write_results_to_sheets(reassignments):
        print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments")
    else:
        print(f"\n❌ Failed to write results to Google Sheets")


if __name__ == "__main__":
    main()
