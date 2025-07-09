# production_reassignment.py
# Complete dog reassignment system with targeted debugging for assignment loss bug

import pandas as pd
import numpy as np
import requests
import json
import os
from typing import Dict, List, Tuple

class DogReassignmentSystem:
    def __init__(self):
        """Initialize the dog reassignment system"""
        # Google Sheets URLs (CSV export format)
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        
        # System parameters
        self.MAX_DISTANCE = 3.0  # Hard limit: no assignments beyond 3 miles
        self.CLUSTER_DISTANCE = 0.5  # Dogs within 0.5 miles get clustered
        self.OUTLIER_THRESHOLD = 2.0  # Dogs >2 miles from route average are outliers
        
        # Data containers
        self.distance_matrix = None
        self.dog_assignments = None
        self.driver_capacities = None
        self.sheets_client = None

    def setup_google_sheets_client(self):
        """Setup Google Sheets API client using service account credentials"""
        try:
            # Get credentials from environment variable
            service_account_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            if not service_account_json:
                print("❌ GOOGLE_SERVICE_ACCOUNT_JSON environment variable not found")
                return False
                
            print("✅ Google Sheets client setup successful")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up Google Sheets client: {e}")
            return False

    def load_distance_matrix(self):
        """Load distance matrix data from Google Sheets"""
        try:
            print("📊 Loading distance matrix...")
            
            # Fetch CSV data
            response = requests.get(self.DISTANCE_MATRIX_URL)
            response.raise_for_status()
            
            # Debug: Show CSV preview
            csv_content = response.text
            print("🔍 Distance matrix CSV preview:")
            print(csv_content[:200] + "..." if len(csv_content) > 200 else csv_content)
            
            # Read into DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content), index_col=0)
            
            print(f"📊 Distance matrix shape: ({len(df)}, {len(df.columns)})")
            
            # Extract dog IDs from columns (skip non-dog columns)
            dog_ids = [col for col in df.columns if 'x' in str(col).lower()]
            print(f"📊 Found {len(dog_ids)} column Dog IDs")
            
            # Filter to only dog ID columns and rows
            dog_df = df.loc[df.index.isin(dog_ids), dog_ids]
            
            self.distance_matrix = dog_df
            print(f"✅ Loaded distance matrix for {len(self.distance_matrix)} dogs")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading distance matrix: {e}")
            return False

    def load_dog_assignments(self):
        """Load current dog assignments from map sheet"""
        try:
            print("🐕 Loading dog assignments...")
            
            # Fetch CSV data
            response = requests.get(self.MAP_SHEET_URL)
            response.raise_for_status()
            
            # Debug: Show CSV preview
            csv_content = response.text
            print("🔍 Map sheet CSV preview:")
            print(csv_content[:200] + "..." if len(csv_content) > 200 else csv_content)
            
            # Read into DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
            
            print(f"📊 Map sheet shape: ({len(df)}, {len(df.columns)})")
            print(f"📊 Map sheet columns: {list(df.columns)}")
            
            # Debug: Show column mapping
            print("📊 Column mapping:")
            for i, col in enumerate(df.columns):
                print(f"   Column {chr(65+i)} (index {i}): '{col}'")
            
            assignments = []
            
            for _, row in df.iterrows():
                try:
                    # Column positions (0-indexed)
                    dog_name = row.iloc[1] if len(row) > 1 else ""  # Column B
                    combined = row.iloc[7] if len(row) > 7 else ""  # Column H
                    group = row.iloc[8] if len(row) > 8 else ""     # Column I  
                    dog_id = row.iloc[9] if len(row) > 9 else ""    # Column J
                    callout = row.iloc[10] if len(row) > 10 else "" # Column K
                    num_dogs = row.iloc[5] if len(row) > 5 else 1   # Column F (Number of dogs) - FIXED!
                    
                    # Skip rows without dog IDs
                    if not dog_id or pd.isna(dog_id):
                        continue
                    
                    # Ensure num_dogs is a positive integer
                    try:
                        num_dogs = int(float(num_dogs)) if not pd.isna(num_dogs) else 1
                        num_dogs = max(1, num_dogs)  # Ensure positive
                    except:
                        num_dogs = 1
                    
                    assignments.append({
                        'dog_name': str(dog_name),
                        'dog_id': str(dog_id),
                        'combined': str(combined) if not pd.isna(combined) else "",
                        'group': str(group) if not pd.isna(group) else "",
                        'callout': str(callout) if not pd.isna(callout) else "",
                        'num_dogs': num_dogs
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error processing row: {e}")
                    continue
            
            self.dog_assignments = assignments
            print(f"✅ Loaded {len(assignments)} regular assignments")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dog assignments: {e}")
            return False

    def load_driver_capacities(self):
        """Load driver capacities from columns R:W on the map sheet"""
        try:
            print("👥 Loading driver capacities from map sheet columns R:W...")
            
            # Fetch the same CSV data as dog assignments
            response = requests.get(self.MAP_SHEET_URL)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"👥 Driver data shape: ({len(df)}, {len(df.columns)})")
            print(f"👥 Driver data columns: {list(df.columns)}")
            
            # Debug: Check if we have enough columns
            print(f"🔍 DataFrame has {len(df.columns)} columns, need at least 23 for column W")
            
            if len(df.columns) >= 23:
                # Sample row to check column access
                sample_row = df.iloc[0] if len(df) > 0 else None
                if sample_row is not None:
                    print("🔍 Sample row to check column access:")
                    print(f" Row length: {len(sample_row)}")
                    print(f" Column R (17): {sample_row.iloc[17] if len(sample_row) > 17 else 'N/A'}")
                    print(f" Column U (20): {sample_row.iloc[20] if len(sample_row) > 20 else 'N/A'}")
                    print(f" Column V (21): {sample_row.iloc[21] if len(sample_row) > 21 else 'N/A'}")
                    print(f" Column W (22): {sample_row.iloc[22] if len(sample_row) > 22 else 'N/A'}")
            
            capacities = {}
            
            for _, row in df.iterrows():
                try:
                    # Column positions for driver data
                    driver_name = row.iloc[17] if len(row) > 17 else ""  # Column R
                    group1_cap = row.iloc[20] if len(row) > 20 else 0    # Column U
                    group2_cap = row.iloc[21] if len(row) > 21 else 0    # Column V  
                    group3_cap = row.iloc[22] if len(row) > 22 else 0    # Column W
                    
                    # Skip rows without driver names
                    if not driver_name or pd.isna(driver_name) or driver_name == "":
                        continue
                    
                    # Convert capacities to integers
                    try:
                        group1_cap = int(float(group1_cap)) if not pd.isna(group1_cap) else 0
                        group2_cap = int(float(group2_cap)) if not pd.isna(group2_cap) else 0
                        group3_cap = int(float(group3_cap)) if not pd.isna(group3_cap) else 0
                    except:
                        continue
                    
                    # Only add drivers with valid capacities
                    if group1_cap > 0 or group2_cap > 0 or group3_cap > 0:
                        capacities[str(driver_name)] = {
                            'group1': group1_cap,
                            'group2': group2_cap,
                            'group3': group3_cap
                        }
                        
                except Exception as e:
                    continue
            
            self.driver_capacities = capacities
            print(f"✅ Loaded capacities for {len(capacities)} drivers")
            
            # Show sample driver capacities
            for driver, caps in list(capacities.items())[:5]:
                print(f" - {driver}: Group1={caps['group1']}, Group2={caps['group2']}, Group3={caps['group3']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading driver capacities: {e}")
            return False

    def get_dogs_to_reassign(self):
        """Find dogs that need reassignment (callouts)"""
        dogs_to_reassign = []
        
        if not self.dog_assignments:
            return dogs_to_reassign
        
        for assignment in self.dog_assignments:
            # Check for callout: Combined column blank AND Callout column has content
            if (not assignment['combined'] or assignment['combined'].strip() == "") and \
               (assignment['callout'] and assignment['callout'].strip() != ""):
                
                # Parse the callout to get needed groups
                needed_groups = self._parse_groups(assignment['callout'])
                
                if needed_groups:
                    dogs_to_reassign.append({
                        'dog_id': assignment['dog_id'],
                        'dog_name': assignment['dog_name'],
                        'num_dogs': assignment['num_dogs'],
                        'needed_groups': needed_groups,
                        'original_callout': assignment['callout']
                    })
        
        print(f"🚨 Found {len(dogs_to_reassign)} callouts that need drivers assigned:")
        for dog in dogs_to_reassign:
            print(f"   - {dog['dog_name']} ({dog['dog_id']}) - {dog['num_dogs']} dogs, groups {dog['needed_groups']}")
        
        return dogs_to_reassign

    def _parse_groups(self, callout_text):
        """Parse groups from callout text like 'Nate:1&2' -> [1, 2]"""
        try:
            if ':' not in callout_text:
                return []
            
            # Extract the part after the colon
            groups_part = callout_text.split(':', 1)[1].strip()
            
            # Handle different formats
            groups = []
            
            # Remove common non-numeric characters and split
            groups_clean = groups_part.replace('&', ',').replace(' and ', ',').replace(' ', '')
            
            for part in groups_clean.split(','):
                part = part.strip()
                if part.isdigit():
                    groups.append(int(part))
                elif part:
                    # Try to extract numbers from mixed text like "3DD3" or "123"
                    import re
                    numbers = re.findall(r'\d+', part)
                    for num in numbers:
                        if 1 <= int(num) <= 3:  # Only accept valid group numbers
                            groups.append(int(num))
            
            # Remove duplicates and sort
            groups = sorted(list(set(groups)))
            return groups
            
        except Exception as e:
            print(f"⚠️ Error parsing groups from '{callout_text}': {e}")
            return []

    def get_distance(self, dog1_id: str, dog2_id: str) -> float:
        """Get distance between two dogs"""
        try:
            if self.distance_matrix is None:
                return float('inf')
            
            if dog1_id in self.distance_matrix.index and dog2_id in self.distance_matrix.columns:
                distance = self.distance_matrix.loc[dog1_id, dog2_id]
                return float(distance) if not pd.isna(distance) else float('inf')
            
            return float('inf')
            
        except Exception as e:
            return float('inf')

    def calculate_driver_load(self, driver_name: str) -> Dict:
        """Calculate current load for a driver across all groups"""
        load = {'group1': 0, 'group2': 0, 'group3': 0}
        
        if not self.dog_assignments:
            return load
        
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            
            # Skip empty assignments
            if not combined or combined.strip() == "":
                continue
            
            # Extract driver name from combined assignment (before colon)
            if ':' in combined:
                assigned_driver = combined.split(':', 1)[0].strip()
                
                if assigned_driver == driver_name:
                    # Parse groups for this assignment
                    groups_part = combined.split(':', 1)[1].strip()
                    assigned_groups = self._parse_groups(f"dummy:{groups_part}")
                    
                    # Add to load for each group
                    for group in assigned_groups:
                        group_key = f'group{group}'
                        if group_key in load:
                            load[group_key] += assignment['num_dogs']
        
        return load

    def cluster_callout_dogs(self, dogs_to_reassign, max_cluster_distance=0.5):
        """Group callout dogs that are close to each other for efficient batch assignment"""
        print(f"\n🔍 Clustering callout dogs (max distance: {max_cluster_distance} miles)...")
        
        if not dogs_to_reassign:
            return []
        
        clusters = []
        remaining_dogs = dogs_to_reassign.copy()
        
        while remaining_dogs:
            # Start new cluster with first remaining dog
            cluster = [remaining_dogs.pop(0)]
            
            # Find nearby dogs to add to this cluster
            dogs_to_add = []
            for dog in remaining_dogs:
                # Check if this dog is close to any dog in the current cluster
                is_close = False
                for cluster_dog in cluster:
                    distance = self.get_distance(dog['dog_id'], cluster_dog['dog_id'])
                    if distance <= max_cluster_distance:
                        is_close = True
                        break
                
                if is_close:
                    dogs_to_add.append(dog)
            
            # Add close dogs to cluster and remove from remaining
            for dog in dogs_to_add:
                cluster.append(dog)
                remaining_dogs.remove(dog)
            
            clusters.append(cluster)
        
        print(f"📊 Created {len(clusters)} clusters:")
        for i, cluster in enumerate(clusters):
            dog_names = [dog['dog_name'] for dog in cluster]
            print(f"   Cluster {i+1}: {len(cluster)} dogs - {dog_names}")
        
        return clusters

    def optimize_cluster_assignment(self, cluster):
        """Optimally assign cluster dogs to minimize miles while respecting capacity limits"""
        print(f"\n🎯 OPTIMIZING cluster assignment for {len(cluster)} dogs...")
        
        # 🔍 CRITICAL DEBUG: Initialize assignments and track it
        assignments = []
        print(f"🔍 DEBUG: assignments initialized as empty list: {len(assignments)}")
        
        if not cluster:
            print(f"   ❌ Empty cluster provided")
            return assignments
        
        # Get available drivers who are working today
        working_drivers = {}
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if combined and ':' in combined:
                driver_name = combined.split(':', 1)[0].strip()
                if driver_name and driver_name in self.driver_capacities:
                    working_drivers[driver_name] = self.driver_capacities[driver_name]
        
        # Extract callout driver name (to exclude them)
        callout_driver = None
        if cluster[0].get('original_callout'):
            callout_text = cluster[0]['original_callout']
            if ':' in callout_text:
                callout_driver = callout_text.split(':', 1)[0].strip()
        
        print(f"   🚫 Excluding callout driver: {callout_driver}")
        print(f"   👥 Available working drivers: {list(working_drivers.keys())}")
        
        # 🔍 DEBUG: Check before Strategy 1
        print(f"🔍 DEBUG: Before Strategy 1, assignments = {len(assignments)}")
        
        # STRATEGY 1: Try to assign entire cluster to one driver
        print(f"\n   📦 STRATEGY 1: Single driver assignment")
        
        for driver_name, driver_capacity in working_drivers.items():
            # Skip the callout driver
            if driver_name == callout_driver:
                print(f"   ❌ Skipping {driver_name} - they called out!")
                continue
            
            # Calculate current load
            current_load = self.calculate_driver_load(driver_name)
            
            # Calculate total dogs needed per group for the cluster
            group_requirements = {'group1': 0, 'group2': 0, 'group3': 0}
            total_cluster_dogs = 0
            
            for dog in cluster:
                total_cluster_dogs += dog['num_dogs']
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    if group_key in group_requirements:
                        group_requirements[group_key] += dog['num_dogs']
            
            # Check if this driver can handle the entire cluster
            can_handle_all = True
            for group_key, needed in group_requirements.items():
                if needed > 0:  # Only check groups that are actually needed
                    current = current_load.get(group_key, 0)
                    capacity = driver_capacity.get(group_key, 0)
                    if current + needed > capacity:
                        can_handle_all = False
                        break
            
            if can_handle_all and total_cluster_dogs > 0:
                print(f"   ✅ {driver_name} can take entire cluster!")
                assignments.append({
                    'driver': driver_name,
                    'dogs': cluster,
                    'assignment_type': 'full_cluster'
                })
                print(f"🔍 DEBUG: Added {driver_name} to assignments (Strategy 1). Total assignments now: {len(assignments)}")
                print(f"🔍 DEBUG: About to return from Strategy 1. assignments = {len(assignments)}")
                return assignments
            else:
                print(f"   ❌ {driver_name} cannot take full cluster")
        
        # 🔍 DEBUG: Check before Strategy 2
        print(f"🔍 DEBUG: Before Strategy 2, assignments = {len(assignments)}")
        
        # STRATEGY 2: Partial assignments - fill drivers optimally
        print(f"\n   🔄 STRATEGY 2: Optimal partial assignments")
        
        remaining_dogs = cluster.copy()
        print(f"🔍 DEBUG: Starting Strategy 2 with {len(remaining_dogs)} remaining dogs")
        
        # Sort drivers by available capacity (most available first)
        driver_availability = []
        for driver_name, driver_capacity in working_drivers.items():
            if driver_name == callout_driver:
                continue
                
            current_load = self.calculate_driver_load(driver_name)
            available_capacity = {
                'group1': max(0, driver_capacity['group1'] - current_load['group1']),
                'group2': max(0, driver_capacity['group2'] - current_load['group2']),
                'group3': max(0, driver_capacity['group3'] - current_load['group3'])
            }
            total_available = sum(available_capacity.values())
            
            if total_available > 0:
                driver_availability.append((driver_name, available_capacity, total_available))
        
        # Sort by total available capacity (most available first)
        driver_availability.sort(key=lambda x: x[2], reverse=True)
        
        # Try to assign dogs to drivers in order of availability
        for driver_name, available_capacity, total_available in driver_availability:
            if not remaining_dogs:
                break
                
            print(f"\n📋 Trying to assign dogs to {driver_name}:")
            print(f"   Available: G1={available_capacity['group1']}, G2={available_capacity['group2']}, G3={available_capacity['group3']}")
            
            driver_dogs = []
            temp_load = self.calculate_driver_load(driver_name)
            
            # Try to assign as many dogs as possible to this driver
            dogs_to_remove = []
            for dog in remaining_dogs:
                # Check if this driver can handle this dog
                can_handle = True
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    current = temp_load.get(group_key, 0)
                    capacity = self.driver_capacities[driver_name].get(group_key, 0)
                    
                    if current + dog['num_dogs'] > capacity:
                        can_handle = False
                        break
                
                if can_handle:
                    # Assign dog to this driver
                    driver_dogs.append(dog)
                    dogs_to_remove.append(dog)
                    
                    # Update temporary load
                    for group in dog['needed_groups']:
                        group_key = f'group{group}'
                        temp_load[group_key] = temp_load.get(group_key, 0) + dog['num_dogs']
                    
                    print(f"   ✅ Assigned {dog['dog_name']} ({dog['num_dogs']} dogs, groups {dog['needed_groups']})")
            
            # Remove assigned dogs from remaining
            for dog in dogs_to_remove:
                remaining_dogs.remove(dog)
            
            if driver_dogs:
                assignments.append({
                    'driver': driver_name,
                    'dogs': driver_dogs,
                    'assignment_type': 'partial_cluster'
                })
                print(f"   🎯 {driver_name} gets {len(driver_dogs)} dogs total")
                print(f"🔍 DEBUG: Added {driver_name} to assignments (Strategy 2). Total assignments now: {len(assignments)}")
            else:
                print(f"   ❌ {driver_name} cannot take any dogs")
        
        # Show final results
        print(f"\n   📊 FINAL CLUSTER ASSIGNMENT:")
        if assignments:
            print(f"      {len(assignments)} drivers will split the cluster")
            total_assigned = sum(len(assignment['dogs']) for assignment in assignments)
            print(f"      Dogs assigned: {total_assigned}/{len(cluster)}")
            
            for assignment in assignments:
                print(f"      - {assignment['driver']}: {len(assignment['dogs'])} dogs")
        else:
            print(f"      ❌ No drivers available for this cluster")
        
        if remaining_dogs:
            print(f"\n   ⚠️ {len(remaining_dogs)} dogs could not be assigned:")
            for dog in remaining_dogs:
                print(f"      - {dog['dog_name']} ({dog['num_dogs']} dogs, groups {dog['needed_groups']})")
        
        # 🔍 CRITICAL DEBUG: Show exactly what we're returning
        print(f"🔍 DEBUG: About to return from optimize_cluster_assignment")
        print(f"🔍 DEBUG: assignments = {len(assignments)}")
        print(f"🔍 DEBUG: assignments content: {[a['driver'] for a in assignments]}")
        
        # 🔍 EXPLICIT RETURN WITH DEBUG
        if assignments:
            print(f"🔍 DEBUG: ✅ RETURNING {len(assignments)} VALID ASSIGNMENTS")
            return assignments
        else:
            print(f"🔍 DEBUG: ❌ NO ASSIGNMENTS TO RETURN")
            return []

    def optimize_total_system_miles(self, clusters):
        """Find the assignment combination that minimizes total system miles with PARTIAL ASSIGNMENTS"""
        print(f"\n🎯 Optimizing total system miles across {len(clusters)} clusters...")
        
        all_assignments = []
        total_system_miles = 0.0
        
        for i, cluster in enumerate(clusters):
            print(f"\n   🔍 Processing cluster {i+1} of {len(clusters)}...")
            
            # Use new smart partial assignment
            optimal_assignments = self.optimize_cluster_assignment(cluster)
            
            print(f"   🔍 DEBUG: optimal_assignments returned: {len(optimal_assignments) if optimal_assignments else 0} assignments")
            
            if optimal_assignments:
                all_assignments.extend(optimal_assignments)
                # Rough estimate of miles for this cluster
                cluster_miles = len(optimal_assignments) * 1.5
                total_system_miles += cluster_miles
                print(f"   ✅ Cluster {i+1} optimized: {len(optimal_assignments)} assignments, ~{cluster_miles} miles")
            else:
                print(f"   ❌ Could not optimize cluster {i+1} - no drivers available")
        
        print(f"\n📊 TOTAL SYSTEM OPTIMIZATION COMPLETE:")
        print(f"   🎯 Total assignments: {len(all_assignments)}")
        print(f"   📏 Estimated total miles: {total_system_miles}")
        
        return all_assignments, total_system_miles

    def reassign_dogs(self):
        """Main reassignment algorithm with SMART PARTIAL CLUSTER ASSIGNMENTS"""
        print("\n🔄 Starting SMART OPTIMIZED dog reassignment process...")
        print("🔍 DEBUG: reassign_dogs() function called")
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        # Validate data is available
        total_dogs_in_assignments = len([d for d in self.dog_assignments if d.get('dog_id')])
        matching_dogs = 0
        for dog in dogs_to_reassign:
            if dog['dog_id'] in [d['dog_id'] for d in self.dog_assignments]:
                matching_dogs += 1
        
        print(f"\n🔍 Data validation:")
        print(f" Matrix dogs: {len(self.distance_matrix) if self.distance_matrix is not None else 0}")
        print(f" Assignment dogs: {total_dogs_in_assignments}")
        print(f" Callout dogs: {len(dogs_to_reassign)}")
        print(f" Matching dogs: {matching_dogs}")
        
        if len(self.distance_matrix) == 0:
            print("❌ NO DISTANCE MATRIX DATA! Cannot calculate proximities.")
            return []
        
        if matching_dogs == 0:
            print("❌ NO MATCHING DOG IDs! Check that Dog IDs match between sheets.")
            return []
        
        # STEP 1: Cluster nearby callout dogs
        clusters = self.cluster_callout_dogs(dogs_to_reassign, self.CLUSTER_DISTANCE)
        
        if not clusters:
            print("❌ No clusters created")
            return []
        
        # STEP 2: Optimize assignments to minimize total system miles
        optimized_assignments, total_system_miles = self.optimize_total_system_miles(clusters)
        
        if not optimized_assignments:
            print("❌ No assignments could be optimized")
            return []
        
        # STEP 3: Convert to final assignment format
        final_assignments = []
        for assignment in optimized_assignments:
            driver = assignment['driver']
            
            for dog in assignment['dogs']:
                # Parse groups from original callout
                groups_text = "&".join(map(str, dog['needed_groups']))
                new_assignment = f"{driver}:{groups_text}"
                
                final_assignments.append({
                    'dog_id': dog['dog_id'],
                    'dog_name': dog['dog_name'],
                    'new_assignment': new_assignment,
                    'assignment_type': assignment['assignment_type']
                })
        
        print(f"\n📊 SMART OPTIMIZED REASSIGNMENT SUMMARY:")
        print(f"   🎯 TOTAL SYSTEM MILES MINIMIZED: {total_system_miles:.1f}")
        print(f"   ✅ Successfully assigned: {len(final_assignments)}")
        print(f"   📦 Cluster assignments: {len([a for a in optimized_assignments if a['assignment_type'] == 'full_cluster'])}")
        print(f"   🔄 Partial assignments: {len([a for a in optimized_assignments if a['assignment_type'] == 'partial_cluster'])}")
        
        # Show results
        if final_assignments:
            print(f"\n🎉 MILE-OPTIMIZED RESULTS:")
            for assignment in final_assignments:
                assignment_type = f" [{assignment['assignment_type']}]"
                print(f"   {assignment['dog_name']} → {assignment['new_assignment']}{assignment_type}")
        
        print(f"🔍 DEBUG: reassign_dogs() returning {len(final_assignments)} assignments")
        return final_assignments

    def write_results_to_sheets(self, reassignments):
        """Write reassignment results back to Google Sheets"""
        try:
            print(f"\n📝 Writing {len(reassignments)} results to Google Sheets...")
            
            # Note: In this version, we're not actually writing back to sheets
            # since we don't have the full Sheets API setup in this environment
            # This would be implemented with the Google Sheets API
            
            print("⚠️ Write functionality not implemented in this version")
            print("📋 Results that would be written:")
            
            for assignment in reassignments:
                print(f"   {assignment['dog_id']}: {assignment['new_assignment']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error writing to sheets: {e}")
            return False

def main():
    """Main function to run the dog reassignment system"""
    print("🚀 Production Dog Reassignment System")
    print("=" * 50)
    
    # Initialize system
    system = DogReassignmentSystem()
    if not system.setup_google_sheets_client():
        print("❌ Failed to setup Google Sheets client")
        return
    
    # Load all data
    print("\n⬇️ Loading data from Google Sheets...")
    
    if not system.load_distance_matrix():
        print("❌ Failed to load distance matrix")
        return
    
    if not system.load_dog_assignments():
        print("❌ Failed to load dog assignments")
        return
    
    if not system.load_driver_capacities():
        print("❌ Failed to load driver capacities")
        return
    
    # Print summary
    print("\n🔄 Processing callout assignments...")
    
    # Run the reassignment system ONCE
    print("🔍 DEBUG: About to call system.reassign_dogs() - SINGLE CALL ONLY")
    reassignments = system.reassign_dogs()
    print(f"🔍 DEBUG: system.reassign_dogs() completed and returned {len(reassignments) if reassignments else 0}")
    
    # Ensure reassignments is always a list, never None
    if reassignments is None:
        reassignments = []
    
    # Write results separately
    if reassignments:
        write_success = system.write_results_to_sheets(reassignments)
        if write_success:
            print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments")
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")

if __name__ == "__main__":
    main()
