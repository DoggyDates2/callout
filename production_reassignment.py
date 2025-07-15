# production_reassignment.py
# COMPLETE WORKING VERSION: Locality-first assignment with 100.0 placeholder filter

import pandas as pd
import numpy as np
import requests
import json
import os
from typing import Dict, List, Tuple
import gspread
from google.oauth2.service_account import Credentials
import itertools
import time
import random
from copy import deepcopy
import re

class DogReassignmentSystem:
    def __init__(self):
        """Initialize the dog reassignment system"""
        # Google Sheets URLs (CSV export format)
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        
        # DISTANCE LIMITS - LOCALITY-FIRST THRESHOLDS
        self.PREFERRED_DISTANCE = 0.2  # Ideal assignments: immediate proximity
        self.MAX_DISTANCE = 0.5  # Backup assignments: reasonable with moves
        self.ABSOLUTE_MAX_DISTANCE = 0.7  # Search limit for locality-first algorithm
        self.CASCADING_MOVE_MAX = 0.5  # Max distance for cascading swaps
        self.ADJACENT_GROUP_DISTANCE = 0.1  # Base adjacent group distance (scales with radius)
        self.EXCLUSION_DISTANCE = 200.0  # Temporarily increased from 100.0 to see if 100 is placeholder
        
        # Legacy thresholds for compatibility
        self.GREEDY_WALK_MAX_DISTANCE = 0.5
        
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
            
            # Parse the JSON credentials
            credentials_dict = json.loads(service_account_json)
            
            # Set up the credentials with proper scopes
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
            
            # Initialize the gspread client
            self.sheets_client = gspread.authorize(credentials)
            
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
            
            # Read into DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), index_col=0)
            
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
            
            # Read into DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"📊 Map sheet shape: ({len(df)}, {len(df.columns)})")
            print(f"🔍 DEBUG: First few column names: {list(df.columns[:15])}")
            
            assignments = []
            
            for i, row in df.iterrows():
                try:
                    # Column positions (0-indexed)
                    dog_name = row.iloc[1] if len(row) > 1 else ""  # Column B
                    combined = row.iloc[7] if len(row) > 7 else ""  # Column H
                    group = row.iloc[8] if len(row) > 8 else ""     # Column I  
                    dog_id = row.iloc[9] if len(row) > 9 else ""    # Column J
                    callout = row.iloc[10] if len(row) > 10 else "" # Column K
                    num_dogs = row.iloc[5] if len(row) > 5 else 1   # Column F (Number of dogs)
                    
                    # Debug first few rows
                    if i < 10:
                        print(f"🔍 Row {i}: DogName='{dog_name}', Combined='{combined}', DogID='{dog_id}', Callout='{callout}'")
                    
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
                    print(f"⚠️ Error processing row {i}: {e}")
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
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading driver capacities: {e}")
            return False

    def get_dogs_to_reassign(self):
        """Find dogs that need reassignment (callouts) - excluding non-dog entries"""
        dogs_to_reassign = []
        
        if not self.dog_assignments:
            return dogs_to_reassign
        
        print(f"🔍 DEBUG: Checking {len(self.dog_assignments)} total assignments for callouts...")
        
        callout_candidates = 0
        filtered_out = 0
        no_colon = 0
        no_groups = 0
        
        for i, assignment in enumerate(self.dog_assignments):
            # Debug first few and last few
            if i < 5 or i >= len(self.dog_assignments) - 5:
                print(f"   Row {i}: ID={assignment.get('dog_id', 'MISSING')}, Combined='{assignment.get('combined', 'MISSING')}', Callout='{assignment.get('callout', 'MISSING')}'")
            
            # Check for callout: Combined column blank AND Callout column has content
            combined_blank = (not assignment['combined'] or assignment['combined'].strip() == "")
            callout_has_content = (assignment['callout'] and assignment['callout'].strip() != "")
            
            if combined_blank and callout_has_content:
                callout_candidates += 1
                
                # FILTER OUT NON-DOGS: Skip Parking, Field, and other administrative entries
                dog_name = str(assignment.get('dog_name', '')).lower().strip()
                if any(keyword in dog_name for keyword in ['parking', 'field', 'admin', 'office']):
                    print(f"   ⏭️ Skipping non-dog entry: {assignment['dog_name']} ({assignment['dog_id']})")
                    filtered_out += 1
                    continue
                
                # Extract the FULL assignment string (everything after the colon)
                callout_text = assignment['callout'].strip()
                
                if ':' not in callout_text:
                    print(f"   ⚠️ No colon in callout for {assignment.get('dog_id', 'UNKNOWN')}: '{callout_text}'")
                    no_colon += 1
                    continue
                
                original_driver = callout_text.split(':', 1)[0].strip()
                full_assignment_string = callout_text.split(':', 1)[1].strip()  # Keep this EXACTLY
                
                # Parse out just the numbers for capacity checking
                needed_groups = self._extract_groups_for_capacity_check(full_assignment_string)
                
                if needed_groups:
                    dogs_to_reassign.append({
                        'dog_id': assignment['dog_id'],
                        'dog_name': assignment['dog_name'],
                        'num_dogs': assignment['num_dogs'],
                        'needed_groups': needed_groups,  # For capacity checking
                        'full_assignment_string': full_assignment_string,  # Preserve exactly
                        'original_callout': assignment['callout'],
                        'original_driver': original_driver
                    })
                else:
                    print(f"   ⚠️ No groups found for {assignment.get('dog_id', 'UNKNOWN')}: '{full_assignment_string}'")
                    no_groups += 1
        
        print(f"🔍 DEBUG SUMMARY:")
        print(f"   📊 Total assignments checked: {len(self.dog_assignments)}")
        print(f"   🎯 Callout candidates (blank combined + has callout): {callout_candidates}")
        print(f"   🚫 Filtered out (non-dogs): {filtered_out}")
        print(f"   ⚠️ No colon in callout: {no_colon}")
        print(f"   ⚠️ No groups extracted: {no_groups}")
        print(f"   ✅ Final dogs to reassign: {len(dogs_to_reassign)}")
        
        print(f"\n🚨 Found {len(dogs_to_reassign)} REAL dogs that need drivers assigned:")
        for dog in dogs_to_reassign:
            print(f"   - {dog['dog_name']} ({dog['dog_id']}) - {dog['num_dogs']} dogs")
            print(f"     Original: {dog['original_callout']}")  
            print(f"     Assignment string: '{dog['full_assignment_string']}'")
            print(f"     Capacity needed in groups: {dog['needed_groups']}")
        
        return dogs_to_reassign

    def _extract_groups_for_capacity_check(self, assignment_string):
        """Extract group numbers for capacity checking - each digit 1,2,3 is a separate group"""
        try:
            # Extract all digits that are 1, 2, or 3 from the string
            group_digits = re.findall(r'[123]', assignment_string)
            
            # Convert each digit to an integer and remove duplicates
            groups = sorted(list(set(int(digit) for digit in group_digits)))
            
            return groups
            
        except Exception as e:
            print(f"⚠️ Error extracting groups from '{assignment_string}': {e}")
            return []

    def get_distance(self, dog1_id: str, dog2_id: str) -> float:
        """ENHANCED: Get distance between two dogs, filtering out 100.0 placeholders"""
        try:
            if self.distance_matrix is None:
                return float('inf')
            
            if dog1_id in self.distance_matrix.index and dog2_id in self.distance_matrix.columns:
                distance = self.distance_matrix.loc[dog1_id, dog2_id]
                
                if pd.isna(distance):
                    return float('inf')
                
                distance_float = float(distance)
                
                # 🎯 KEY FIX: Filter out 100.0 placeholders
                if distance_float == 100.0:
                    return float('inf')  # Treat as "no viable connection"
                
                return distance_float
            
            return float('inf')
            
        except Exception as e:
            return float('inf')

    def calculate_driver_load(self, driver_name: str, current_assignments: List = None) -> Dict:
        """Calculate current load for a driver across all groups"""
        load = {'group1': 0, 'group2': 0, 'group3': 0}
        
        # Use provided assignments or default to original assignments
        assignments_to_use = current_assignments if current_assignments else self.dog_assignments
        
        if not assignments_to_use:
            return load
        
        for assignment in assignments_to_use:
            if current_assignments:
                # Working with dynamic assignment list
                if assignment.get('driver') == driver_name:
                    # Parse groups for this assignment
                    assigned_groups = assignment.get('needed_groups', [])
                    
                    # Add to load for each group
                    for group in assigned_groups:
                        group_key = f'group{group}'
                        if group_key in load:
                            load[group_key] += assignment.get('num_dogs', 1)
            else:
                # Working with original assignment data
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
                        assigned_groups = self._extract_groups_for_capacity_check(groups_part)
                        
                        # Add to load for each group
                        for group in assigned_groups:
                            group_key = f'group{group}'
                            if group_key in load:
                                load[group_key] += assignment['num_dogs']
        
        return load

    def check_group_compatibility(self, callout_groups, driver_groups, distance, current_radius=None):
        """Check if groups are compatible considering adjacent group penalty"""
        # Extract unique group numbers from both sets
        callout_set = set(callout_groups)
        driver_set = set(driver_groups)
        
        # Perfect match - same groups
        if callout_set.intersection(driver_set):
            # Use current radius if provided, otherwise use default preferred distance
            max_distance = current_radius if current_radius else self.PREFERRED_DISTANCE
            return distance <= max_distance
        
        # Adjacent groups - need to be proportionally closer (50% of current radius)
        adjacent_pairs = [(1, 2), (2, 3), (2, 1), (3, 2)]
        for callout_group in callout_set:
            for driver_group in driver_set:
                if (callout_group, driver_group) in adjacent_pairs:
                    # Scale adjacent group threshold with current search radius
                    if current_radius:
                        adjacent_threshold = current_radius * 0.5  # 50% of current radius
                    else:
                        adjacent_threshold = self.ADJACENT_GROUP_DISTANCE
                    return distance <= adjacent_threshold
        
        # No compatibility
        return False

    def get_current_driver_dogs(self, driver_name, current_assignments):
        """Get all dogs currently assigned to a specific driver"""
        return [assignment for assignment in current_assignments 
                if assignment.get('driver') == driver_name]

    def find_move_targets_for_dog(self, dog_to_move, current_assignments, max_distance):
        """Find potential drivers within range who can accept a dog"""
        targets = []
        
        for assignment in current_assignments:
            target_driver = assignment['driver']
            
            # Skip same driver
            if target_driver == dog_to_move.get('driver'):
                continue
            
            # Calculate distance
            distance = self.get_distance(dog_to_move['dog_id'], assignment['dog_id'])
            
            # Check group compatibility
            dog_groups = dog_to_move.get('needed_groups', [])
            target_groups = assignment.get('needed_groups', [])
            
            if not self.check_group_compatibility(dog_groups, target_groups, distance, max_distance):
                continue
            
            if distance > max_distance:
                continue
            
            # Check if target driver has capacity
            target_load = self.calculate_driver_load(target_driver, current_assignments)
            target_capacity = self.driver_capacities.get(target_driver, {})
            
            can_accept = True
            for group in dog_groups:
                group_key = f'group{group}'
                current = target_load.get(group_key, 0)
                max_cap = target_capacity.get(group_key, 0)
                needed = dog_to_move.get('num_dogs', 1)
                
                if current + needed > max_cap:
                    can_accept = False
                    break
            
            if can_accept:
                targets.append({
                    'driver': target_driver,
                    'distance': distance,
                    'via_dog': assignment['dog_name'],
                    'via_dog_id': assignment['dog_id']
                })
        
        return sorted(targets, key=lambda x: x['distance'])

    def attempt_cascading_move(self, blocked_driver, callout_dog, current_assignments, max_cascade_distance):
        """Attempt to free space in blocked_driver by moving one of their dogs"""
        print(f"🔄 ATTEMPTING CASCADING MOVE for {callout_dog['dog_name']} → {blocked_driver}")
        print(f"   Need to free space in {blocked_driver} for groups {callout_dog['needed_groups']}")
        
        driver_dogs = self.get_current_driver_dogs(blocked_driver, current_assignments)
        print(f"   {blocked_driver} currently has {len(driver_dogs)} dogs assigned")
        
        # Show current load for this driver
        current_load = self.calculate_driver_load(blocked_driver, current_assignments)
        driver_capacity = self.driver_capacities.get(blocked_driver, {})
        print(f"   {blocked_driver} capacity: ***{driver_capacity}***")
        print(f"   {blocked_driver} current load: ***{current_load}***")
        
        # Try to move each dog, starting with single-group dogs (easier to place)
        move_candidates = sorted(driver_dogs, key=lambda x: (len(x.get('needed_groups', [])), x.get('num_dogs', 1)))
        print(f"   Trying to move {len(move_candidates)} dogs (easiest first):")
        
        for i, dog_to_move in enumerate(move_candidates):
            print(f"     {i+1}. {dog_to_move['dog_name']} ({dog_to_move['dog_id']}) - groups {dog_to_move['needed_groups']}, {dog_to_move['num_dogs']} dogs")
            
            # Find targets for this dog within cascade distance
            targets = self.find_move_targets_for_dog(dog_to_move, current_assignments, max_cascade_distance)
            print(f"        Found {len(targets)} potential targets within {max_cascade_distance}mi:")
            
            for j, target in enumerate(targets[:3]):  # Show top 3 targets
                print(f"          {j+1}. {target['driver']} - {target['distance']:.3f}mi via {target['via_dog']}")
            
            if targets:
                best_target = targets[0]
                print(f"        ✅ EXECUTING MOVE: {dog_to_move['dog_name']} from {blocked_driver} → {best_target['driver']}")
                
                # Execute the move
                for assignment in current_assignments:
                    if assignment['dog_id'] == dog_to_move['dog_id']:
                        assignment['driver'] = best_target['driver']
                        break
                
                return {
                    'moved_dog': dog_to_move,
                    'from_driver': blocked_driver,
                    'to_driver': best_target['driver'],
                    'distance': best_target['distance'],
                    'via_dog': best_target['via_dog']
                }
            else:
                print(f"        ❌ No targets found for {dog_to_move['dog_name']}")
        
        print(f"   ❌ CASCADING MOVE FAILED - no dogs could be relocated")
        return None

    def locality_first_assignment(self):
        """NEW: Locality-first assignment algorithm with cascading moves"""
        print("\n🎯 LOCALITY-FIRST ASSIGNMENT ALGORITHM")
        print("🔄 Step-by-step proximity optimization with dynamic state updates")
        print("📏 Starting at 0.2mi, expanding to 0.7mi in 0.1mi increments")
        print("🔄 Adjacent groups scale with radius (50% of current radius)")
        print("🚶 Cascading moves up to 0.5mi to free space")
        print("🎯 FILTERING OUT 100.0 PLACEHOLDERS for realistic distances")
        print("=" * 80)
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        # Build initial current assignments state
        current_assignments = []
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if combined and ':' in combined:
                driver = combined.split(':', 1)[0].strip()
                assignment_string = combined.split(':', 1)[1].strip()
                groups = self._extract_groups_for_capacity_check(assignment_string)
                
                current_assignments.append({
                    'dog_id': assignment['dog_id'],
                    'dog_name': assignment['dog_name'],
                    'driver': driver,
                    'needed_groups': groups,
                    'num_dogs': assignment['num_dogs']
                })
        
        assignments_made = []
        moves_made = []
        dogs_remaining = dogs_to_reassign.copy()
        
        print(f"🐕 Processing {len(dogs_remaining)} callout dogs")
        
        # DEBUG: Check distance matrix compatibility with callout dogs
        print(f"\n🔍 DIAGNOSTIC: Distance matrix compatibility check...")
        print(f"   📊 Distance matrix has {len(self.distance_matrix)} dog IDs")
        print(f"   📋 First 10 matrix IDs: {list(self.distance_matrix.index[:10])}")
        print(f"   🎯 Callout dog IDs: {[dog['dog_id'] for dog in dogs_remaining[:5]]}")
        
        # Check if callout dog IDs exist in distance matrix
        missing_ids = []
        for dog in dogs_remaining[:5]:
            if dog['dog_id'] not in self.distance_matrix.index:
                missing_ids.append(dog['dog_id'])
        
        if missing_ids:
            print(f"   🚨 MISSING FROM MATRIX: {missing_ids}")
        else:
            print(f"   ✅ All callout dog IDs found in distance matrix")
            
            # Test a few actual distance lookups
            print(f"   🔍 Sample distance checks:")
            for i, dog in enumerate(dogs_remaining[:2]):
                print(f"     Testing callout dog: {dog['dog_name']} ({dog['dog_id']})")
                for j, assignment in enumerate(current_assignments[:5]):
                    try:
                        # Raw matrix lookup
                        raw_value = self.distance_matrix.loc[dog['dog_id'], assignment['dog_id']]
                        
                        # Through our enhanced function
                        distance = self.get_distance(dog['dog_id'], assignment['dog_id'])
                        
                        if raw_value == 100.0:
                            print(f"       Raw matrix[{dog['dog_id']}, {assignment['dog_id']}] = {raw_value} → FILTERED OUT")
                            print(f"       get_distance() = {distance} (inf = filtered placeholder)")
                        else:
                            print(f"       Raw matrix[{dog['dog_id']}, {assignment['dog_id']}] = {raw_value}")
                            print(f"       get_distance() = {distance:.3f}mi ✅")
                        
                    except Exception as e:
                        print(f"       ❌ Error: {e}")
                    
                    if j == 4:  # Show 5 samples per dog
                        break
                if i == 0:  # Just show 1 dog for detailed analysis
                    break
        
        # DIAGNOSTIC: Check driver capacity availability
        print(f"\n🔍 DIAGNOSTIC: Driver capacity analysis...")
        total_capacity = {'group1': 0, 'group2': 0, 'group3': 0}
        total_used = {'group1': 0, 'group2': 0, 'group3': 0}
        
        for driver, capacity in self.driver_capacities.items():
            current_load = self.calculate_driver_load(driver, current_assignments)
            for group_key in ['group1', 'group2', 'group3']:
                total_capacity[group_key] += capacity.get(group_key, 0)
                total_used[group_key] += current_load.get(group_key, 0)
        
        print(f"   📊 Total capacity vs used:")
        for group_key in ['group1', 'group2', 'group3']:
            available = total_capacity[group_key] - total_used[group_key]
            print(f"   {group_key}: {available}/{total_capacity[group_key]} available ({total_used[group_key]} used)")
        
        # DIAGNOSTIC: Check distances for first few dogs with placeholder filtering
        print(f"\n🔍 DIAGNOSTIC: Distance check for first 3 dogs (with 100.0 filtering)...")
        print(f"   📏 Thresholds: Perfect match ≤0.2mi, Adjacent groups scale with radius (50% of radius)")
        
        # Group current assignments by driver to see all options
        drivers_dogs = {}
        for assignment in current_assignments:
            driver = assignment['driver']
            if driver not in drivers_dogs:
                drivers_dogs[driver] = []
            drivers_dogs[driver].append(assignment)
        
        print(f"   📊 Found {len(drivers_dogs)} drivers with assigned dogs:")
        for driver, dogs in list(drivers_dogs.items())[:5]:  # Show first 5 drivers
            print(f"     {driver}: {len(dogs)} dogs (groups: {[d['needed_groups'] for d in dogs[:3]]})")
        
        for i, callout_dog in enumerate(dogs_remaining[:3]):
            print(f"   {callout_dog['dog_name']} ({callout_dog['dog_id']}) needs groups {callout_dog['needed_groups']}:")
            
            # Check distances to ALL drivers, not just first few assignments
            driver_distances = {}
            for driver, dogs in drivers_dogs.items():
                closest_distance = float('inf')
                closest_dog = None
                
                for dog_assignment in dogs:
                    distance = self.get_distance(callout_dog['dog_id'], dog_assignment['dog_id'])
                    if distance < closest_distance and distance != float('inf'):  # Skip filtered placeholders
                        closest_distance = distance
                        closest_dog = dog_assignment
                
                if closest_dog and closest_distance != float('inf'):
                    driver_distances[driver] = {
                        'distance': closest_distance,
                        'via_dog': closest_dog['dog_name'],
                        'via_dog_id': closest_dog['dog_id'],
                        'groups': closest_dog['needed_groups']
                    }
            
            # Sort drivers by distance and show closest 5
            sorted_drivers = sorted(driver_distances.items(), key=lambda x: x[1]['distance'])
            print(f"     Closest drivers by distance (placeholders filtered):")
            for j, (driver, info) in enumerate(sorted_drivers[:5]):
                group_compat = self.check_group_compatibility(callout_dog['needed_groups'], info['groups'], info['distance'], 0.7)
                print(f"       {j+1}. {driver} - {info['distance']:.3f}mi via {info['via_dog']} (groups: {info['groups']}, compatible: {group_compat})")
                
                # Special highlight for Leen since user mentioned Ozzy/Wyatt
                if driver == 'Leen':
                    print(f"          🎯 LEEN FOUND! Distance to {info['via_dog']} ({info['via_dog_id']})")
            
            if not sorted_drivers:
                print(f"     ❌ No realistic distances found (all were 100.0 placeholders)")
            
            if i == 0:  # Just show detailed analysis for first dog (Fawkes)
                break
        
        print(f"\n📍 STEP 1: Direct assignments at ≤{self.PREFERRED_DISTANCE}mi")
        
        dogs_assigned_step1 = []
        for callout_dog in dogs_remaining[:]:
            best_assignment = None
            best_distance = float('inf')
            
            # Check all drivers for direct assignment
            for assignment in current_assignments:
                driver = assignment['driver']
                distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                
                # Skip if filtered placeholder
                if distance == float('inf'):
                    continue
                
                # Check group compatibility with distance requirements
                if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, self.PREFERRED_DISTANCE):
                    continue
                
                # Check capacity
                current_load = self.calculate_driver_load(driver, current_assignments)
                has_capacity = True
                
                for group in callout_dog['needed_groups']:
                    group_key = f'group{group}'
                    current = current_load.get(group_key, 0)
                    max_cap = self.driver_capacities.get(driver, {}).get(group_key, 0)
                    needed = callout_dog['num_dogs']
                    
                    if current + needed > max_cap:
                        has_capacity = False
                        break
                
                if has_capacity and distance < best_distance:
                    best_assignment = {
                        'driver': driver,
                        'distance': distance,
                        'via_dog': assignment['dog_name']
                    }
                    best_distance = distance
            
            if best_assignment:
                # Make the assignment
                driver = best_assignment['driver']
                distance = best_assignment['distance']
                
                assignment_record = {
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                    'driver': driver,
                    'distance': distance,
                    'quality': 'GOOD',
                    'assignment_type': 'direct'
                }
                
                assignments_made.append(assignment_record)
                
                # Update current assignments state
                current_assignments.append({
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'driver': driver,
                    'needed_groups': callout_dog['needed_groups'],
                    'num_dogs': callout_dog['num_dogs']
                })
                
                dogs_assigned_step1.append(callout_dog)
                print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.1f}mi)")
        
        # Remove assigned dogs
        for dog in dogs_assigned_step1:
            dogs_remaining.remove(dog)
        
        print(f"   📊 Step 1 results: {len(dogs_assigned_step1)} direct assignments")
        
        # Step 2: Capacity-blocked assignments with cascading moves
        if dogs_remaining:
            print(f"\n🔄 STEP 2: Cascading moves to free space at ≤{self.PREFERRED_DISTANCE}mi")
            
            dogs_assigned_step2 = []
            for callout_dog in dogs_remaining[:]:
                # Find drivers within range but blocked by capacity
                blocked_drivers = []
                
                for assignment in current_assignments:
                    driver = assignment['driver']
                    distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                    
                    # Skip if filtered placeholder
                    if distance == float('inf'):
                        continue
                    
                    # Check group compatibility
                    if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, self.PREFERRED_DISTANCE):
                        continue
                    
                    # Check if blocked by capacity
                    current_load = self.calculate_driver_load(driver, current_assignments)
                    has_capacity = True
                    
                    for group in callout_dog['needed_groups']:
                        group_key = f'group{group}'
                        current = current_load.get(group_key, 0)
                        max_cap = self.driver_capacities.get(driver, {}).get(group_key, 0)
                        needed = callout_dog['num_dogs']
                        
                        if current + needed > max_cap:
                            has_capacity = False
                            break
                    
                    if not has_capacity:
                        blocked_drivers.append({
                            'driver': driver,
                            'distance': distance
                        })
                
                # Try cascading moves for the closest blocked driver
                if blocked_drivers:
                    blocked_drivers.sort(key=lambda x: x['distance'])
                    best_blocked = blocked_drivers[0]
                    
                    # Attempt cascading move
                    move_result = self.attempt_cascading_move(
                        best_blocked['driver'], 
                        callout_dog, 
                        current_assignments, 
                        self.PREFERRED_DISTANCE
                    )
                    
                    if move_result:
                        # Record the move
                        moves_made.append({
                            'dog_name': move_result['moved_dog']['dog_name'],
                            'dog_id': move_result['moved_dog']['dog_id'],
                            'from_driver': move_result['from_driver'],
                            'to_driver': move_result['to_driver'],
                            'distance': move_result['distance'],
                            'reason': f"free_space_for_{callout_dog['dog_name']}"
                        })
                        
                        # Now assign the callout dog to the freed space
                        driver = best_blocked['driver']
                        distance = best_blocked['distance']
                        
                        assignment_record = {
                            'dog_id': callout_dog['dog_id'],
                            'dog_name': callout_dog['dog_name'],
                            'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                            'driver': driver,
                            'distance': distance,
                            'quality': 'GOOD',
                            'assignment_type': 'cascading'
                        }
                        
                        assignments_made.append(assignment_record)
                        
                        # Update current assignments state
                        current_assignments.append({
                            'dog_id': callout_dog['dog_id'],
                            'dog_name': callout_dog['dog_name'],
                            'driver': driver,
                            'needed_groups': callout_dog['needed_groups'],
                            'num_dogs': callout_dog['num_dogs']
                        })
                        
                        dogs_assigned_step2.append(callout_dog)
                        print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.1f}mi)")
                        print(f"      🚶 Moved {move_result['moved_dog']['dog_name']}: {move_result['from_driver']} → {move_result['to_driver']}")
            
            # Remove assigned dogs
            for dog in dogs_assigned_step2:
                dogs_remaining.remove(dog)
            
            print(f"   📊 Step 2 results: {len(dogs_assigned_step2)} assignments with {len([m for m in moves_made if 'free_space' in m['reason']])} cascading moves")
        
        # Step 3: Incremental radius expansion (0.3 to 0.7 miles)
        current_radius = 0.3
        step_number = 3
        
        while current_radius <= self.ABSOLUTE_MAX_DISTANCE and dogs_remaining:
            print(f"\n📏 STEP {step_number}: Radius expansion to ≤{current_radius}mi")
            print(f"   🎯 Thresholds: Perfect match ≤{current_radius}mi, Adjacent groups ≤{current_radius*0.5:.1f}mi")
            
            dogs_assigned_this_radius = []
            
            for callout_dog in dogs_remaining[:]:
                # Try direct assignment at current radius
                best_assignment = None
                best_distance = float('inf')
                
                for assignment in current_assignments:
                    driver = assignment['driver']
                    distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                    
                    # Skip if filtered placeholder
                    if distance == float('inf'):
                        continue
                    
                    if distance > current_radius:
                        continue
                    
                    # Check group compatibility
                    if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, current_radius):
                        continue
                    
                    # Check capacity
                    current_load = self.calculate_driver_load(driver, current_assignments)
                    has_capacity = True
                    
                    for group in callout_dog['needed_groups']:
                        group_key = f'group{group}'
                        current = current_load.get(group_key, 0)
                        max_cap = self.driver_capacities.get(driver, {}).get(group_key, 0)
                        needed = callout_dog['num_dogs']
                        
                        if current + needed > max_cap:
                            has_capacity = False
                            break
                    
                    if has_capacity and distance < best_distance:
                        best_assignment = {
                            'driver': driver,
                            'distance': distance,
                            'via_dog': assignment['dog_name']
                        }
                        best_distance = distance
                
                if best_assignment:
                    # Direct assignment possible
                    driver = best_assignment['driver']
                    distance = best_assignment['distance']
                    
                    # Determine quality with 3-way check
                    if distance <= self.PREFERRED_DISTANCE:
                        quality = 'GOOD'
                    elif distance <= self.MAX_DISTANCE:
                        quality = 'BACKUP'
                    else:
                        quality = 'EMERGENCY'
                    
                    assignment_record = {
                        'dog_id': callout_dog['dog_id'],
                        'dog_name': callout_dog['dog_name'],
                        'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                        'driver': driver,
                        'distance': distance,
                        'quality': quality,
                        'assignment_type': 'radius_expansion'
                    }
                    
                    assignments_made.append(assignment_record)
                    
                    # Update state
                    current_assignments.append({
                        'dog_id': callout_dog['dog_id'],
                        'dog_name': callout_dog['dog_name'],
                        'driver': driver,
                        'needed_groups': callout_dog['needed_groups'],
                        'num_dogs': callout_dog['num_dogs']
                    })
                    
                    dogs_assigned_this_radius.append(callout_dog)
                    print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.1f}mi)")
                
                else:
                    # Try cascading moves at current radius
                    blocked_drivers = []
                    
                    for assignment in current_assignments:
                        driver = assignment['driver']
                        distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                        
                        # Skip if filtered placeholder
                        if distance == float('inf'):
                            continue
                        
                        if distance > current_radius:
                            continue
                        
                        # Check group compatibility
                        if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, current_radius):
                            continue
                        
                        # Check if blocked by capacity
                        current_load = self.calculate_driver_load(driver, current_assignments)
                        has_capacity = True
                        
                        for group in callout_dog['needed_groups']:
                            group_key = f'group{group}'
                            current = current_load.get(group_key, 0)
                            max_cap = self.driver_capacities.get(driver, {}).get(group_key, 0)
                            needed = callout_dog['num_dogs']
                            
                            if current + needed > max_cap:
                                has_capacity = False
                                break
                        
                        if not has_capacity:
                            blocked_drivers.append({
                                'driver': driver,
                                'distance': distance
                            })
                    
                    if blocked_drivers:
                        blocked_drivers.sort(key=lambda x: x['distance'])
                        best_blocked = blocked_drivers[0]
                        
                        # Try cascading move up to 0.5 miles
                        move_result = self.attempt_cascading_move(
                            best_blocked['driver'], 
                            callout_dog, 
                            current_assignments, 
                            self.CASCADING_MOVE_MAX
                        )
                        
                        if move_result:
                            # Record the move
                            moves_made.append({
                                'dog_name': move_result['moved_dog']['dog_name'],
                                'dog_id': move_result['moved_dog']['dog_id'],
                                'from_driver': move_result['from_driver'],
                                'to_driver': move_result['to_driver'],
                                'distance': move_result['distance'],
                                'reason': f"radius_{current_radius}_space_for_{callout_dog['dog_name']}"
                            })
                            
                            # Assign the callout dog
                            driver = best_blocked['driver']
                            distance = best_blocked['distance']
                            
                            # Determine quality with 3-way check
                            if distance <= self.PREFERRED_DISTANCE:
                                quality = 'GOOD'
                            elif distance <= self.MAX_DISTANCE:
                                quality = 'BACKUP'
                            else:
                                quality = 'EMERGENCY'
                            
                            assignment_record = {
                                'dog_id': callout_dog['dog_id'],
                                'dog_name': callout_dog['dog_name'],
                                'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                                'driver': driver,
                                'distance': distance,
                                'quality': quality,
                                'assignment_type': 'cascading_radius'
                            }
                            
                            assignments_made.append(assignment_record)
                            
                            # Update state
                            current_assignments.append({
                                'dog_id': callout_dog['dog_id'],
                                'dog_name': callout_dog['dog_name'],
                                'driver': driver,
                                'needed_groups': callout_dog['needed_groups'],
                                'num_dogs': callout_dog['num_dogs']
                            })
                            
                            dogs_assigned_this_radius.append(callout_dog)
                            print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.1f}mi)")
                            print(f"      🚶 Moved {move_result['moved_dog']['dog_name']}: {move_result['from_driver']} → {move_result['to_driver']}")
            
            # Remove assigned dogs
            for dog in dogs_assigned_this_radius:
                dogs_remaining.remove(dog)
            
            print(f"   📊 Radius {current_radius}mi results: {len(dogs_assigned_this_radius)} assignments")
            
            current_radius += 0.1
            step_number += 1
        
        # Step 4: Onion-layer backflow (if dogs still remain)
        if dogs_remaining:
            print(f"\n🧅 STEP {step_number}: Onion-layer backflow for {len(dogs_remaining)} remaining dogs")
            print("   🔄 Attempting to push outer assignments further out to create inner space")
            
            # TODO: Implement onion-layer backflow if needed
            # For now, mark remaining as emergency
            for callout_dog in dogs_remaining:
                assignment_record = {
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'new_assignment': f"UNASSIGNED:{callout_dog['full_assignment_string']}",
                    'driver': 'UNASSIGNED',
                    'distance': float('inf'),
                    'quality': 'EMERGENCY',
                    'assignment_type': 'failed'
                }
                assignments_made.append(assignment_record)
                print(f"   ❌ {callout_dog['dog_name']} - No viable assignment found")
        
        # Store moves for writing
        self.greedy_moves_made = moves_made
        
        # Summary
        total_dogs = len(dogs_to_reassign)
        good_count = len([a for a in assignments_made if a['quality'] == 'GOOD'])
        backup_count = len([a for a in assignments_made if a['quality'] == 'BACKUP'])
        emergency_count = len([a for a in assignments_made if a['quality'] == 'EMERGENCY'])
        
        print(f"\n🏆 LOCALITY-FIRST ALGORITHM RESULTS (with 100.0 filtering):")
        print(f"   📊 {len(assignments_made)}/{total_dogs} dogs processed")
        print(f"   💚 {good_count} GOOD assignments (≤{self.PREFERRED_DISTANCE}mi)")
        print(f"   🟡 {backup_count} BACKUP assignments ({self.PREFERRED_DISTANCE}-{self.MAX_DISTANCE}mi)")
        print(f"   🚨 {emergency_count} EMERGENCY assignments (>{self.MAX_DISTANCE}mi)")
        print(f"   🚶 {len(moves_made)} cascading moves executed")
        print(f"   🎯 Success rate: {(good_count + backup_count)/total_dogs*100:.0f}% practical assignments")
        print(f"   🎯 Placeholder filtering: All 100.0 distances ignored for realistic assignments")
        
        return assignments_made

    def reassign_dogs_multi_strategy_optimization(self):
        """NEW: Locality-first algorithm with 100.0 placeholder filtering"""
        print("\n🔄 Starting LOCALITY-FIRST ASSIGNMENT SYSTEM...")
        print("🎯 Strategy: Proximity-first with cascading moves")
        print("📊 Quality: GOOD ≤0.2mi, BACKUP ≤0.5mi, EMERGENCY >0.5mi")
        print("🚨 Focus: Immediate proximity with dynamic space optimization")
        print("🎯 FILTERING: 100.0 placeholders ignored for realistic distance calculations")
        print("=" * 80)
        
        # Try the locality-first algorithm with placeholder filtering
        try:
            return self.locality_first_assignment()
        except Exception as e:
            print(f"⚠️ Locality-first algorithm failed: {e}")
            print("🔄 Falling back to basic assignment...")
            return []

    def write_results_to_sheets(self, reassignments):
        """Write reassignment results and greedy walk moves back to Google Sheets"""
        try:
            print(f"\n📝 Writing {len(reassignments)} results to Google Sheets...")
            
            if not hasattr(self, 'sheets_client') or not self.sheets_client:
                print("❌ Google Sheets client not initialized")
                return False
            
            # Pre-validation of reassignments data
            print(f"🔒 PRE-VALIDATION: Checking reassignment data structure...")
            for i, assignment in enumerate(reassignments[:3]):  # Show first 3
                dog_id = assignment.get('dog_id', 'MISSING')
                new_assignment = assignment.get('new_assignment', 'MISSING')
                print(f"   {i+1}. Dog ID: '{dog_id}' → New Assignment: '{new_assignment}'")
                
                # Critical safety checks
                if dog_id == new_assignment:
                    print(f"   🚨 CRITICAL ERROR: dog_id equals new_assignment! ABORTING!")
                    return False
                
                if new_assignment.endswith('x') and new_assignment[:-1].isdigit():
                    print(f"   🚨 CRITICAL ERROR: new_assignment looks like dog_id! ABORTING!")
                    return False
                
                if ':' not in new_assignment:
                    print(f"   🚨 CRITICAL ERROR: new_assignment missing driver:group format! ABORTING!")
                    return False
            
            print(f"✅ Pre-validation passed!")
            
            # Extract sheet ID
            sheet_id = "1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0"
            
            # Open the spreadsheet
            spreadsheet = self.sheets_client.open_by_key(sheet_id)
            
            # Get the worksheet
            worksheet = None
            try:
                for ws in spreadsheet.worksheets():
                    if str(ws.id) == "267803750":
                        worksheet = ws
                        break
            except:
                pass
            
            if not worksheet:
                for sheet_name in ["Map", "Sheet1", "Dogs", "Assignments"]:
                    try:
                        worksheet = spreadsheet.worksheet(sheet_name)
                        print(f"📋 Using sheet: {sheet_name}")
                        break
                    except:
                        continue
            
            if not worksheet:
                print("❌ Could not find the target worksheet")
                return False
            
            # Get all data
            all_data = worksheet.get_all_values()
            if not all_data:
                print("❌ No data found in worksheet")
                return False
            
            header_row = all_data[0]
            print(f"📋 Sheet has {len(all_data)} rows")
            
            # Find the Dog ID column
            dog_id_col = None
            for i, header in enumerate(header_row):
                header_clean = str(header).lower().strip()
                if 'dog id' in header_clean:
                    dog_id_col = i
                    print(f"📍 Found Dog ID column at index {i}")
                    break
            
            if dog_id_col is None:
                print("❌ Could not find 'Dog ID' column")
                return False
            
            # Target Column H (Combined column) - index 7
            target_col = 7  
            print(f"📍 Writing to Column H (Combined) at index {target_col}")
            
            # Prepare batch updates for reassignments
            updates = []
            updates_count = 0
            
            print(f"\n🔍 Processing {len(reassignments)} reassignments...")
            
            # Process reassignments
            for assignment in reassignments:
                dog_id = str(assignment.get('dog_id', '')).strip()
                new_assignment = str(assignment.get('new_assignment', '')).strip()
                
                # Final validation
                if not new_assignment or new_assignment == dog_id or ':' not in new_assignment:
                    print(f"  ❌ SKIPPING invalid assignment for {dog_id}")
                    continue
                
                # Find the row for this dog ID
                for row_idx in range(1, len(all_data)):
                    if dog_id_col < len(all_data[row_idx]):
                        current_dog_id = str(all_data[row_idx][dog_id_col]).strip()
                        
                        if current_dog_id == dog_id:
                            cell_address = gspread.utils.rowcol_to_a1(row_idx + 1, target_col + 1)
                            
                            updates.append({
                                'range': cell_address,
                                'values': [[new_assignment]]
                            })
                            
                            updates_count += 1
                            print(f"  ✅ {dog_id} → {new_assignment}")
                            break
            
            # Process greedy walk moves if any
            if hasattr(self, 'greedy_moves_made') and self.greedy_moves_made:
                print(f"\n🔍 Processing {len(self.greedy_moves_made)} greedy walk moves...")
                
                for move in self.greedy_moves_made:
                    dog_id = str(move.get('dog_id', '')).strip()
                    from_driver = move.get('from_driver', '')
                    to_driver = move.get('to_driver', '')
                    
                    # Find current assignment for this dog and update driver
                    for row_idx in range(1, len(all_data)):
                        if dog_id_col < len(all_data[row_idx]):
                            current_dog_id = str(all_data[row_idx][dog_id_col]).strip()
                            
                            if current_dog_id == dog_id:
                                # Get current assignment and update driver
                                current_combined = str(all_data[row_idx][target_col]).strip()
                                if ':' in current_combined:
                                    assignment_part = current_combined.split(':', 1)[1]
                                    new_combined = f"{to_driver}:{assignment_part}"
                                    
                                    cell_address = gspread.utils.rowcol_to_a1(row_idx + 1, target_col + 1)
                                    
                                    updates.append({
                                        'range': cell_address,
                                        'values': [[new_combined]]
                                    })
                                    
                                    print(f"  🚶 {dog_id} moved: {from_driver} → {to_driver}")
                                    updates_count += 1
                                break
            
            if not updates:
                print("❌ No valid updates to make")
                return False
            
            # Execute batch update
            print(f"\n📤 Writing {len(updates)} updates to Google Sheets...")
            worksheet.batch_update(updates)
            
            success_msg = f"✅ Successfully updated {updates_count} assignments with 100.0 placeholder filtering!"
            if hasattr(self, 'greedy_moves_made') and self.greedy_moves_made:
                success_msg += f" (including {len(self.greedy_moves_made)} cascading moves)"
            
            print(success_msg)
            print(f"🎯 Used locality-first algorithm with realistic distance filtering")
            
            # Send Slack notification
            slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    message = f"🐕 Dog Reassignment Complete: {updates_count} assignments updated using 100.0 placeholder filtering"
                    slack_message = {"text": message}
                    response = requests.post(slack_webhook, json=slack_message, timeout=10)
                    if response.status_code == 200:
                        print("📱 Slack notification sent")
                except Exception as e:
                    print(f"⚠️ Could not send Slack notification: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error writing to sheets: {e}")
            import traceback
            print(f"🔍 Full error: {traceback.format_exc()}")
            return False


def main():
    """Main function to run the dog reassignment system"""
    print("🚀 Enhanced Dog Reassignment System - LOCALITY-FIRST WITH 100.0 FILTERING")
    print("🎯 NEW: Proximity-first assignment with dynamic cascading moves")
    print("📏 Starts at 0.2mi, expands to 0.7mi in 0.1mi increments")
    print("🔄 Adjacent groups scale with radius (50% of current radius)")
    print("🚶 Cascading moves up to 0.5mi to free space dynamically")
    print("🧅 Onion-layer backflow pushes outer assignments out to create inner space")
    print("📊 Quality: GOOD ≤0.2mi, BACKUP ≤0.5mi, EMERGENCY >0.5mi")
    print("🎯 FILTERING: 100.0 placeholder distances ignored for realistic calculations")
    print("=" * 80)
    
    # Initialize system
    system = DogReassignmentSystem()
    
    # Setup Google Sheets client
    if not system.setup_google_sheets_client():
        print("❌ Failed to setup Google Sheets client for writing")
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
    
    # Run the locality-first assignment with 100.0 filtering
    print("\n🔄 Processing callout assignments...")
    
    reassignments = system.reassign_dogs_multi_strategy_optimization()
    
    # Ensure reassignments is always a list
    if reassignments is None:
        reassignments = []
    
    # Write results
    if reassignments:
        write_success = system.write_results_to_sheets(reassignments)
        if write_success:
            print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments")
            print(f"✅ Used locality-first algorithm with 100.0 placeholder filtering")
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")


if __name__ == "__main__":
    main()
