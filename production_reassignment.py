# production_reassignment.py
# COMPLETE WORKING VERSION: Locality-first assignment with cascading moves + DEBUG

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
        self.ADJACENT_GROUP_DISTANCE = 0.1  # Adjacent groups need to be twice as close
        self.EXCLUSION_DISTANCE = 100.0  # Recognize 100+ as "do not assign" placeholders
        
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
        """FIXED: Get distance between two dogs using the distance matrix"""
        try:
            if self.distance_matrix is None:
                return float('inf')
            
            if dog1_id in self.distance_matrix.index and dog2_id in self.distance_matrix.columns:
                distance = self.distance_matrix.loc[dog1_id, dog2_id]
                return float(distance) if not pd.isna(distance) else float('inf')
            
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

    def check_group_compatibility(self, callout_groups, driver_groups, distance):
        """Check if groups are compatible considering adjacent group penalty"""
        # Extract unique group numbers from both sets
        callout_set = set(callout_groups)
        driver_set = set(driver_groups)
        
        # Perfect match - same groups
        if callout_set.intersection(driver_set):
            return distance <= self.PREFERRED_DISTANCE
        
        # Adjacent groups - need to be twice as close
        adjacent_pairs = [(1, 2), (2, 3), (2, 1), (3, 2)]
        for callout_group in callout_set:
            for driver_group in driver_set:
                if (callout_group, driver_group) in adjacent_pairs:
                    return distance <= self.ADJACENT_GROUP_DISTANCE
        
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
            
            if not self.check_group_compatibility(dog_groups, target_groups, distance):
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
        driver_dogs = self.get_current_driver_dogs(blocked_driver, current_assignments)
        
        # Try to move each dog, starting with single-group dogs (easier to place)
        move_candidates = sorted(driver_dogs, key=lambda x: (len(x.get('needed_groups', [])), x.get('num_dogs', 1)))
        
        for dog_to_move in move_candidates:
            # Find targets for this dog within cascade distance
            targets = self.find_move_targets_for_dog(dog_to_move, current_assignments, max_cascade_distance)
            
            if targets:
                best_target = targets[0]
                
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
        
        return None

    def locality_first_assignment(self):
        """NEW: Locality-first assignment algorithm with cascading moves"""
        print("\n🎯 LOCALITY-FIRST ASSIGNMENT ALGORITHM")
        print("🔄 Step-by-step proximity optimization with dynamic state updates")
        print("📏 Starting at 0.2mi, expanding to 0.7mi in 0.1mi increments")
        print("🔄 Adjacent groups require 0.1mi proximity (2x closer)")
        print("🚶 Cascading moves up to 0.5mi to free space")
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
        
        # Step 1: Direct assignments at 0.2 miles
        print(f"\n📍 STEP 1: Direct assignments at ≤{self.PREFERRED_DISTANCE}mi")
        
        dogs_assigned_step1 = []
        for callout_dog in dogs_remaining[:]:
            best_assignment = None
            best_distance = float('inf')
            
            # Check all drivers for direct assignment
            for assignment in current_assignments:
                driver = assignment['driver']
                distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                
                # Check group compatibility with distance requirements
                if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance):
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
                    
                    # Check group compatibility
                    if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance):
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
            
            dogs_assigned_this_radius = []
            
            for callout_dog in dogs_remaining[:]:
                # Try direct assignment at current radius
                best_assignment = None
                best_distance = float('inf')
                
                for assignment in current_assignments:
                    driver = assignment['driver']
                    distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                    
                    if distance > current_radius:
                        continue
                    
                    # Check group compatibility
                    if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance):
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
                        
                        if distance > current_radius:
                            continue
                        
                        # Check group compatibility
                        if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance):
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
        
        print(f"\n🏆 LOCALITY-FIRST ALGORITHM RESULTS:")
        print(f"   📊 {len(assignments_made)}/{total_dogs} dogs processed")
        print(f"   💚 {good_count} GOOD assignments (≤{self.PREFERRED_DISTANCE}mi)")
        print(f"   🟡 {backup_count} BACKUP assignments ({self.PREFERRED_DISTANCE}-{self.MAX_DISTANCE}mi)")
        print(f"   🚨 {emergency_count} EMERGENCY assignments (>{self.MAX_DISTANCE}mi)")
        print(f"   🚶 {len(moves_made)} cascading moves executed")
        print(f"   🎯 Success rate: {(good_count + backup_count)/total_dogs*100:.0f}% practical assignments")
        
        return assignments_made

    def analyze_dog_constraints(self, dogs_to_reassign, current_assignments):
        """NEW: Analyze how many viable driver options each dog has - with quality tiers"""
        print(f"\n🔍 ANALYZING ASSIGNMENT CONSTRAINTS for {len(dogs_to_reassign)} dogs...")
        print(f"🎯 QUALITY THRESHOLDS: ≤{self.PREFERRED_DISTANCE}mi = GOOD, ≤{self.MAX_DISTANCE}mi = BACKUP")
        
        dog_options = []
        
        for dog in dogs_to_reassign:
            good_options = []  # ≤0.5 miles
            backup_options = []  # 0.5-1.0 miles
            
            # Find all drivers with capacity and within range
            for assignment in current_assignments:
                driver = assignment['driver']
                
                # Calculate distance to this dog
                distance = self.get_distance(dog['dog_id'], assignment['dog_id'])
                
                # Skip if too far or invalid
                if distance > self.ABSOLUTE_MAX_DISTANCE or distance >= self.EXCLUSION_DISTANCE:
                    continue
                
                # Check if this driver has capacity in ALL needed groups
                current_load = self.calculate_driver_load(driver, current_assignments)
                has_capacity = True
                
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    current = current_load.get(group_key, 0)
                    max_cap = self.driver_capacities.get(driver, {}).get(group_key, 0)
                    needed = dog['num_dogs']
                    
                    if current + needed > max_cap:
                        has_capacity = False
                        break
                
                if has_capacity:
                    option = {
                        'driver': driver,
                        'distance': distance,
                        'via_dog': assignment['dog_name']
                    }
                    
                    if distance <= self.PREFERRED_DISTANCE:
                        good_options.append(option)
                    elif distance <= self.MAX_DISTANCE:
                        backup_options.append(option)
            
            # Remove duplicates within each tier - keep closest
            def dedupe_options(options_list):
                driver_best = {}
                for candidate in options_list:
                    driver = candidate['driver']
                    if driver not in driver_best or candidate['distance'] < driver_best[driver]['distance']:
                        driver_best[driver] = candidate
                return list(driver_best.values())
            
            good_options = dedupe_options(good_options)
            backup_options = dedupe_options(backup_options)
            
            all_options = good_options + backup_options
            
            dog_options.append({
                'dog': dog,
                'good_options': len(good_options),
                'backup_options': len(backup_options),
                'total_options': len(all_options),
                'best_good': sorted(good_options, key=lambda x: x['distance'])[:2] if good_options else [],
                'best_backup': sorted(backup_options, key=lambda x: x['distance'])[:2] if backup_options else [],
                'min_distance': min([opt['distance'] for opt in all_options]) if all_options else float('inf'),
                'has_good_option': len(good_options) > 0
            })
        
        # Sort by constraint level: good options first, then total options, then distance
        dog_options.sort(key=lambda x: (x['good_options'], x['total_options'], x['min_distance']))
        
        print(f"📊 CONSTRAINT ANALYSIS (most constrained first):")
        good_count = sum(1 for opt in dog_options if opt['has_good_option'])
        print(f"💚 {good_count}/{len(dog_options)} dogs have GOOD options (≤{self.PREFERRED_DISTANCE}mi)")
        print(f"🟡 {len(dog_options) - good_count} dogs need BACKUP assignments (>{self.PREFERRED_DISTANCE}mi)")
        print()
        
        for i, option in enumerate(dog_options):
            dog = option['dog']
            good_count = option['good_options']
            backup_count = option['backup_options']
            min_dist = option['min_distance']
            
            # Status emoji based on good options available
            if good_count == 0 and backup_count == 0:
                status_emoji = "🚨"  # No options
            elif good_count == 0:
                status_emoji = "⚠️"   # Only backup options
            elif good_count == 1:
                status_emoji = "🟡"   # One good option
            else:
                status_emoji = "💚"   # Multiple good options
            
            print(f"   {i+1:2d}. {status_emoji} {dog['dog_name']}: {good_count} GOOD + {backup_count} backup")
            print(f"       Groups: {dog['needed_groups']}, Dogs: {dog['num_dogs']}")
            
            if good_count > 0:
                print(f"       💚 GOOD OPTIONS (≤{self.PREFERRED_DISTANCE}mi):")
                for j, opt in enumerate(option['best_good']):
                    print(f"         {j+1}. {opt['driver']} - {opt['distance']:.1f}mi via {opt['via_dog']}")
            
            if backup_count > 0:
                print(f"       🟡 BACKUP OPTIONS ({self.PREFERRED_DISTANCE}-{self.MAX_DISTANCE}mi):")
                for j, opt in enumerate(option['best_backup']):
                    print(f"         {j+1}. {opt['driver']} - {opt['distance']:.1f}mi via {opt['via_dog']}")
            
            if good_count == 0 and backup_count == 0:
                print(f"       ❌ NO VIABLE OPTIONS")
            print()
        
        return dog_options

    def create_ordering_strategies(self, dog_constraints):
        """NEW: Create different ordering strategies for assignment - with quality awareness"""
        
        strategies = {}
        
        # Strategy 1: Most Constrained First (fewest GOOD options first)
        most_constrained = sorted(dog_constraints, key=lambda x: (x['good_options'], x['total_options'], x['min_distance']))
        strategies['most_constrained'] = [opt['dog'] for opt in most_constrained]
        
        # Strategy 2: Least Constrained First (most GOOD options first) 
        least_constrained = sorted(dog_constraints, key=lambda x: (-x['good_options'], -x['total_options'], x['min_distance']))
        strategies['least_constrained'] = [opt['dog'] for opt in least_constrained]
        
        # Strategy 3: Good Options First (dogs with good options get priority)
        good_first = sorted(dog_constraints, key=lambda x: (-x['good_options'], x['min_distance'], x['total_options']))
        strategies['good_first'] = [opt['dog'] for opt in good_first]
        
        # Strategy 4: Shortest Distance First
        shortest_distance = sorted(dog_constraints, key=lambda x: (x['min_distance'], x['good_options']))
        strategies['shortest_distance'] = [opt['dog'] for opt in shortest_distance]
        
        # Strategy 5: Difficulty First (original approach)
        def difficulty_score(dog_constraint):
            dog = dog_constraint['dog']
            score = 0
            if len(dog['needed_groups']) > 1:
                score += len(dog['needed_groups']) * 100
            if dog['num_dogs'] > 1:
                score += dog['num_dogs'] * 50
            return score
        
        difficulty_first = sorted(dog_constraints, key=difficulty_score, reverse=True)
        strategies['difficulty_first'] = [opt['dog'] for opt in difficulty_first]
        
        return strategies

    def _try_greedy_walk(self, callout_dog, current_assignments):
        """Try to make space by moving an existing dog to a very close alternative"""
        print(f"         🚶 Trying greedy walk for {callout_dog['dog_name']}...")
        
        # Find drivers who have the needed groups but are at capacity
        potential_moves = []
        
        for driver, capacity in self.driver_capacities.items():
            # Check if this driver has the needed groups
            has_needed_groups = True
            current_load = self.calculate_driver_load(driver, current_assignments)
            
            for group in callout_dog['needed_groups']:
                group_key = f'group{group}'
                if capacity.get(group_key, 0) == 0:  # Driver doesn't do this group
                    has_needed_groups = False
                    break
            
            if not has_needed_groups:
                continue
            
            # Check if driver would have capacity if we moved one dog
            would_have_capacity_after_move = True
            for group in callout_dog['needed_groups']:
                group_key = f'group{group}'
                # Assume we free up 1 dog worth of space
                if current_load.get(group_key, 0) - 1 + callout_dog['num_dogs'] > capacity.get(group_key, 0):
                    would_have_capacity_after_move = False
                    break
            
            if not would_have_capacity_after_move:
                continue
            
            # Find this driver's current dogs that could be moved
            driver_dogs = [a for a in current_assignments if a.get('driver') == driver]
            
            for driver_dog in driver_dogs:
                # Skip if this dog needs multiple groups (harder to relocate)
                if len(driver_dog.get('needed_groups', [])) > 1:
                    continue
                
                # Skip if this dog has multiple physical dogs (harder to relocate)
                if driver_dog.get('num_dogs', 1) > 1:
                    continue
                
                # Find alternative drivers very close to this dog
                alternatives = []
                
                for alt_assignment in current_assignments:
                    if alt_assignment.get('driver') == driver:  # Skip same driver
                        continue
                    
                    alt_driver = alt_assignment.get('driver')
                    distance = self.get_distance(driver_dog['dog_id'], alt_assignment['dog_id'])
                    
                    if distance > self.GREEDY_WALK_MAX_DISTANCE:
                        continue
                    
                    # Check if alternative driver has capacity for this dog
                    alt_load = self.calculate_driver_load(alt_driver, current_assignments)
                    alt_capacity = self.driver_capacities.get(alt_driver, {})
                    
                    can_take_dog = True
                    for group in driver_dog.get('needed_groups', []):
                        group_key = f'group{group}'
                        if alt_load.get(group_key, 0) + driver_dog.get('num_dogs', 1) > alt_capacity.get(group_key, 0):
                            can_take_dog = False
                            break
                    
                    if can_take_dog:
                        alternatives.append({
                            'driver': alt_driver,
                            'distance': distance,
                            'via_dog': alt_assignment['dog_name']
                        })
                
                if alternatives:
                    # Sort by distance, pick closest
                    alternatives.sort(key=lambda x: x['distance'])
                    best_alt = alternatives[0]
                    
                    potential_moves.append({
                        'original_driver': driver,
                        'dog_to_move': driver_dog,
                        'new_driver': best_alt['driver'],
                        'move_distance': best_alt['distance'],
                        'via_dog': best_alt['via_dog']
                    })
        
        if not potential_moves:
            print(f"         ❌ No greedy walk options found")
            return None
        
        # Sort by move distance - prefer shortest moves
        potential_moves.sort(key=lambda x: x['move_distance'])
        best_move = potential_moves[0]
        
        print(f"         🎯 Found greedy walk option:")
        print(f"           Move {best_move['dog_to_move']['dog_name']} from {best_move['original_driver']} → {best_move['new_driver']}")
        print(f"           Distance: {best_move['move_distance']:.1f}mi via {best_move['via_dog']}")
        print(f"           This frees space in {best_move['original_driver']} for {callout_dog['dog_name']}")
        
        return best_move

    def _run_assignment_with_strategy(self, ordered_dogs, strategy_name):
        """Run assignment algorithm with a specific dog ordering strategy"""
        print(f"\n   🎯 Running strategy: {strategy_name.upper()}")
        
        assignments = []
        moves_made = []
        
        # Build dynamic list of current assignments
        current_assignments = []
        
        # Start with existing assignments
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
        
        # Process each callout dog in the given order
        for callout_dog in ordered_dogs:
            # Find ALL drivers with capacity, calculate distances to their dogs
            candidate_drivers = []
            
            for assignment in current_assignments:
                driver = assignment['driver']
                
                # Calculate distance to this dog
                distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                
                # Skip if too far or invalid
                if distance > self.ABSOLUTE_MAX_DISTANCE or distance >= self.EXCLUSION_DISTANCE:
                    continue
                
                # Check if this driver has capacity in ALL needed groups
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
                
                if has_capacity:
                    candidate_drivers.append({
                        'driver': driver,
                        'distance': distance,
                        'via_dog': assignment['dog_name'],
                        'via_dog_id': assignment['dog_id']
                    })
            
            if candidate_drivers:
                # Remove duplicates (same driver multiple times) - keep closest
                driver_best_distance = {}
                for candidate in candidate_drivers:
                    driver = candidate['driver']
                    if driver not in driver_best_distance or candidate['distance'] < driver_best_distance[driver]['distance']:
                        driver_best_distance[driver] = candidate
                
                unique_candidates = list(driver_best_distance.values())
                
                # Sort by distance - CLOSEST WINS!
                unique_candidates.sort(key=lambda x: x['distance'])
                
                # Pick the closest driver
                winner = unique_candidates[0]
                driver = winner['driver']
                distance = winner['distance']
                
            else:
                # No drivers with available capacity - try greedy walk
                greedy_move = self._try_greedy_walk(callout_dog, current_assignments)
                
                if not greedy_move:
                    # Failed to assign this dog
                    continue
                
                # Execute the greedy walk move
                dog_to_move = greedy_move['dog_to_move']
                original_driver = greedy_move['original_driver']
                new_driver = greedy_move['new_driver']
                
                # Update the moved dog's assignment in current_assignments
                for assignment in current_assignments:
                    if assignment['dog_id'] == dog_to_move['dog_id']:
                        assignment['driver'] = new_driver
                        break
                
                # Record the move
                moves_made.append({
                    'dog_name': dog_to_move['dog_name'],
                    'dog_id': dog_to_move['dog_id'],
                    'from_driver': original_driver,
                    'to_driver': new_driver,
                    'distance': greedy_move['move_distance'],
                    'reason': f"make_space_for_{callout_dog['dog_name']}"
                })
                
                # Now assign the callout dog to the freed space
                driver = original_driver
                distance = self.get_distance(callout_dog['dog_id'], dog_to_move['dog_id'])
            
            # Create the assignment with quality assessment
            new_assignment = f"{driver}:{callout_dog['full_assignment_string']}"
            
            # Determine assignment quality
            if distance <= self.PREFERRED_DISTANCE:
                quality = "GOOD"
                quality_emoji = "💚"
            elif distance <= self.MAX_DISTANCE:
                quality = "BACKUP"
                quality_emoji = "🟡"
            else:
                quality = "EMERGENCY"
                quality_emoji = "🚨"
            
            # Record the assignment
            assignments.append({
                'dog_id': callout_dog['dog_id'],
                'dog_name': callout_dog['dog_name'],
                'new_assignment': new_assignment,
                'driver': driver,
                'distance': distance,
                'quality': quality,
                'assignment_type': 'regular'
            })
            
            # Update current_assignments with the new dog
            current_assignments.append({
                'dog_id': callout_dog['dog_id'],
                'dog_name': callout_dog['dog_name'],
                'driver': driver,
                'needed_groups': callout_dog['needed_groups'],
                'num_dogs': callout_dog['num_dogs']
            })
        
        return assignments, moves_made

    def reassign_dogs_multi_strategy_optimization(self):
        """NEW: Locality-first algorithm with fallback to multi-strategy"""
        print("\n🔄 Starting LOCALITY-FIRST ASSIGNMENT SYSTEM...")
        print("🎯 Strategy: Proximity-first with cascading moves")
        print("📊 Quality: GOOD ≤0.2mi, BACKUP ≤0.5mi, EMERGENCY >0.5mi")
        print("🚨 Focus: Immediate proximity with dynamic space optimization")
        print("=" * 80)
        
        # Try the new locality-first algorithm
        try:
            return self.locality_first_assignment()
        except Exception as e:
            print(f"⚠️ Locality-first algorithm failed: {e}")
            print("🔄 Falling back to multi-strategy optimization...")
            return self._fallback_multi_strategy_optimization()

    def _fallback_multi_strategy_optimization(self):
        """Fallback: Try multiple ordering strategies and pick the best practical result"""
        print("\n🔄 FALLBACK: PRACTICAL MULTI-STRATEGY OPTIMIZATION...")
        print("🎯 Strategy: Maximize assignments ≤0.5mi (GOOD quality)")
        print("📊 Evaluation: Quality-weighted scoring (GOOD=1000, BACKUP=300, EMERGENCY=50)")
        print("🚨 Focus: Which dogs can get truly workable assignments vs need review")
        print("=" * 80)
        
        # Temporarily adjust thresholds for legacy algorithm
        original_preferred = self.PREFERRED_DISTANCE
        original_max = self.MAX_DISTANCE
        
        self.PREFERRED_DISTANCE = 0.5
        self.MAX_DISTANCE = 1.0
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            # Restore original thresholds
            self.PREFERRED_DISTANCE = original_preferred
            self.MAX_DISTANCE = original_max
            return []
        
        num_dogs = len(dogs_to_reassign)
        print(f"📊 Found {num_dogs} callout dogs")
        
        # Build initial current assignments for constraint analysis
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
        
        # Analyze constraints for each dog
        dog_constraints = self.analyze_dog_constraints(dogs_to_reassign, current_assignments)
        
        # Create different ordering strategies
        strategies = self.create_ordering_strategies(dog_constraints)
        
        print(f"\n🧪 TESTING {len(strategies)} ORDERING STRATEGIES (Quality-Focused)...")
        
        strategy_results = {}
        
        for strategy_name, ordered_dogs in strategies.items():
            print(f"\n{'='*60}")
            print(f"🧪 Testing Strategy: {strategy_name.upper()}")
            print(f"📋 Order: {', '.join([dog['dog_name'] for dog in ordered_dogs[:5]])}{'...' if len(ordered_dogs) > 5 else ''}")
            
            start_time = time.time()
            assignments, moves = self._run_assignment_with_strategy(ordered_dogs, strategy_name)
            elapsed_time = time.time() - start_time
            
            # Calculate strategy score with quality weighting
            total_assigned = len(assignments)
            good_assigned = len([a for a in assignments if a.get('quality') == 'GOOD'])
            backup_assigned = len([a for a in assignments if a.get('quality') == 'BACKUP'])
            emergency_assigned = len([a for a in assignments if a.get('quality') == 'EMERGENCY'])
            avg_distance = np.mean([a['distance'] for a in assignments]) if assignments else float('inf')
            total_moves = len(moves)
            
            # Quality-weighted scoring: heavily reward good assignments, penalize backup/emergency
            score = (good_assigned * 1000) + (backup_assigned * 300) + (emergency_assigned * 50) - (total_moves * 5)
            
            strategy_results[strategy_name] = {
                'assignments': assignments,
                'moves': moves,
                'total_assigned': total_assigned,
                'good_assigned': good_assigned,
                'backup_assigned': backup_assigned,
                'emergency_assigned': emergency_assigned,
                'avg_distance': avg_distance,
                'total_moves': total_moves,
                'score': score,
                'elapsed_time': elapsed_time
            }
            
            print(f"   📊 Results: {total_assigned}/{num_dogs} assigned ({good_assigned} GOOD, {backup_assigned} backup, {emergency_assigned} emergency)")
            print(f"   📏 Average distance: {avg_distance:.1f}mi, {total_moves} moves")
            print(f"   🏆 Quality score: {score:.1f} (higher = better)")
            print(f"   ⏱️ Time: {elapsed_time:.1f}s")
        
        # Find the best strategy
        best_strategy = max(strategy_results.items(), key=lambda x: x[1]['score'])
        best_name, best_result = best_strategy
        
        print(f"\n🏆 WINNING STRATEGY: {best_name.upper()}")
        print(f"   📊 {best_result['total_assigned']}/{num_dogs} dogs assigned")
        print(f"   💚 {best_result['good_assigned']} GOOD assignments (≤{self.PREFERRED_DISTANCE}mi)")
        print(f"   🟡 {best_result['backup_assigned']} BACKUP assignments ({self.PREFERRED_DISTANCE}-{self.MAX_DISTANCE}mi)")
        print(f"   🚨 {best_result['emergency_assigned']} EMERGENCY assignments (>{self.MAX_DISTANCE}mi)")
        print(f"   📏 Average distance: {best_result['avg_distance']:.1f} miles")
        print(f"   🚶 Greedy walk moves: {best_result['total_moves']}")
        print(f"   🏆 Quality score: {best_result['score']:.1f}")
        
        # Show comparison table
        print(f"\n📊 STRATEGY COMPARISON:")
        print(f"{'Strategy':<15} {'Total':<6} {'Good':<5} {'Backup':<7} {'Emerg':<6} {'Score':<8}")
        print("-" * 65)
        
        for name, result in sorted(strategy_results.items(), key=lambda x: x[1]['score'], reverse=True):
            total_str = f"{result['total_assigned']}/{num_dogs}"
            winner_mark = "🏆 " if name == best_name else "   "
            
            print(f"{winner_mark}{name:<12} {total_str:<6} {result['good_assigned']:<5} {result['backup_assigned']:<7} {result['emergency_assigned']:<6} {result['score']:<8.0f}")
        
        # Show detailed assignment breakdown
        if best_result['assignments']:
            print(f"\n📋 DETAILED ASSIGNMENT BREAKDOWN:")
            
            good_assignments = [a for a in best_result['assignments'] if a.get('quality') == 'GOOD']
            backup_assignments = [a for a in best_result['assignments'] if a.get('quality') == 'BACKUP']
            emergency_assignments = [a for a in best_result['assignments'] if a.get('quality') == 'EMERGENCY']
            
            if good_assignments:
                print(f"\n💚 GOOD ASSIGNMENTS (≤{self.PREFERRED_DISTANCE}mi) - Ready to deploy:")
                for assignment in good_assignments:
                    print(f"   ✅ {assignment['dog_name']} → {assignment['driver']} ({assignment['distance']:.1f}mi)")
            
            if backup_assignments:
                print(f"\n🟡 BACKUP ASSIGNMENTS ({self.PREFERRED_DISTANCE}-{self.MAX_DISTANCE}mi) - May need review:")
                for assignment in backup_assignments:
                    print(f"   ⚠️ {assignment['dog_name']} → {assignment['driver']} ({assignment['distance']:.1f}mi)")
            
            if emergency_assignments:
                print(f"\n🚨 EMERGENCY ASSIGNMENTS (>{self.MAX_DISTANCE}mi) - Needs manual intervention:")
                for assignment in emergency_assignments:
                    print(f"   ❌ {assignment['dog_name']} → {assignment['driver']} ({assignment['distance']:.1f}mi)")
            
            # Summary recommendation
            practical_count = len(good_assignments)
            print(f"\n🎯 RECOMMENDATION:")
            print(f"   💚 {practical_count}/{num_dogs} dogs have practical assignments (≤{self.PREFERRED_DISTANCE}mi)")
            
            if backup_assignments:
                print(f"   🟡 {len(backup_assignments)} assignments need review (drivers may find distance challenging)")
            
            if emergency_assignments:
                print(f"   🚨 {len(emergency_assignments)} dogs need manual intervention (distance too far for regular assignment)")
            
            if practical_count / num_dogs >= 0.8:
                print(f"   ✅ GOOD SUCCESS RATE: {practical_count/num_dogs*100:.0f}% of dogs have workable assignments")
            elif practical_count / num_dogs >= 0.5:
                print(f"   ⚠️ MODERATE SUCCESS RATE: {practical_count/num_dogs*100:.0f}% of dogs have workable assignments")
            else:
                print(f"   ❌ LOW SUCCESS RATE: Only {practical_count/num_dogs*100:.0f}% of dogs have workable assignments")
                print(f"      Consider expanding driver network or adjusting group boundaries")
        
        # Store moves for potential writing to sheets
        self.greedy_moves_made = best_result['moves']
        
        # Return the best assignments with a practical summary
        practical_assignments = [a for a in best_result['assignments'] if a.get('quality') == 'GOOD']
        
        print(f"\n🎯 PRACTICAL SUMMARY:")
        print(f"   💚 {len(practical_assignments)}/{num_dogs} dogs getting GOOD assignments (≤{self.PREFERRED_DISTANCE}mi)")
        print(f"   🟡 {best_result['backup_assigned']} dogs need backup assignments (review recommended)")
        print(f"   🚨 {best_result['emergency_assigned']} dogs need manual intervention")
        
        # Restore original thresholds
        self.PREFERRED_DISTANCE = original_preferred
        self.MAX_DISTANCE = original_max
        
        return best_result['assignments']

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
            
            success_msg = f"✅ Successfully updated {updates_count} assignments using locality-first optimization!"
            if hasattr(self, 'greedy_moves_made') and self.greedy_moves_made:
                success_msg += f" (including {len(self.greedy_moves_made)} cascading moves)"
            
            print(success_msg)
            print(f"🎯 Used locality-first algorithm with cascading moves")
            
            # Send Slack notification
            slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    message = f"🐕 Dog Reassignment Complete: {updates_count} assignments updated using locality-first + cascading moves"
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
    print("🚀 Enhanced Dog Reassignment System - LOCALITY-FIRST OPTIMIZATION")
    print("🎯 NEW: Proximity-first assignment with dynamic cascading moves")
    print("📏 Starts at 0.2mi, expands to 0.7mi in 0.1mi increments")
    print("🔄 Adjacent groups require 0.1mi (2x closer), multi-group dogs stay together")
    print("🚶 Cascading moves up to 0.5mi to free space dynamically")
    print("🧅 Onion-layer backflow pushes outer assignments out to create inner space")
    print("📊 Quality: GOOD ≤0.2mi, BACKUP ≤0.5mi, EMERGENCY >0.5mi")
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
    
    # Run the locality-first assignment
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
            print(f"✅ Used locality-first algorithm with cascading moves")
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")

if __name__ == "__main__":
    main()
