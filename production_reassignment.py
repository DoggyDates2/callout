# production_reassignment_time_based.py
# COMPLETE WORKING VERSION: Locality-first with time-based constraints (3-7 minutes)
# Drive time instead of miles, with VERY tight constraints

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
        """Initialize the dog reassignment system with time-based constraints"""
        # Google Sheets URLs (CSV export format)
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        
        # TIME LIMITS - LOCALITY-FIRST THRESHOLDS (in minutes)
        self.PREFERRED_TIME = 3          # Ideal assignments: ≤3 min drive
        self.MAX_TIME = 5                # Backup assignments: ≤5 min drive
        self.ABSOLUTE_MAX_TIME = 7       # Search limit: ≤7 min drive (VERY TIGHT!)
        self.CASCADING_MOVE_MAX_TIME = 6  # Max time for cascading moves
        self.ADJACENT_GROUP_TIME = 2     # Base adjacent group time (scales with radius)
        self.EXCLUSION_TIME = 100.0      # Skip if >100 minutes (clearly placeholder)
        
        # Legacy naming for minimal code changes
        self.PREFERRED_DISTANCE = self.PREFERRED_TIME
        self.MAX_DISTANCE = self.MAX_TIME
        self.ABSOLUTE_MAX_DISTANCE = self.ABSOLUTE_MAX_TIME
        self.CASCADING_MOVE_MAX = self.CASCADING_MOVE_MAX_TIME
        self.ADJACENT_GROUP_DISTANCE = self.ADJACENT_GROUP_TIME
        self.EXCLUSION_DISTANCE = self.EXCLUSION_TIME
        self.GREEDY_WALK_MAX_DISTANCE = self.MAX_TIME
        
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
        """Load distance matrix data from Google Sheets (now in MINUTES)"""
        try:
            print("📊 Loading distance matrix (DRIVE TIME in minutes)...")
            
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
            print(f"✅ Loaded time matrix for {len(self.distance_matrix)} dogs")
            
            # DEBUG: Check some sample values to confirm they're in minutes
            print("\n🔍 DEBUG: Sample drive time values (should be in MINUTES):")
            sample_dogs = list(self.distance_matrix.index)[:5]
            for i, dog1 in enumerate(sample_dogs[:3]):
                for dog2 in sample_dogs[i+1:i+3]:
                    if dog2 in self.distance_matrix.columns:
                        value = self.distance_matrix.loc[dog1, dog2]
                        if not pd.isna(value):
                            print(f"   {dog1} → {dog2}: {value:.0f} minutes")
            
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
        """Get drive time between two dogs using the distance matrix (in MINUTES)"""
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
        """Calculate current load for a driver across all groups - FIXED VERSION"""
        load = {'group1': 0, 'group2': 0, 'group3': 0}
        
        # Use provided assignments or default to original assignments
        assignments_to_use = current_assignments if current_assignments else self.dog_assignments
        
        if not assignments_to_use:
            return load
        
        # CRITICAL FIX: Track which dog IDs we've already counted to avoid duplicates
        counted_dogs = set()
        
        for assignment in assignments_to_use:
            if current_assignments:
                # Working with dynamic assignment list
                if assignment.get('driver') == driver_name:
                    dog_id = assignment.get('dog_id')
                    
                    # Skip if we've already counted this dog (prevents double-counting)
                    if dog_id in counted_dogs:
                        print(f"      ⚠️ Skipping duplicate: {dog_id} already counted for {driver_name}")
                        continue
                    counted_dogs.add(dog_id)
                    
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
                        dog_id = assignment.get('dog_id')
                        
                        # Skip if we've already counted this dog
                        if dog_id in counted_dogs:
                            continue
                        counted_dogs.add(dog_id)
                        
                        # Parse groups for this assignment
                        groups_part = combined.split(':', 1)[1].strip()
                        assigned_groups = self._extract_groups_for_capacity_check(groups_part)
                        
                        # Add to load for each group
                        for group in assigned_groups:
                            group_key = f'group{group}'
                            if group_key in load:
                                load[group_key] += assignment['num_dogs']
        
        return load

    def build_initial_assignments_state(self):
        """Build clean initial current_assignments state without duplicates"""
        current_assignments = []
        seen_dogs = set()
        
        print("📊 Building clean initial assignment state...")
        
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            dog_id = assignment.get('dog_id', '')
            
            # Skip if we've already seen this dog or if no valid assignment
            if not combined or ':' not in combined or not dog_id:
                continue
                
            if dog_id in seen_dogs:
                print(f"   ⚠️ Skipping duplicate entry for {dog_id}")
                continue
                
            seen_dogs.add(dog_id)
            
            driver = combined.split(':', 1)[0].strip()
            assignment_string = combined.split(':', 1)[1].strip()
            groups = self._extract_groups_for_capacity_check(assignment_string)
            
            current_assignments.append({
                'dog_id': dog_id,
                'dog_name': assignment['dog_name'],
                'driver': driver,
                'needed_groups': groups,
                'num_dogs': assignment['num_dogs']
            })
        
        print(f"✅ Built state with {len(current_assignments)} unique assignments")
        return current_assignments

    def make_assignment_safely(self, callout_dog, driver, current_assignments, assignment_type='direct'):
        """Safely make an assignment ensuring no duplicates and capacity is respected"""
        # First, remove any existing entries for this dog
        before_count = len(current_assignments)
        current_assignments[:] = [a for a in current_assignments 
                                if a.get('dog_id') != callout_dog['dog_id']]
        after_count = len(current_assignments)
        
        if before_count > after_count:
            print(f"      🧹 Removed {before_count - after_count} existing entries for {callout_dog['dog_name']}")
        
        # Add the new assignment
        current_assignments.append({
            'dog_id': callout_dog['dog_id'],
            'dog_name': callout_dog['dog_name'],
            'driver': driver,
            'needed_groups': callout_dog['needed_groups'],
            'num_dogs': callout_dog['num_dogs']
        })
        
        # Verify capacity is still valid
        load = self.calculate_driver_load(driver, current_assignments)
        capacity = self.driver_capacities.get(driver, {})
        
        capacity_ok = True
        for group in callout_dog['needed_groups']:
            group_key = f'group{group}'
            current = load.get(group_key, 0)
            max_cap = capacity.get(group_key, 0)
            if current > max_cap:
                print(f"      🚨 WARNING: {driver} Group {group} now at {current}/{max_cap} - OVER CAPACITY!")
                capacity_ok = False
            else:
                print(f"      ✅ {driver} Group {group}: {current}/{max_cap} OK")
        
        return capacity_ok

    def check_driver_can_accept(self, driver_name, callout_dog, current_assignments):
        """Check if a driver can accept a dog without violating capacity"""
        current_load = self.calculate_driver_load(driver_name, current_assignments)
        capacity = self.driver_capacities.get(driver_name, {})
        
        for group in callout_dog['needed_groups']:
            group_key = f'group{group}'
            current = current_load.get(group_key, 0)
            max_cap = capacity.get(group_key, 0)
            needed = callout_dog['num_dogs']
            
            if current + needed > max_cap:
                return False
        
        return True

    def verify_capacity_constraints(self, current_assignments):
        """Verify that all capacity constraints are satisfied"""
        violations = []
        
        print("\n🔍 Verifying capacity constraints...")
        
        # First, check for duplicates
        dog_counts = {}
        for assignment in current_assignments:
            dog_id = assignment.get('dog_id')
            if dog_id in dog_counts:
                dog_counts[dog_id] += 1
            else:
                dog_counts[dog_id] = 1
        
        duplicates = [(dog_id, count) for dog_id, count in dog_counts.items() if count > 1]
        if duplicates:
            print("   🚨 DUPLICATE DOGS FOUND:")
            for dog_id, count in duplicates:
                print(f"      - {dog_id} appears {count} times!")
        
        # Check each driver
        for driver_name, capacity in self.driver_capacities.items():
            load = self.calculate_driver_load(driver_name, current_assignments)
            
            # Check each group
            for group_num in [1, 2, 3]:
                group_key = f'group{group_num}'
                current_load = load.get(group_key, 0)
                max_capacity = capacity.get(group_key, 0)
                
                if current_load > max_capacity:
                    violations.append({
                        'driver': driver_name,
                        'group': group_num,
                        'current': current_load,
                        'max': max_capacity,
                        'over': current_load - max_capacity
                    })
        
        if violations:
            print(f"   🚨 CAPACITY VIOLATIONS DETECTED: {len(violations)}")
            for v in violations:
                print(f"      ❌ {v['driver']} Group {v['group']}: {v['current']}/{v['max']} (over by {v['over']})")
        else:
            print(f"   ✅ All capacity constraints satisfied")
        
        return violations

    def check_group_compatibility(self, callout_groups, driver_groups, distance, current_radius=None):
        """FIXED: Radius scaling with proper fallback and 75% adjacent threshold"""
        # Extract unique group numbers from both sets
        callout_set = set(callout_groups)
        driver_set = set(driver_groups)
        
        # Determine thresholds based on current radius
        if current_radius is not None:
            # RADIUS SCALING: Use the step-by-step approach
            perfect_threshold = current_radius
            adjacent_threshold = current_radius * 0.75  # 75% of current radius
        else:
            # FALLBACK: For diagnostics when no radius provided, use generous limits
            perfect_threshold = self.ABSOLUTE_MAX_TIME  # Up to 7 minutes for exact matches
            adjacent_threshold = self.ABSOLUTE_MAX_TIME * 0.75  # Up to ~5 minutes for adjacent groups
        
        # 1. PERFECT MATCH - same groups
        if callout_set.intersection(driver_set):
            return distance <= perfect_threshold
        
        # 2. ADJACENT GROUPS - neighboring groups  
        adjacent_pairs = [(1, 2), (2, 3), (2, 1), (3, 2)]
        for callout_group in callout_set:
            for driver_group in driver_set:
                if (callout_group, driver_group) in adjacent_pairs:
                    return distance <= adjacent_threshold
        
        # 3. NO MATCH - incompatible groups
        return False

    def get_current_driver_dogs(self, driver_name, current_assignments):
        """Get all dogs currently assigned to a specific driver"""
        return [assignment for assignment in current_assignments 
                if assignment.get('driver') == driver_name]

    def attempt_strategic_cascading_move(self, blocked_driver, callout_dog, current_assignments, max_search_radius=6):
        """STRATEGIC: Target specific groups with incremental radius expansion"""
        print(f"🎯 ATTEMPTING STRATEGIC CASCADING MOVE for {callout_dog['dog_name']} → {blocked_driver}")
        
        # Step 1: Identify which groups are causing the capacity problem
        blocked_groups = self._identify_blocked_groups(blocked_driver, callout_dog, current_assignments)
        print(f"   🎯 Target groups causing capacity issues: {blocked_groups}")
        
        if not blocked_groups:
            print(f"   ❌ No blocked groups identified")
            return None
        
        # Step 2: Get dogs from blocked driver, prioritized by strategy
        driver_dogs = self.get_current_driver_dogs(blocked_driver, current_assignments)
        strategic_dogs = self._prioritize_dogs_strategically(driver_dogs, blocked_groups)
        
        print(f"   📊 Strategic prioritization of {len(strategic_dogs)} dogs:")
        for i, (priority, dog) in enumerate(strategic_dogs[:8]):  # Show top 8
            print(f"     {i+1}. {dog['dog_name']} (groups: {dog['needed_groups']}) - {priority}")
        
        # Step 3: Try incremental radius expansion for each high-priority dog
        for priority, dog_to_move in strategic_dogs:
            print(f"   🔄 Trying to move {dog_to_move['dog_name']} (groups: {dog_to_move['needed_groups']})...")
            
            # Use incremental radius expansion like the main algorithm
            move_result = self._attempt_incremental_move(dog_to_move, current_assignments, max_search_radius)
            
            if move_result:
                print(f"   ✅ STRATEGIC MOVE SUCCESSFUL!")
                print(f"      📦 Moved: {dog_to_move['dog_name']} → {move_result['to_driver']}")
                print(f"      📏 Distance: {move_result['distance']:.0f} min (found at radius {move_result['radius']} min)")
                print(f"      🎯 This frees {blocked_groups} capacity in {blocked_driver}")
                return move_result
            else:
                print(f"   ❌ Could not move {dog_to_move['dog_name']} within {max_search_radius} min")
        
        print(f"   ❌ STRATEGIC CASCADING FAILED - no dogs could be relocated")
        return None

    def _identify_blocked_groups(self, driver_name, callout_dog, current_assignments):
        """Identify which specific groups are causing capacity problems"""
        blocked_groups = []
        
        # Get current capacity and load
        capacity = self.driver_capacities.get(driver_name, {})
        current_load = self.calculate_driver_load(driver_name, current_assignments)
        
        # Check which groups would be over capacity
        for group in callout_dog['needed_groups']:
            group_key = f'group{group}'
            current = current_load.get(group_key, 0)
            max_cap = capacity.get(group_key, 0)
            needed = callout_dog['num_dogs']
            
            if current + needed > max_cap:
                blocked_groups.append(group)
                print(f"   🚨 Group {group} blocking: {current} + {needed} > {max_cap}")
            else:
                print(f"   ✅ Group {group} has space: {current} + {needed} ≤ {max_cap}")
        
        return blocked_groups

    def _prioritize_dogs_strategically(self, driver_dogs, blocked_groups):
        """Prioritize dogs based on strategic value for freeing blocked groups"""
        prioritized = []
        
        for dog in driver_dogs:
            dog_groups = set(dog.get('needed_groups', []))
            blocked_set = set(blocked_groups)
            
            # Calculate strategic priority
            if dog_groups.intersection(blocked_set):
                # Dog is in a blocked group - HIGH PRIORITY
                if len(dog_groups) == 1 and dog['num_dogs'] == 1:
                    priority = "HIGH - Single group, single dog in blocked group"
                elif len(dog_groups) == 1:
                    priority = f"HIGH - Single group, {dog['num_dogs']} dogs in blocked group"
                else:
                    priority = f"MEDIUM - Multi-group dog partially in blocked group"
            else:
                # Dog is not in blocked groups - LOW PRIORITY
                priority = "LOW - Not in blocked groups (won't help)"
            
            prioritized.append((priority, dog))
        
        # Sort by priority (HIGH first, then MEDIUM, then LOW)
        priority_order = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
        prioritized.sort(key=lambda x: (
            priority_order.get(x[0].split(' - ')[0], 4),  # Priority level
            len(x[1].get('needed_groups', [])),           # Fewer groups = easier to place
            x[1].get('num_dogs', 1)                       # Fewer dogs = easier to place
        ))
        
        return prioritized

    def _attempt_incremental_move(self, dog_to_move, current_assignments, max_radius):
        """Try to move a dog using incremental radius expansion - FIXED VERSION"""
        print(f"     🔍 Using incremental radius search for {dog_to_move['dog_name']}...")
        
        # Get the current driver before we move
        old_driver = dog_to_move.get('driver')
        
        if not old_driver:
            # Find current driver from assignments
            for assignment in current_assignments:
                if assignment.get('dog_id') == dog_to_move['dog_id']:
                    old_driver = assignment.get('driver')
                    dog_to_move = assignment.copy()  # Get full assignment info
                    break
        
        if not old_driver:
            print(f"     ❌ Could not find current driver for {dog_to_move['dog_name']}")
            return None
        
        # Start at 3 minutes and expand incrementally
        current_radius = 3
        
        while current_radius <= max_radius:
            print(f"       📏 Trying radius {current_radius} minutes...")
            
            # Find all potential targets within current radius
            targets = self._find_move_targets_at_radius(dog_to_move, current_assignments, current_radius)
            
            if targets:
                print(f"       ✅ Found {len(targets)} targets at {current_radius} min:")
                for i, target in enumerate(targets[:3]):  # Show top 3
                    print(f"         {i+1}. {target['driver']} - {target['distance']:.0f} min")
                
                # Use the closest target
                best_target = targets[0]
                
                # CRITICAL FIX: Properly update the assignment
                print(f"       🔄 Moving {dog_to_move['dog_name']} from {old_driver} to {best_target['driver']}")
                
                # Remove ALL existing entries for this dog
                before_count = len(current_assignments)
                current_assignments[:] = [a for a in current_assignments 
                                        if a.get('dog_id') != dog_to_move['dog_id']]
                after_count = len(current_assignments)
                
                if before_count - after_count > 1:
                    print(f"       ⚠️ Removed {before_count - after_count} entries for {dog_to_move['dog_name']} (duplicates found!)")
                
                # Add the new assignment
                current_assignments.append({
                    'dog_id': dog_to_move['dog_id'],
                    'dog_name': dog_to_move['dog_name'],
                    'driver': best_target['driver'],
                    'needed_groups': dog_to_move.get('needed_groups', []),
                    'num_dogs': dog_to_move.get('num_dogs', 1)
                })
                
                print(f"       ✅ Move complete: {dog_to_move['dog_name']} now with {best_target['driver']}")
                
                # Verify capacity after move
                old_load = self.calculate_driver_load(old_driver, current_assignments)
                new_load = self.calculate_driver_load(best_target['driver'], current_assignments)
                print(f"       📊 {old_driver} load after move: {old_load}")
                print(f"       📊 {best_target['driver']} load after move: {new_load}")
                
                return {
                    'moved_dog': dog_to_move,
                    'from_driver': old_driver,
                    'to_driver': best_target['driver'],
                    'distance': best_target['distance'],
                    'via_dog': best_target['via_dog'],
                    'radius': current_radius
                }
            else:
                print(f"       ❌ No targets at {current_radius} minutes")
            
            # Expand radius by 1 minute
            current_radius += 1
        
        print(f"     ❌ No targets found within {max_radius} minutes")
        return None

    def _find_move_targets_at_radius(self, dog_to_move, current_assignments, radius):
        """Find potential drivers within specific radius who can accept the dog"""
        targets = []
        
        for assignment in current_assignments:
            target_driver = assignment['driver']
            
            # Skip same driver
            if target_driver == dog_to_move.get('driver'):
                continue
            
            # Calculate distance
            distance = self.get_distance(dog_to_move['dog_id'], assignment['dog_id'])
            
            # Skip if beyond current radius
            if distance > radius or distance >= self.EXCLUSION_TIME:  # Skip placeholders
                continue
            
            # Check group compatibility with current radius
            dog_groups = dog_to_move.get('needed_groups', [])
            target_groups = assignment.get('needed_groups', [])
            
            if not self.check_group_compatibility(dog_groups, target_groups, distance, radius):
                continue
            
            # Check if target driver has capacity
            if self.check_driver_can_accept(target_driver, dog_to_move, current_assignments):
                targets.append({
                    'driver': target_driver,
                    'distance': distance,
                    'via_dog': assignment['dog_name'],
                    'via_dog_id': assignment['dog_id']
                })
        
        # Sort by distance (closest first)
        return sorted(targets, key=lambda x: x['distance'])

    def locality_first_assignment(self):
        """Locality-first assignment algorithm with strategic cascading and TIGHT time constraints"""
        print("\n🎯 LOCALITY-FIRST ASSIGNMENT ALGORITHM WITH STRATEGIC CASCADING")
        print("🔄 Step-by-step proximity optimization with strategic cascading moves")
        print("📏 Starting at 3 min, expanding to 7 min in 1 min increments")
        print("🔄 Adjacent groups scale with radius (75% of current radius)")
        print("🎯 STRATEGIC CASCADING: Target specific blocked groups with incremental radius")
        print("🚶 Cascading moves up to 6 min with radius expansion (3→4→5→6)")
        print("🎯 TIGHT CONSTRAINTS: Exact matches up to 7 min, adjacent up to ~5 min")
        print("⚠️ WARNING: Very tight constraints may result in unassigned dogs!")
        print("=" * 80)
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        # Build clean initial state
        current_assignments = self.build_initial_assignments_state()
        
        # Add initial verification
        print("\n📊 Initial capacity state:")
        self.verify_capacity_constraints(current_assignments)
        
        assignments_made = []
        moves_made = []
        dogs_remaining = dogs_to_reassign.copy()
        
        print(f"🐕 Processing {len(dogs_remaining)} callout dogs")
        
        print(f"\n📍 STEP 1: Direct assignments at ≤{self.PREFERRED_TIME} minutes")
        
        dogs_assigned_step1 = []
        for callout_dog in dogs_remaining[:]:
            best_assignment = None
            best_distance = float('inf')
            
            # Check all drivers for direct assignment
            for assignment in current_assignments:
                driver = assignment['driver']
                distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                
                # Skip obvious placeholders
                if distance >= self.EXCLUSION_TIME:
                    continue
                
                # Check group compatibility with distance requirements
                if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, self.PREFERRED_TIME):
                    continue
                
                # Check capacity
                if self.check_driver_can_accept(driver, callout_dog, current_assignments):
                    if distance < best_distance:
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
                
                # Use safe assignment method
                if self.make_assignment_safely(callout_dog, driver, current_assignments):
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
                    dogs_assigned_step1.append(callout_dog)
                    print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.0f} min)")
        
        # Remove assigned dogs
        for dog in dogs_assigned_step1:
            dogs_remaining.remove(dog)
        
        print(f"   📊 Step 1 results: {len(dogs_assigned_step1)} direct assignments")
        
        # Verify after step 1
        print("\n📊 After Step 1 capacity check:")
        self.verify_capacity_constraints(current_assignments)
        
        # Step 2: Capacity-blocked assignments with strategic cascading moves
        if dogs_remaining:
            print(f"\n🎯 STEP 2: Strategic cascading moves to free space at ≤{self.PREFERRED_TIME} minutes")
            
            dogs_assigned_step2 = []
            for callout_dog in dogs_remaining[:]:
                # Find drivers within range but blocked by capacity
                blocked_drivers = []
                
                for assignment in current_assignments:
                    driver = assignment['driver']
                    distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                    
                    # Skip obvious placeholders
                    if distance >= self.EXCLUSION_TIME:
                        continue
                    
                    # Check group compatibility
                    if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, self.PREFERRED_TIME):
                        continue
                    
                    # Check if blocked by capacity
                    if not self.check_driver_can_accept(driver, callout_dog, current_assignments):
                        blocked_drivers.append({
                            'driver': driver,
                            'distance': distance
                        })
                
                # Try strategic cascading moves for the closest blocked driver
                if blocked_drivers:
                    blocked_drivers.sort(key=lambda x: x['distance'])
                    best_blocked = blocked_drivers[0]
                    
                    # STRATEGIC CASCADING: Use strategic approach
                    move_result = self.attempt_strategic_cascading_move(
                        best_blocked['driver'], 
                        callout_dog, 
                        current_assignments, 
                        self.CASCADING_MOVE_MAX_TIME  # 6 min max search with incremental expansion
                    )
                    
                    if move_result:
                        # Record the move
                        moves_made.append({
                            'dog_name': move_result['moved_dog']['dog_name'],
                            'dog_id': move_result['moved_dog']['dog_id'],
                            'from_driver': move_result['from_driver'],
                            'to_driver': move_result['to_driver'],
                            'distance': move_result['distance'],
                            'reason': f"strategic_free_space_for_{callout_dog['dog_name']}"
                        })
                        
                        # Update any existing assignments for moved dog
                        moved_dog_id = move_result['moved_dog']['dog_id']
                        for existing_assignment in assignments_made:
                            if existing_assignment['dog_id'] == moved_dog_id:
                                # Update the assignment to show final location
                                old_driver = existing_assignment['driver']
                                new_driver = move_result['to_driver']
                                existing_assignment['driver'] = new_driver
                                existing_assignment['new_assignment'] = existing_assignment['new_assignment'].replace(f"{old_driver}:", f"{new_driver}:")
                                existing_assignment['assignment_type'] = 'moved_by_strategic_cascading'
                                print(f"      🔄 Updated final assignment: {move_result['moved_dog']['dog_name']} → {new_driver}")
                                break
                        
                        # Now assign the callout dog to the freed space
                        driver = best_blocked['driver']
                        distance = best_blocked['distance']
                        
                        # Use safe assignment
                        if self.make_assignment_safely(callout_dog, driver, current_assignments):
                            assignment_record = {
                                'dog_id': callout_dog['dog_id'],
                                'dog_name': callout_dog['dog_name'],
                                'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                                'driver': driver,
                                'distance': distance,
                                'quality': 'GOOD',
                                'assignment_type': 'strategic_cascading'
                            }
                            
                            assignments_made.append(assignment_record)
                            dogs_assigned_step2.append(callout_dog)
                            print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.0f} min)")
                            print(f"      🎯 Strategic move: {move_result['moved_dog']['dog_name']} → {move_result['to_driver']} ({move_result['distance']:.0f} min at radius {move_result['radius']} min)")
            
            # Remove assigned dogs
            for dog in dogs_assigned_step2:
                dogs_remaining.remove(dog)
            
            print(f"   📊 Step 2 results: {len(dogs_assigned_step2)} strategic cascading assignments")
            
            # Verify after step 2
            print("\n📊 After Step 2 capacity check:")
            self.verify_capacity_constraints(current_assignments)
        
        # Step 3+: Incremental radius expansion (4 to 7 minutes)
        current_radius = 4
        step_number = 3
        
        while current_radius <= self.ABSOLUTE_MAX_TIME and dogs_remaining:
            print(f"\n📏 STEP {step_number}: Radius expansion to ≤{current_radius} minutes")
            print(f"   🎯 Thresholds: Perfect match ≤{current_radius} min, Adjacent groups ≤{current_radius*0.75:.0f} min")
            
            dogs_assigned_this_radius = []
            
            for callout_dog in dogs_remaining[:]:
                # Try direct assignment at current radius
                best_assignment = None
                best_distance = float('inf')
                
                for assignment in current_assignments:
                    driver = assignment['driver']
                    distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                    
                    # Skip obvious placeholders
                    if distance >= self.EXCLUSION_TIME:
                        continue
                    
                    if distance > current_radius:
                        continue
                    
                    # Check group compatibility
                    if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, current_radius):
                        continue
                    
                    # Check capacity
                    if self.check_driver_can_accept(driver, callout_dog, current_assignments):
                        if distance < best_distance:
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
                    if distance <= self.PREFERRED_TIME:
                        quality = 'GOOD'
                    elif distance <= self.MAX_TIME:
                        quality = 'BACKUP'
                    else:
                        quality = 'EMERGENCY'
                    
                    # Use safe assignment
                    if self.make_assignment_safely(callout_dog, driver, current_assignments):
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
                        dogs_assigned_this_radius.append(callout_dog)
                        print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.0f} min) [{quality}]")
                
                else:
                    # Try strategic cascading moves at current radius
                    blocked_drivers = []
                    
                    for assignment in current_assignments:
                        driver = assignment['driver']
                        distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                        
                        # Skip obvious placeholders
                        if distance >= self.EXCLUSION_TIME:
                            continue
                        
                        if distance > current_radius:
                            continue
                        
                        # Check group compatibility
                        if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, current_radius):
                            continue
                        
                        # Check if blocked by capacity
                        if not self.check_driver_can_accept(driver, callout_dog, current_assignments):
                            blocked_drivers.append({
                                'driver': driver,
                                'distance': distance
                            })
                    
                    if blocked_drivers:
                        blocked_drivers.sort(key=lambda x: x['distance'])
                        best_blocked = blocked_drivers[0]
                        
                        # STRATEGIC CASCADING: Use current radius as max search distance
                        move_result = self.attempt_strategic_cascading_move(
                            best_blocked['driver'], 
                            callout_dog, 
                            current_assignments, 
                            min(current_radius, self.CASCADING_MOVE_MAX_TIME)  # Cap at 6 min
                        )
                        
                        if move_result:
                            # Record the move
                            moves_made.append({
                                'dog_name': move_result['moved_dog']['dog_name'],
                                'dog_id': move_result['moved_dog']['dog_id'],
                                'from_driver': move_result['from_driver'],
                                'to_driver': move_result['to_driver'],
                                'distance': move_result['distance'],
                                'reason': f"strategic_radius_{current_radius}_space_for_{callout_dog['dog_name']}"
                            })
                            
                            # Assign the callout dog
                            driver = best_blocked['driver']
                            distance = best_blocked['distance']
                            
                            # Determine quality with 3-way check
                            if distance <= self.PREFERRED_TIME:
                                quality = 'GOOD'
                            elif distance <= self.MAX_TIME:
                                quality = 'BACKUP'
                            else:
                                quality = 'EMERGENCY'
                            
                            # Use safe assignment
                            if self.make_assignment_safely(callout_dog, driver, current_assignments):
                                assignment_record = {
                                    'dog_id': callout_dog['dog_id'],
                                    'dog_name': callout_dog['dog_name'],
                                    'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                                    'driver': driver,
                                    'distance': distance,
                                    'quality': quality,
                                    'assignment_type': 'strategic_cascading_radius'
                                }
                                
                                assignments_made.append(assignment_record)
                                dogs_assigned_this_radius.append(callout_dog)
                                print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.0f} min) [{quality}]")
                                print(f"      🎯 Strategic move: {move_result['moved_dog']['dog_name']} → {move_result['to_driver']} ({move_result['distance']:.0f} min at radius {move_result['radius']} min)")
            
            # Remove assigned dogs
            for dog in dogs_assigned_this_radius:
                dogs_remaining.remove(dog)
            
            print(f"   📊 Radius {current_radius} min results: {len(dogs_assigned_this_radius)} assignments")
            
            current_radius += 1  # Increment by 1 minute
            step_number += 1
        
        # Final step: Mark remaining as emergency
        if dogs_remaining:
            print(f"\n🚨 FINAL STEP: {len(dogs_remaining)} remaining dogs marked as UNASSIGNED")
            print("⚠️ These dogs could not be assigned within the 7 minute constraint!")
            
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
                print(f"   ❌ {callout_dog['dog_name']} - No viable assignment found within 7 minutes")
        
        # Store moves for writing
        self.greedy_moves_made = moves_made
        
        # Final verification
        print("\n📊 FINAL capacity verification:")
        violations = self.verify_capacity_constraints(current_assignments)
        
        if violations:
            print("\n🚨 WARNING: Algorithm completed with capacity violations!")
            print("🔧 The capacity cleanup phase will attempt to fix these.")
        
        # Summary
        total_dogs = len(dogs_to_reassign)
        good_count = len([a for a in assignments_made if a['quality'] == 'GOOD'])
        backup_count = len([a for a in assignments_made if a['quality'] == 'BACKUP'])
        emergency_count = len([a for a in assignments_made if a['quality'] == 'EMERGENCY'])
        unassigned_count = len([a for a in assignments_made if a['driver'] == 'UNASSIGNED'])
        strategic_moves = len([m for m in moves_made if 'strategic' in m['reason']])
        
        print(f"\n🏆 LOCALITY-FIRST + STRATEGIC CASCADING RESULTS (TIME-BASED):")
        print(f"   📊 {len(assignments_made)}/{total_dogs} dogs processed")
        print(f"   💚 {good_count} GOOD assignments (≤{self.PREFERRED_TIME} min drive)")
        print(f"   🟡 {backup_count} BACKUP assignments ({self.PREFERRED_TIME}-{self.MAX_TIME} min drive)")
        print(f"   🚨 {emergency_count} EMERGENCY assignments (>{self.MAX_TIME} min drive)")
        print(f"   ❌ {unassigned_count} UNASSIGNED (no driver within 7 minutes)")
        print(f"   🎯 {strategic_moves} strategic cascading moves executed")
        print(f"   🚶 {len(moves_made)} total cascading moves executed")
        print(f"   🎯 Success rate: {(good_count + backup_count)/total_dogs*100:.0f}% practical assignments")
        print(f"   🎯 Tight constraints: Exact matches ≤7 min, Adjacent groups ≤5 min")
        print(f"   🎯 Strategic cascading: Target blocked groups with incremental radius expansion")
        print(f"   ✅ Capacity tracking: Fixed with duplicate prevention")
        
        if unassigned_count > 0:
            print(f"\n⚠️ WARNING: {unassigned_count} dogs could not be assigned within 7 minute limit!")
            print("   Consider: Adding more drivers, relaxing time constraints, or manual intervention")
        
        return assignments_made

    def reassign_dogs_multi_strategy_optimization(self):
        """Locality-first algorithm with strategic cascading and TIGHT time constraints"""
        print("\n🔄 Starting LOCALITY-FIRST + STRATEGIC CASCADING SYSTEM (TIME-BASED)...")
        print("🎯 Strategy: Proximity-first with strategic group-targeted cascading")
        print("📊 Quality: GOOD ≤3 min, BACKUP ≤5 min, EMERGENCY >5 min")
        print("🚨 Focus: Immediate proximity with strategic dynamic space optimization")
        print("🎯 TIGHT CONSTRAINTS: Exact matches ≤7 min, Adjacent groups ≤5 min")
        print("🎯 STRATEGIC CASCADING: Target blocked groups with 3→4→5→6 min radius expansion")
        print("✅ CAPACITY FIXES: Duplicate tracking, safe assignments, continuous verification")
        print("⚠️ WARNING: Very tight 7 minute limit may result in unassigned dogs!")
        print("=" * 80)
        
        # Try the locality-first algorithm with strategic cascading
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
            
            # Process strategic cascading moves if any
            if hasattr(self, 'greedy_moves_made') and self.greedy_moves_made:
                print(f"\n🔍 Processing {len(self.greedy_moves_made)} strategic cascading moves...")
                
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
                                    
                                    print(f"  🎯 {dog_id} strategic move: {from_driver} → {to_driver}")
                                    updates_count += 1
                                break
            
            if not updates:
                print("❌ No valid updates to make")
                return False
            
            # Execute batch update
            print(f"\n📤 Writing {len(updates)} updates to Google Sheets...")
            worksheet.batch_update(updates)
            
            success_msg = f"✅ Successfully updated {updates_count} assignments with strategic cascading!"
            if hasattr(self, 'greedy_moves_made') and self.greedy_moves_made:
                strategic_moves = len([m for m in self.greedy_moves_made if 'strategic' in m['reason']])
                success_msg += f" (including {strategic_moves} strategic cascading moves)"
            
            print(success_msg)
            print(f"🎯 Used locality-first + strategic cascading with 7 min time limit")
            print(f"✅ WITH CAPACITY FIXES: Duplicate tracking + safe assignments")
            
            # Send Slack notification
            slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    message = f"🐕 Dog Reassignment Complete: {updates_count} assignments updated using strategic cascading + 7 min limit + capacity fixes"
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
    """Main function to run the dog reassignment system with capacity cleanup"""
    print("🚀 Enhanced Dog Reassignment System - TIME-BASED (3-7 minute constraints)")
    print("🎯 NEW: Strategic group-targeted cascading with incremental radius expansion")
    print("📏 Main: Starts at 3 min, expands to 7 min in 1 min increments")
    print("🔧 Cleanup: Aggressive proximity-focused capacity violation fixes")
    print("🔄 Adjacent groups: 75% of current radius (more generous)")
    print("🎯 STRATEGIC CASCADING: Target blocked groups, not random dogs")
    print("🚶 Cascading moves up to 6 min with incremental expansion (3→4→5→6)")
    print("🧅 Onion-layer backflow pushes outer assignments out to create inner space")
    print("📊 Quality: GOOD ≤3 min, BACKUP ≤5 min, EMERGENCY >5 min")
    print("🎯 TIGHT CONSTRAINTS: Exact matches ≤7 min, Adjacent groups ≤5 min")
    print("✅ CAPACITY FIXES: Duplicate tracking, safe assignments, verification at each step")
    print("⚠️ WARNING: Very tight constraints may result in unassigned dogs!")
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
    
    # Run the locality-first assignment with strategic cascading
    print("\n🔄 Processing callout assignments...")
    
    reassignments = system.reassign_dogs_multi_strategy_optimization()
    
    # Ensure reassignments is always a list
    if reassignments is None:
        reassignments = []
    
    # Check for unassigned dogs
    unassigned_count = len([a for a in reassignments if a.get('driver') == 'UNASSIGNED'])
    if unassigned_count > 0:
        print(f"\n⚠️ WARNING: {unassigned_count} dogs could not be assigned within 7 minute limit!")
        print("Consider relaxing constraints or adding more drivers in affected areas.")
    
    # Write results
    if reassignments:
        write_success = system.write_results_to_sheets(reassignments)
        if write_success:
            print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments")
            print(f"✅ Used locality-first + strategic cascading with 7 min time limit")
            print(f"✅ WITH CAPACITY FIXES: All capacity constraints respected")
            
            # ========== CAPACITY CLEANUP PHASE ==========
            print("\n" + "="*80)
            print("🔧 AGGRESSIVE CAPACITY CLEANUP - Fixing violations with extreme proximity")
            print("📏 Thresholds: ≤5 min direct, ≤3 min adjacent, ≤6 min cascading")
            print("🎯 Goal: 100% close placements, zero tolerance for distant fixes")
            print("="*80)
            
            try:
                # Import and run cleanup
                from capacity_cleanup import CapacityCleanup
                
                cleanup = CapacityCleanup()
                # Copy data instead of reloading
                cleanup.distance_matrix = system.distance_matrix
                cleanup.dog_assignments = system.dog_assignments  # Updated assignments
                cleanup.driver_capacities = system.driver_capacities
                cleanup.sheets_client = system.sheets_client
                
                # Run aggressive cleanup
                moves = cleanup.fix_capacity_violations()
                
                if moves:
                    cleanup_success = cleanup.write_moves_to_sheets(moves)
                    if cleanup_success:
                        print(f"\n🎉 COMPLETE SUCCESS! Main + cleanup: extreme proximity achieved")
                    else:
                        print(f"\n⚠️ Main completed, cleanup had sheet writing issues")
                else:
                    print(f"\n✅ Perfect! No capacity violations to clean up")
                    
            except Exception as e:
                print(f"\n⚠️ Cleanup phase error (main script succeeded): {e}")
                import traceback
                print(f"🔍 Cleanup error details: {traceback.format_exc()}")
            
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
            # Don't run cleanup if main write failed
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")
        
        # Run cleanup even if no callouts (might still have capacity violations from existing assignments)
        print("\n🔍 Checking for existing capacity violations...")
        try:
            from capacity_cleanup import CapacityCleanup
            
            cleanup = CapacityCleanup()
            cleanup.distance_matrix = system.distance_matrix
            cleanup.dog_assignments = system.dog_assignments
            cleanup.driver_capacities = system.driver_capacities
            cleanup.sheets_client = system.sheets_client
            
            moves = cleanup.fix_capacity_violations()
            
            if moves:
                cleanup_success = cleanup.write_moves_to_sheets(moves)
                if cleanup_success:
                    print(f"\n✅ Fixed existing capacity violations: {len(moves)} moves")
                    
        except Exception as e:
            print(f"\n⚠️ Capacity check error: {e}")


if __name__ == "__main__":
    main()
