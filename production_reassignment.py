# production_reassignment.py
# Complete dog reassignment system with smart optimization, 3-mile distance limit, and regular dogs priority

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
        
        # DISTANCE LIMITS - Stricter limits with different thresholds
        self.MAX_DISTANCE = 3.0  # Hard limit: no assignments beyond 3 miles
        self.EXACT_MATCH_MAX_DISTANCE = 3.0  # Exact group matches: 3 miles
        self.ADJACENT_MATCH_MAX_DISTANCE = 0.6  # Adjacent/compatible groups: 0.6 miles (much stricter)
        self.EXCLUSION_DISTANCE = 100.0  # Still recognize 100+ as "do not assign" placeholders
        
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
            
            assignments = []
            
            for _, row in df.iterrows():
                try:
                    # Column positions (0-indexed)
                    dog_name = row.iloc[1] if len(row) > 1 else ""  # Column B
                    combined = row.iloc[7] if len(row) > 7 else ""  # Column H
                    group = row.iloc[8] if len(row) > 8 else ""     # Column I  
                    dog_id = row.iloc[9] if len(row) > 9 else ""    # Column J
                    callout = row.iloc[10] if len(row) > 10 else "" # Column K
                    num_dogs = row.iloc[5] if len(row) > 5 else 1   # Column F (Number of dogs)
                    
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
        """Find dogs that need reassignment (callouts) - preserving full assignment strings"""
        dogs_to_reassign = []
        
        if not self.dog_assignments:
            return dogs_to_reassign
        
        for assignment in self.dog_assignments:
            # Check for callout: Combined column blank AND Callout column has content
            if (not assignment['combined'] or assignment['combined'].strip() == "") and \
               (assignment['callout'] and assignment['callout'].strip() != ""):
                
                # Extract the FULL assignment string (everything after the colon)
                callout_text = assignment['callout'].strip()
                
                if ':' not in callout_text:
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
        
        print(f"🚨 Found {len(dogs_to_reassign)} callouts that need drivers assigned:")
        for dog in dogs_to_reassign:
            print(f"   - {dog['dog_name']} ({dog['dog_id']}) - {dog['num_dogs']} dogs")
            print(f"     Original: {dog['original_callout']}")  
            print(f"     Assignment string: '{dog['full_assignment_string']}'")
            print(f"     Capacity needed in groups: {dog['needed_groups']}")
        
        return dogs_to_reassign

    def _extract_groups_for_capacity_check(self, assignment_string):
        """Extract group numbers for capacity checking - each digit 1,2,3 is a separate group"""
        try:
            print(f"      🔍 Extracting groups from assignment string: '{assignment_string}'")
            
            # Extract all digits that are 1, 2, or 3 from the string
            # Each digit represents a separate group:
            # "23" → [2, 3] (groups 2 and 3)
            # "123" → [1, 2, 3] (groups 1, 2, and 3)  
            # "1&2" → [1, 2] (groups 1 and 2)
            # "2DD2" → [2] (group 2, ignore duplicate and codes)
            # "3LM" → [3] (group 3, ignore codes)
            
            # Find all occurrences of digits 1, 2, or 3
            group_digits = re.findall(r'[123]', assignment_string)
            
            # Convert each digit to an integer and remove duplicates
            groups = sorted(list(set(int(digit) for digit in group_digits)))
            
            print(f"      ✅ Groups for capacity check: {groups}")
            
            # Show the parsing logic for clarity
            if len(group_digits) != len(groups):
                print(f"      📝 Found digits: {group_digits} → Unique groups: {groups}")
            
            return groups
            
        except Exception as e:
            print(f"⚠️ Error extracting groups from '{assignment_string}': {e}")
            return []

    def get_distance(self, dog1_id: str, dog2_id: str) -> float:
        """Get distance between two dogs using the distance matrix"""
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
                    assigned_groups = self._extract_groups_for_capacity_check(groups_part)
                    
                    # Add to load for each group
                    for group in assigned_groups:
                        group_key = f'group{group}'
                        if group_key in load:
                            load[group_key] += assignment['num_dogs']
        
        return load

    def _run_greedy_assignment_with_order(self, ordered_dogs):
        """Run assignment keeping the exact assignment strings, only changing driver names"""
        print("   🎯 Finding drivers with capacity - prioritizing regular dogs over parking/field...")
        
        # Build list of all dogs currently going today (with drivers)
        regular_dogs_going_today = []
        parking_field_dogs_going_today = []
        
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if combined and ':' in combined:
                driver = combined.split(':', 1)[0].strip()
                assignment_string = combined.split(':', 1)[1].strip()
                groups = self._extract_groups_for_capacity_check(assignment_string)
                
                dog_info = {
                    'dog_id': assignment['dog_id'],
                    'dog_name': assignment['dog_name'],
                    'driver': driver,
                    'groups': groups,
                    'num_dogs': assignment['num_dogs']
                }
                
                # Check if this is a parking or field dog
                dog_name = str(assignment['dog_name']).lower().strip()
                if 'parking' in dog_name or 'field' in dog_name:
                    parking_field_dogs_going_today.append(dog_info)
                else:
                    regular_dogs_going_today.append(dog_info)
        
        print(f"       📊 Available: {len(regular_dogs_going_today)} regular dogs, {len(parking_field_dogs_going_today)} parking/field dogs")
        
        assignments = []
        current_driver_loads = {}
        
        # Initialize driver loads
        for driver in self.driver_capacities.keys():
            current_driver_loads[driver] = self.calculate_driver_load(driver)
        
        # Process each callout dog in the specified order
        for callout_dog in ordered_dogs:
            print(f"\n   🐕 Processing {callout_dog['dog_name']}")
            print(f"       Original: {callout_dog['original_callout']}")
            print(f"       Assignment string to preserve: '{callout_dog['full_assignment_string']}'")
            print(f"       Needs capacity in groups: {callout_dog['needed_groups']}")
            print(f"       Physical dogs: {callout_dog['num_dogs']}")
            
            # PRIORITY 1: Try exact matches in regular dogs first (up to 3 miles)
            exact_match_distances = []
            adjacent_match_distances = []
            
            for going_dog in regular_dogs_going_today:
                distance = self.get_distance(callout_dog['dog_id'], going_dog['dog_id'])
                
                # Check if this would be an exact match
                needed_groups = set(callout_dog['needed_groups'])
                current_groups = set(going_dog['groups'])
                is_exact_match = needed_groups == current_groups
                
                if is_exact_match and distance <= self.EXACT_MATCH_MAX_DISTANCE:
                    exact_match_distances.append({
                        'dog_id': going_dog['dog_id'],
                        'dog_name': going_dog['dog_name'],
                        'driver': going_dog['driver'],
                        'groups': going_dog['groups'],
                        'distance': distance,
                        'type': 'regular',
                        'match_type': 'exact'
                    })
                elif not is_exact_match and distance <= self.ADJACENT_MATCH_MAX_DISTANCE:
                    adjacent_match_distances.append({
                        'dog_id': going_dog['dog_id'],
                        'dog_name': going_dog['dog_name'],
                        'driver': going_dog['driver'],
                        'groups': going_dog['groups'],
                        'distance': distance,
                        'type': 'regular',
                        'match_type': 'adjacent'
                    })
            
            # PRIORITY 2: Try exact matches in parking/field dogs (up to 3 miles)
            exact_match_parking_distances = []
            adjacent_match_parking_distances = []
            
            for going_dog in parking_field_dogs_going_today:
                distance = self.get_distance(callout_dog['dog_id'], going_dog['dog_id'])
                
                # Check if this would be an exact match
                needed_groups = set(callout_dog['needed_groups'])
                current_groups = set(going_dog['groups'])
                is_exact_match = needed_groups == current_groups
                
                if is_exact_match and distance <= self.EXACT_MATCH_MAX_DISTANCE:
                    exact_match_parking_distances.append({
                        'dog_id': going_dog['dog_id'],
                        'dog_name': going_dog['dog_name'],
                        'driver': going_dog['driver'],
                        'groups': going_dog['groups'],
                        'distance': distance,
                        'type': 'parking/field',
                        'match_type': 'exact'
                    })
                elif not is_exact_match and distance <= self.ADJACENT_MATCH_MAX_DISTANCE:
                    adjacent_match_parking_distances.append({
                        'dog_id': going_dog['dog_id'],
                        'dog_name': going_dog['dog_name'],
                        'driver': going_dog['driver'],
                        'groups': going_dog['groups'],
                        'distance': distance,
                        'type': 'parking/field',
                        'match_type': 'adjacent'
                    })
            
            # Sort each list by distance (closest first)
            exact_match_distances.sort(key=lambda x: x['distance'])
            adjacent_match_distances.sort(key=lambda x: x['distance'])
            exact_match_parking_distances.sort(key=lambda x: x['distance'])
            adjacent_match_parking_distances.sort(key=lambda x: x['distance'])
            
            # Combine in priority order: 
            # 1. Exact matches in regular dogs
            # 2. Adjacent matches in regular dogs  
            # 3. Exact matches in parking/field dogs
            # 4. Adjacent matches in parking/field dogs
            all_distances = (exact_match_distances + adjacent_match_distances + 
                           exact_match_parking_distances + adjacent_match_parking_distances)
            
            print(f"       📏 Found matches within limits:")
            print(f"         🎯 Regular: {len(exact_match_distances)} exact (≤{self.EXACT_MATCH_MAX_DISTANCE}mi) + {len(adjacent_match_distances)} adjacent (≤{self.ADJACENT_MATCH_MAX_DISTANCE}mi)")
            print(f"         ⚠️ Parking/Field: {len(exact_match_parking_distances)} exact + {len(adjacent_match_parking_distances)} adjacent")
            
            if exact_match_distances:
                print(f"       🎯 Trying exact matches in regular dogs first...")
            elif adjacent_match_distances:
                print(f"       🎯 No exact regular matches - trying adjacent regular matches...")
            elif exact_match_parking_distances:
                print(f"       ⚠️ No regular matches - trying exact parking/field matches...")
            elif adjacent_match_parking_distances:
                print(f"       ⚠️ No regular matches - trying adjacent parking/field matches as last resort...")
            else:
                print(f"       ❌ No valid matches found within distance limits")
            
            # Try to assign to drivers in priority order
            assigned = False
            
            for close_dog in all_distances:
                driver = close_dog['driver']
                distance = close_dog['distance']
                dog_type = close_dog['type']
                match_type = close_dog['match_type']
                
                type_indicator = "🎯" if dog_type == 'regular' else "⚠️"
                match_indicator = "🎯" if match_type == 'exact' else "📍"
                
                print(f"\n       🚗 Checking driver {driver} {type_indicator}{match_indicator} (via {close_dog['dog_name']} at {distance:.1f}mi - {dog_type} {match_type} match)")
                
                # Check if this driver has capacity in ALL needed groups
                can_handle_all_groups = True
                capacity_status = []
                
                for group in callout_dog['needed_groups']:
                    group_key = f'group{group}'
                    current_load = current_driver_loads.get(driver, {}).get(group_key, 0)
                    capacity = self.driver_capacities.get(driver, {}).get(group_key, 0)
                    needed = callout_dog['num_dogs']
                    
                    if current_load + needed <= capacity:
                        capacity_status.append(f"Group {group}: {current_load + needed}/{capacity} ✅")
                    else:
                        capacity_status.append(f"Group {group}: {current_load + needed}/{capacity} ❌")
                        can_handle_all_groups = False
                
                print(f"          📊 Capacity check: {', '.join(capacity_status)}")
                
                if not can_handle_all_groups:
                    print(f"          ❌ Insufficient capacity")
                    continue
                
                # Distance check already passed during filtering, so we know it's within limits
                max_distance = self.EXACT_MATCH_MAX_DISTANCE if match_type == 'exact' else self.ADJACENT_MATCH_MAX_DISTANCE
                print(f"          📏 {match_type.upper()} MATCH: {distance:.1f}mi ≤ {max_distance:.1f}mi ✅")
                
                # SUCCESS! Create new assignment with same assignment string, new driver
                new_assignment = f"{driver}:{callout_dog['full_assignment_string']}"
                
                if dog_type == 'regular' and match_type == 'exact':
                    success_indicator = "🎉"
                    assignment_desc = "PERFECT MATCH! (regular exact)"
                elif dog_type == 'regular' and match_type == 'adjacent':
                    success_indicator = "✅"
                    assignment_desc = "GOOD MATCH (regular adjacent)"
                elif dog_type == 'parking/field' and match_type == 'exact':
                    success_indicator = "🔄"
                    assignment_desc = "BACKUP MATCH (parking/field exact)"
                else:
                    success_indicator = "⚠️"
                    assignment_desc = "LAST RESORT (parking/field adjacent)"
                
                print(f"          {success_indicator} {assignment_desc}")
                print(f"          📝 New assignment: {new_assignment}")
                print(f"          📋 (Preserved exact string: '{callout_dog['full_assignment_string']}')")
                
                assignments.append({
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'new_assignment': new_assignment,
                    'driver': driver,
                    'distance': distance,
                    'closest_dog': close_dog['dog_name'],
                    'assignment_type': dog_type,
                    'match_type': match_type,
                    'reason': f"capacity_available_via_{close_dog['dog_name']}_{dog_type}_{match_type}"
                })
                
                # UPDATE STATE: Add this dog's load to the driver
                if driver not in current_driver_loads:
                    current_driver_loads[driver] = {'group1': 0, 'group2': 0, 'group3': 0}
                
                for group in callout_dog['needed_groups']:
                    group_key = f'group{group}'
                    current_driver_loads[driver][group_key] += callout_dog['num_dogs']
                
                print(f"          📊 Updated {driver} load: {current_driver_loads[driver]}")
                
                # Add to regular dogs list for future iterations (becomes a regular assignment)
                regular_dogs_going_today.append({
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'driver': driver,
                    'groups': callout_dog['needed_groups'],
                    'num_dogs': callout_dog['num_dogs']
                })
                
                assigned = True
                break
            
            if not assigned:
                print(f"       ❌ NO VALID ASSIGNMENT for {callout_dog['dog_name']}")
                print(f"          No drivers found with capacity in groups {callout_dog['needed_groups']} within 3.0 miles")
        
        return assignments

    def _calculate_permutation_cost(self, assignments, ordered_dogs):
        """Calculate total cost for a permutation (total miles + penalty for unassigned)"""
        # Base cost: actual distances
        total_distance_cost = sum(assignment['distance'] for assignment in assignments)
        
        # Penalty cost: unassigned dogs (5 miles each)
        assigned_dog_ids = {assignment['dog_id'] for assignment in assignments}
        unassigned_count = 0
        
        for dog in ordered_dogs:
            if dog['dog_id'] not in assigned_dog_ids:
                unassigned_count += 1
        
        penalty_cost = unassigned_count * 5.0
        
        total_cost = total_distance_cost + penalty_cost
        
        return total_cost

    def _order_by_difficulty(self, dogs_to_reassign):
        """Order dogs by placement difficulty - using preserved assignment strings"""
        print("   🔍 Analyzing placement difficulty based on driver capacity...")
        
        def calculate_difficulty_score(dog):
            """Calculate difficulty score considering the preserved assignment requirements"""
            dog_id = dog['dog_id']
            needed_groups = dog['needed_groups']  # For capacity checking
            num_dogs = dog['num_dogs']
            assignment_string = dog['full_assignment_string']  # What they'll actually get
            
            # Get current driver loads
            current_driver_loads = {}
            for driver in self.driver_capacities.keys():
                current_driver_loads[driver] = self.calculate_driver_load(driver)
            
            # Count drivers with capacity in ALL needed groups
            drivers_with_capacity = 0
            viable_close_options = 0
            
            for assignment in self.dog_assignments:
                if assignment.get('combined') and ':' in assignment['combined']:
                    other_dog_id = assignment['dog_id']
                    distance = self.get_distance(dog_id, other_dog_id)
                    
                    if distance <= 3.0:  # Use the same 3 mile limit
                        # Check if this driver has capacity in all needed groups
                        combined = assignment['combined']
                        driver = combined.split(':', 1)[0].strip()
                        
                        can_handle = True
                        current_load = current_driver_loads.get(driver, {})
                        driver_capacity = self.driver_capacities.get(driver, {})
                        
                        for group in needed_groups:
                            group_key = f'group{group}'
                            if current_load.get(group_key, 0) + num_dogs > driver_capacity.get(group_key, 0):
                                can_handle = False
                                break
                        
                        if can_handle:
                            viable_close_options += 1
            
            # Calculate total drivers with capacity (regardless of distance)
            for driver, capacity in self.driver_capacities.items():
                can_handle_all = True
                current_load = current_driver_loads.get(driver, {})
                
                for group in needed_groups:
                    group_key = f'group{group}'
                    if current_load.get(group_key, 0) + num_dogs > capacity.get(group_key, 0):
                        can_handle_all = False
                        break
                
                if can_handle_all:
                    drivers_with_capacity += 1
            
            # Calculate difficulty
            if viable_close_options == 0:
                difficulty = 100  # No close viable options
            elif viable_close_options <= 2:
                difficulty = 60   # Very few options
            elif viable_close_options <= 5:
                difficulty = 30   # Some options
            else:
                difficulty = 10   # Many options
            
            # Add penalties
            if num_dogs > 1:
                difficulty += (num_dogs - 1) * 10  # Multiple dogs penalty
            
            if len(needed_groups) > 1:
                difficulty += len(needed_groups) * 5  # Multi-group penalty
            
            return {
                'dog': dog,
                'total_difficulty': difficulty,
                'viable_close_options': viable_close_options,
                'drivers_with_capacity': drivers_with_capacity
            }
        
        # Calculate difficulty for all dogs
        difficulty_analysis = []
        
        for dog in dogs_to_reassign:
            analysis = calculate_difficulty_score(dog)
            difficulty_analysis.append(analysis)
        
        # Sort by difficulty (hardest first)
        difficulty_analysis.sort(key=lambda x: x['total_difficulty'], reverse=True)
        
        # Show difficulty analysis
        print("   📋 Difficulty ranking (hardest to assign first):")
        for i, analysis in enumerate(difficulty_analysis[:5]):  # Show top 5
            dog = analysis['dog']
            print(f"      {i+1}. {dog['dog_name']} - Difficulty: {analysis['total_difficulty']}")
            print(f"         Assignment: '{dog['full_assignment_string']}'")
            print(f"         Capacity needed: Groups {dog['needed_groups']}")
            print(f"         🎯 {analysis['viable_close_options']} close viable options")
            print(f"         🏢 {analysis['drivers_with_capacity']} total drivers with capacity")
        
        # Return the ordered list of dogs
        return [analysis['dog'] for analysis in difficulty_analysis]

    def _order_by_centrality(self, dogs_to_reassign):
        """Order dogs by distance to centroid of all going dogs (central first)"""
        # Calculate centroid position (average of all going dogs)
        going_dog_positions = []
        for assignment in self.dog_assignments:
            if assignment.get('combined') and ':' in assignment['combined']:
                going_dog_positions.append(assignment['dog_id'])
        
        def centrality_score(dog):
            # Average distance to all going dogs
            total_distance = 0
            valid_distances = 0
            for going_dog_id in going_dog_positions[:20]:  # Sample to avoid slowdown
                distance = self.get_distance(dog['dog_id'], going_dog_id)
                if distance < self.EXCLUSION_DISTANCE:
                    total_distance += distance
                    valid_distances += 1
            
            return total_distance / max(valid_distances, 1)
        
        return sorted(dogs_to_reassign, key=centrality_score)

    def _order_by_constraint_count(self, dogs_to_reassign):
        """Order dogs by number of valid assignment options (fewest options first)"""
        print("   🔍 Counting valid assignment options for each dog...")
        
        def count_valid_options(dog):
            """Count how many drivers can actually take this dog"""
            dog_id = dog['dog_id']
            needed_groups = dog['needed_groups']
            num_dogs = dog['num_dogs']
            
            valid_options = 0
            current_driver_loads = {}
            
            # Calculate current loads for all drivers
            for driver in self.driver_capacities.keys():
                current_driver_loads[driver] = self.calculate_driver_load(driver)
            
            # Check each going dog to see if their driver can take our callout dog
            for assignment in self.dog_assignments:
                if assignment.get('combined') and ':' in assignment['combined']:
                    other_dog_id = assignment['dog_id']
                    distance = self.get_distance(dog_id, other_dog_id)
                    
                    # Must be within 3 miles
                    if distance > 3.0:
                        continue
                    
                    # Get the driver
                    combined = assignment['combined']
                    driver = combined.split(':', 1)[0].strip()
                    
                    # Check capacity constraints
                    current_load = current_driver_loads.get(driver, {})
                    driver_capacity = self.driver_capacities.get(driver, {})
                    
                    can_handle = True
                    for group in needed_groups:
                        group_key = f'group{group}'
                        if current_load.get(group_key, 0) + num_dogs > driver_capacity.get(group_key, 0):
                            can_handle = False
                            break
                    
                    if can_handle:
                        valid_options += 1
            
            return valid_options
        
        # Calculate valid options for each dog
        constraint_analysis = []
        
        for dog in dogs_to_reassign:
            options = count_valid_options(dog)
            constraint_analysis.append({
                'dog': dog,
                'valid_options': options
            })
        
        # Sort by fewest options first (most constrained first)
        constraint_analysis.sort(key=lambda x: x['valid_options'])
        
        # Show constraint analysis
        print("   📋 Constraint ranking (most constrained first):")
        for i, analysis in enumerate(constraint_analysis[:5]):  # Show top 5
            dog = analysis['dog']
            options = analysis['valid_options']
            print(f"      {i+1}. {dog['dog_name']} (Groups: {dog['needed_groups']}) - {options} valid options")
            
            if options == 0:
                print(f"         ❌ No valid drivers found!")
            elif options <= 2:
                print(f"         ⚠️ Very constrained")
            elif options <= 5:
                print(f"         ⚡ Moderately constrained")
        
        return [analysis['dog'] for analysis in constraint_analysis]

    def _order_by_group_priority(self, dogs_to_reassign):
        """Order dogs by group priority (Group 1 first, then multi-group, then others)"""
        def group_priority(dog):
            groups = dog['needed_groups']
            if 1 in groups and len(groups) == 1:
                return 1  # Group 1 only (highest priority)
            elif len(groups) > 1:
                return 2  # Multi-group (second priority)
            elif 2 in groups:
                return 3  # Group 2 only
            else:
                return 4  # Group 3 only
        
        return sorted(dogs_to_reassign, key=group_priority)

    def _order_randomly(self, dogs_to_reassign):
        """Random ordering for baseline comparison"""
        random_order = dogs_to_reassign.copy()
        random.shuffle(random_order)
        return random_order

    def reassign_dogs_optimal_permutation(self):
        """Try all permutations for small numbers of dogs (≤6)"""
        print("\n🔄 Starting OPTIMAL PERMUTATION ASSIGNMENT...")
        print("🎯 Trying all possible orderings to find globally optimal solution")
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        num_dogs = len(dogs_to_reassign)
        num_permutations = 1
        for i in range(1, num_dogs + 1):
            num_permutations *= i
        
        print(f"📊 Found {num_dogs} callout dogs")
        print(f"🔢 Total permutations to try: {num_permutations}")
        
        # Safety check: Don't try more than 720 permutations (6! = reasonable limit)
        MAX_PERMUTATIONS = 720
        if num_permutations > MAX_PERMUTATIONS:
            print(f"⚠️ Too many permutations ({num_permutations} > {MAX_PERMUTATIONS})")
            print(f"🔄 Falling back to smart optimization...")
            return self.reassign_dogs_smart_optimization()
        
        print(f"⏱️ Estimated processing time: {num_permutations * 2}s (2s per permutation)")
        
        best_assignments = []
        best_total_cost = float('inf')
        best_order = None
        best_details = {}
        
        all_permutations = list(itertools.permutations(dogs_to_reassign))
        
        print(f"\n🧪 TESTING ALL {num_permutations} PERMUTATIONS:")
        
        start_time = time.time()
        
        for perm_idx, dog_order in enumerate(all_permutations):
            print(f"\n📋 Permutation {perm_idx + 1}/{num_permutations}:")
            order_names = [dog['dog_name'] for dog in dog_order]
            print(f"   Order: {' → '.join(order_names)}")
            
            # Run greedy assignment with this specific order
            assignments = self._run_greedy_assignment_with_order(list(dog_order))
            
            # Calculate total cost for this permutation
            total_cost = self._calculate_permutation_cost(assignments, list(dog_order))
            
            print(f"   📊 Result: {len(assignments)} assigned, total cost: {total_cost:.1f} miles")
            
            # Track if this is the best so far
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_assignments = assignments
                best_order = order_names.copy()
                best_details = {
                    'permutation': perm_idx + 1,
                    'assigned_count': len(assignments),
                    'unassigned_count': len(dog_order) - len(assignments),
                    'total_miles': total_cost
                }
                print(f"   🏆 NEW BEST! Cost: {total_cost:.1f}")
            
            # Progress update every 20 permutations
            if (perm_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                remaining = (num_permutations - perm_idx - 1) * (elapsed / (perm_idx + 1))
                print(f"   ⏱️ Progress: {perm_idx + 1}/{num_permutations}, ~{remaining:.0f}s remaining")
        
        elapsed_time = time.time() - start_time
        
        # Show results
        print(f"\n🏆 OPTIMAL PERMUTATION RESULTS:")
        print(f"   ⏱️ Total processing time: {elapsed_time:.1f} seconds")
        print(f"   🎯 Best permutation: #{best_details['permutation']}")
        print(f"   📋 Best order: {' → '.join(best_order)}")
        print(f"   📊 Dogs assigned: {best_details['assigned_count']}/{num_dogs}")
        print(f"   📏 Total cost: {best_details['total_miles']:.1f} miles")
        
        if best_details['unassigned_count'] > 0:
            print(f"   ⚠️ Unassigned dogs: {best_details['unassigned_count']} (counted as 5 miles each)")
        
        # Show the best assignments
        if best_assignments:
            print(f"\n🎉 OPTIMAL ASSIGNMENTS:")
            for assignment in best_assignments:
                assignment_type = assignment.get('assignment_type', 'regular')
                type_indicator = "🎯" if assignment_type == 'regular' else "⚠️"
                assignment_description = "" if assignment_type == 'regular' else " (via parking/field)"
                
                print(f"      {type_indicator} {assignment['dog_name']} → {assignment['new_assignment']}{assignment_description}")
                print(f"         Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
        
        return best_assignments

    def reassign_dogs_smart_optimization(self):
        """Smart heuristic + local optimization for large numbers of callout dogs"""
        print("\n🔄 Starting SMART OPTIMIZATION ASSIGNMENT...")
        print("🎯 Using intelligent heuristics (exact ≤3mi, adjacent ≤0.6mi, regular dogs priority)")
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        num_dogs = len(dogs_to_reassign)
        print(f"📊 Found {num_dogs} callout dogs")
        
        # Show what each callout dog needs
        print(f"\n📋 CALLOUT REQUIREMENTS:")
        for dog in dogs_to_reassign:
            original_assignment = dog['original_callout']
            needed_groups = dog['needed_groups']
            print(f"   🐕 {dog['dog_name']} ({dog['dog_id']}): was {original_assignment} → needs driver with capacity in groups {needed_groups}")
        
        # Show current driver capacity status
        print(f"\n📊 CURRENT DRIVER CAPACITY:")
        for driver, capacity in self.driver_capacities.items():
            current_load = self.calculate_driver_load(driver)
            available = {}
            for group in [1, 2, 3]:
                group_key = f'group{group}'
                available[group_key] = capacity.get(group_key, 0) - current_load.get(group_key, 0)
            
            print(f"   🚗 {driver}: Available capacity G1:{available['group1']}, G2:{available['group2']}, G3:{available['group3']}")
        
        if num_dogs <= 6:
            print(f"🔢 Small number of dogs - using exhaustive permutation optimization")
            return self.reassign_dogs_optimal_permutation()
        
        print(f"🧠 Large number of dogs - using smart heuristic approach")
        
        start_time = time.time()
        
        # Step 1: Try multiple intelligent ordering strategies
        strategies = [
            ("Hardest First", self._order_by_difficulty),
            ("Closest to Center", self._order_by_centrality), 
            ("Fewest Options", self._order_by_constraint_count),
            ("Group Priority", self._order_by_group_priority),
            ("Random Baseline", self._order_randomly)
        ]
        
        best_assignments = []
        best_cost = float('inf')
        best_strategy = ""
        strategy_results = []
        
        print(f"\n🧪 TESTING {len(strategies)} ORDERING STRATEGIES:")
        
        for strategy_name, ordering_func in strategies:
            print(f"\n📋 Strategy: {strategy_name}")
            
            # Get initial ordering from strategy
            ordered_dogs = ordering_func(dogs_to_reassign)
            order_names = [dog['dog_name'] for dog in ordered_dogs]
            print(f"   Order: {' → '.join(order_names[:5])}{'...' if len(order_names) > 5 else ''}")
            
            # Run assignment with this ordering
            assignments = self._run_greedy_assignment_with_order(ordered_dogs)
            cost = self._calculate_permutation_cost(assignments, ordered_dogs)
            
            strategy_results.append({
                'name': strategy_name,
                'cost': cost,
                'assignments': assignments,
                'assigned_count': len(assignments)
            })
            
            print(f"   📊 Result: {len(assignments)}/{num_dogs} assigned, cost: {cost:.1f} miles")
            
            if cost < best_cost:
                best_cost = cost
                best_assignments = assignments
                best_strategy = strategy_name
                print(f"   🏆 NEW BEST!")
        
        elapsed_time = time.time() - start_time
        
        # Show results
        print(f"\n🏆 SMART OPTIMIZATION RESULTS:")
        print(f"   ⏱️ Total processing time: {elapsed_time:.1f} seconds")
        print(f"   🎯 Best strategy: {best_strategy}")
        print(f"   📊 Dogs assigned: {len(best_assignments)}/{num_dogs}")
        print(f"   📏 Total cost: {best_cost:.1f} miles")
        
        # Show strategy comparison
        print(f"\n📊 STRATEGY COMPARISON:")
        strategy_results.sort(key=lambda x: x['cost'])
        for i, result in enumerate(strategy_results):
            status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"   {status} {result['name']}: {result['assigned_count']}/{num_dogs} assigned, {result['cost']:.1f} miles")
        
        # Show the best assignments
        if best_assignments:
            print(f"\n🎉 OPTIMAL ASSIGNMENTS ({best_strategy}):")
            
            # Categorize assignments by type and match
            perfect_matches = [a for a in best_assignments if a.get('assignment_type') == 'regular' and a.get('match_type') == 'exact']
            good_matches = [a for a in best_assignments if a.get('assignment_type') == 'regular' and a.get('match_type') == 'adjacent']
            backup_exact = [a for a in best_assignments if a.get('assignment_type') == 'parking/field' and a.get('match_type') == 'exact']
            backup_adjacent = [a for a in best_assignments if a.get('assignment_type') == 'parking/field' and a.get('match_type') == 'adjacent']
            
            if perfect_matches:
                print(f"      🎉 PERFECT (regular exact matches):")
                for assignment in perfect_matches:
                    print(f"         ✅ {assignment['dog_name']} → {assignment['new_assignment']}")
                    print(f"            Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
            
            if good_matches:
                print(f"      ✅ GOOD (regular adjacent matches ≤0.6mi):")
                for assignment in good_matches:
                    print(f"         📍 {assignment['dog_name']} → {assignment['new_assignment']}")
                    print(f"            Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
            
            if backup_exact:
                print(f"      🔄 BACKUP (parking/field exact matches):")
                for assignment in backup_exact:
                    print(f"         ⚠️ {assignment['dog_name']} → {assignment['new_assignment']}")
                    print(f"            Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
            
            if backup_adjacent:
                print(f"      ⚠️ LAST RESORT (parking/field adjacent ≤0.6mi):")
                for assignment in backup_adjacent:
                    print(f"         🆘 {assignment['dog_name']} → {assignment['new_assignment']}")
                    print(f"            Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
        
        return best_assignments

    def write_results_to_sheets(self, reassignments):
        """Write reassignment results back to Google Sheets"""
        try:
            print(f"\n📝 Writing {len(reassignments)} results to Google Sheets...")
            
            if not hasattr(self, 'sheets_client') or not self.sheets_client:
                print("❌ Google Sheets client not initialized")
                return False
            
            # Extract sheet ID from your existing MAP_SHEET_URL
            sheet_id = "1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0"
            
            # Open the spreadsheet
            spreadsheet = self.sheets_client.open_by_key(sheet_id)
            
            # Get the worksheet (try the specific gid first, then common names)
            worksheet = None
            try:
                # Get all worksheets and find the one with gid 267803750
                for ws in spreadsheet.worksheets():
                    if str(ws.id) == "267803750":
                        worksheet = ws
                        break
            except:
                pass
            
            # Fallback to common sheet names if gid lookup failed
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
            
            # Get all data to understand the structure
            all_data = worksheet.get_all_values()
            if not all_data:
                print("❌ No data found in worksheet")
                return False
            
            header_row = all_data[0]
            print(f"📋 Sheet has {len(all_data)} rows and columns: {header_row[:10]}...")
            
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
            
            # Always use Column H (Combined column) - index 7
            target_col = 7
            print(f"📍 Writing to Column H (Combined) at index {target_col}")
            
            # Prepare batch updates
            updates = []
            updates_count = 0
            
            print(f"\n🔍 Processing {len(reassignments)} reassignments...")
            
            for assignment in reassignments:
                dog_id = str(assignment['dog_id']).strip()
                new_assignment = assignment['new_assignment']
                
                # Find the row for this dog ID
                found = False
                for row_idx in range(1, len(all_data)):  # Skip header row
                    if dog_id_col < len(all_data[row_idx]):
                        current_dog_id = str(all_data[row_idx][dog_id_col]).strip()
                        
                        if current_dog_id == dog_id:
                            # Convert to A1 notation (row is 1-indexed, col is 1-indexed)
                            cell_address = gspread.utils.rowcol_to_a1(row_idx + 1, target_col + 1)
                            
                            updates.append({
                                'range': cell_address,
                                'values': [[new_assignment]]
                            })
                            
                            updates_count += 1
                            print(f"  ✅ {dog_id} → {new_assignment} (cell {cell_address})")
                            found = True
                            break
                
                if not found:
                    print(f"  ⚠️ Could not find row for dog ID: {dog_id}")
            
            if not updates:
                print("❌ No valid updates to make")
                return False
            
            # Execute batch update
            print(f"\n📤 Sending {len(updates)} updates to Google Sheets...")
            
            # Use batch_update for efficiency
            worksheet.batch_update(updates)
            
            print(f"✅ Successfully updated {updates_count} assignments in Google Sheets!")
            
            # Optional: Send Slack notification if webhook is configured
            slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    slack_message = {
                        "text": f"🐕 Dog Reassignment Complete: Updated {updates_count} assignments using 3-mile limit optimization (regular dogs priority)"
                    }
                    response = requests.post(slack_webhook, json=slack_message, timeout=10)
                    if response.status_code == 200:
                        print("📱 Slack notification sent")
                    else:
                        print(f"⚠️ Slack notification failed: {response.status_code}")
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
    print("🚀 Production Dog Reassignment System - SMART OPTIMIZATION")
    print("📏 Distance Limits: Exact matches ≤3mi, Adjacent matches ≤0.6mi, Regular dogs priority")
    print("=" * 95)
    
    # Initialize system
    system = DogReassignmentSystem()
    
    # Setup Google Sheets client for WRITING (reading still uses CSV URLs)
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
    
    # Run the smart optimization assignment
    print("\n🔄 Processing callout assignments...")
    
    reassignments = system.reassign_dogs_smart_optimization()
    
    # Ensure reassignments is always a list
    if reassignments is None:
        reassignments = []
    
    # Write results
    if reassignments:
        write_success = system.write_results_to_sheets(reassignments)
        if write_success:
            print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments using 3-mile limit with regular dogs priority")
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")

if __name__ == "__main__":
    main()
