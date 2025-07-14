# production_reassignment.py
# FIXED VERSION: Complete dog reassignment system with bug fixes and relaxed constraints

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
        
        # DISTANCE LIMITS - RELAXED BY 10% from your original settings
        self.MAX_DISTANCE = 3.3  # Was 3.0, now 3.3 (+10%)
        self.EXACT_MATCH_MAX_DISTANCE = 3.3  # Was 3.0, now 3.3 (+10%)
        self.ADJACENT_MATCH_MAX_DISTANCE = 0.66  # Was 0.6, now 0.66 (+10%)
        self.PARKING_EXACT_MAX_DISTANCE = 0.44  # Was 0.4, now 0.44 (+10%)
        self.PARKING_ADJACENT_MAX_DISTANCE = 0.44  # Was 0.4, now 0.44 (+10%)
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
        """Find dogs that need reassignment (callouts) - excluding non-dog entries"""
        dogs_to_reassign = []
        
        if not self.dog_assignments:
            return dogs_to_reassign
        
        for assignment in self.dog_assignments:
            # Check for callout: Combined column blank AND Callout column has content
            if (not assignment['combined'] or assignment['combined'].strip() == "") and \
               (assignment['callout'] and assignment['callout'].strip() != ""):
                
                # FILTER OUT NON-DOGS: Skip Parking, Field, and other administrative entries
                dog_name = str(assignment['dog_name']).lower().strip()
                if any(keyword in dog_name for keyword in ['parking', 'field', 'admin', 'office']):
                    print(f"   ⏭️ Skipping non-dog entry: {assignment['dog_name']} ({assignment['dog_id']})")
                    continue
                
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
        
        print(f"🚨 Found {len(dogs_to_reassign)} REAL dogs that need drivers assigned:")
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
            
            # PRIORITY 1: Try exact matches in regular dogs first
            exact_match_distances = []
            adjacent_match_distances = []
            
            for going_dog in regular_dogs_going_today:
                distance = self.get_distance(callout_dog['dog_id'], going_dog['dog_id'])
                
                # Check if this would be an exact or adjacent match based on GROUP NUMBERS
                needed_groups = set(callout_dog['needed_groups'])
                driver_groups = set(going_dog['groups'])
                
                # Exact match: driver already has the same groups
                is_exact_match = needed_groups == driver_groups
                
                # Adjacent match: driver has groups that are numerically adjacent
                is_adjacent_match = False
                if not is_exact_match:
                    for needed_group in needed_groups:
                        for driver_group in driver_groups:
                            # Check if groups are numerically adjacent
                            if abs(needed_group - driver_group) == 1:
                                is_adjacent_match = True
                                break
                        if is_adjacent_match:
                            break
                
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
                elif is_adjacent_match and distance <= self.ADJACENT_MATCH_MAX_DISTANCE:
                    adjacent_match_distances.append({
                        'dog_id': going_dog['dog_id'],
                        'dog_name': going_dog['dog_name'],
                        'driver': going_dog['driver'],
                        'groups': going_dog['groups'],
                        'distance': distance,
                        'type': 'regular',
                        'match_type': 'adjacent'
                    })
            
            # PRIORITY 2: Try exact matches in parking/field dogs
            exact_match_parking_distances = []
            adjacent_match_parking_distances = []
            
            for going_dog in parking_field_dogs_going_today:
                distance = self.get_distance(callout_dog['dog_id'], going_dog['dog_id'])
                
                needed_groups = set(callout_dog['needed_groups'])
                driver_groups = set(going_dog['groups'])
                
                # Exact match: driver already has the same groups
                is_exact_match = needed_groups == driver_groups
                
                # Adjacent match: driver has groups that are numerically adjacent
                is_adjacent_match = False
                if not is_exact_match:
                    for needed_group in needed_groups:
                        for driver_group in driver_groups:
                            if abs(needed_group - driver_group) == 1:
                                is_adjacent_match = True
                                break
                        if is_adjacent_match:
                            break
                
                if is_exact_match and distance <= self.PARKING_EXACT_MAX_DISTANCE:
                    exact_match_parking_distances.append({
                        'dog_id': going_dog['dog_id'],
                        'dog_name': going_dog['dog_name'],
                        'driver': going_dog['driver'],
                        'groups': going_dog['groups'],
                        'distance': distance,
                        'type': 'parking/field',
                        'match_type': 'exact'
                    })
                elif is_adjacent_match and distance <= self.PARKING_ADJACENT_MAX_DISTANCE:
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
            
            # Combine in priority order
            all_distances = (exact_match_distances + adjacent_match_distances + 
                           exact_match_parking_distances + adjacent_match_parking_distances)
            
            print(f"       📏 GROUP-BASED matching for groups {callout_dog['needed_groups']}:")
            print(f"         🎯 Regular: {len(exact_match_distances)} exact group matches (≤{self.EXACT_MATCH_MAX_DISTANCE}mi) + {len(adjacent_match_distances)} adjacent group matches (≤{self.ADJACENT_MATCH_MAX_DISTANCE}mi)")
            print(f"         ⚠️ Parking/Field: {len(exact_match_parking_distances)} exact (≤{self.PARKING_EXACT_MAX_DISTANCE}mi) + {len(adjacent_match_parking_distances)} adjacent (≤{self.PARKING_ADJACENT_MAX_DISTANCE}mi)")
            
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
                
                # SUCCESS! Create new assignment with same assignment string, new driver
                new_assignment = f"{driver}:{callout_dog['full_assignment_string']}"
                
                if dog_type == 'regular' and match_type == 'exact':
                    success_indicator = "🎉"
                    assignment_desc = "PERFECT MATCH! (same groups)"
                elif dog_type == 'regular' and match_type == 'adjacent':
                    success_indicator = "✅"
                    assignment_desc = "GOOD MATCH (adjacent groups)"
                elif dog_type == 'parking/field' and match_type == 'exact':
                    success_indicator = "🔄"
                    assignment_desc = "BACKUP MATCH (parking/field same groups)"
                else:
                    success_indicator = "⚠️"
                    assignment_desc = "LAST RESORT (parking/field adjacent groups)"
                
                print(f"          {success_indicator} {assignment_desc}")
                print(f"          📝 New assignment: {new_assignment}")
                print(f"          📋 (Preserved exact string: '{callout_dog['full_assignment_string']}')")
                
                assignments.append({
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'new_assignment': new_assignment,  # 🔧 CRITICAL: This must be "Driver:Groups", not just dog_id!
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
                
                # Add to regular dogs list for future iterations
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
                print(f"          No drivers found with capacity in groups {callout_dog['needed_groups']} within distance limits")
        
        return assignments

    def _order_by_difficulty(self, dogs_to_reassign):
        """Order dogs by placement difficulty"""
        print("   🔍 Analyzing placement difficulty based on driver capacity...")
        
        def calculate_difficulty_score(dog):
            """Calculate difficulty score considering the preserved assignment requirements"""
            dog_id = dog['dog_id']
            needed_groups = dog['needed_groups']
            num_dogs = dog['num_dogs']
            
            # Get current driver loads
            current_driver_loads = {}
            for driver in self.driver_capacities.keys():
                current_driver_loads[driver] = self.calculate_driver_load(driver)
            
            # Count drivers with capacity in ALL needed groups
            viable_close_options = 0
            
            for assignment in self.dog_assignments:
                if assignment.get('combined') and ':' in assignment['combined']:
                    other_dog_id = assignment['dog_id']
                    distance = self.get_distance(dog_id, other_dog_id)
                    
                    if distance <= 3.3:  # Use the relaxed 3.3 mile limit
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
                'viable_close_options': viable_close_options
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
        
        # Return the ordered list of dogs
        return [analysis['dog'] for analysis in difficulty_analysis]

    def reassign_dogs_smart_optimization(self):
        """Smart heuristic assignment for callout dogs"""
        print("\n🔄 Starting SMART OPTIMIZATION ASSIGNMENT...")
        print("🎯 Using RELAXED constraints: same groups ≤3.3mi, adjacent ≤0.66mi, parking ≤0.44mi")
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        num_dogs = len(dogs_to_reassign)
        print(f"📊 Found {num_dogs} callout dogs")
        
        # Show current driver capacity status
        print(f"\n📊 CURRENT DRIVER CAPACITY:")
        for driver, capacity in self.driver_capacities.items():
            current_load = self.calculate_driver_load(driver)
            available = {}
            for group in [1, 2, 3]:
                group_key = f'group{group}'
                available[group_key] = capacity.get(group_key, 0) - current_load.get(group_key, 0)
            
            print(f"   🚗 {driver}: Available capacity G1:{available['group1']}, G2:{available['group2']}, G3:{available['group3']}")
        
        start_time = time.time()
        
        # Use difficulty-based ordering for optimal results
        print(f"\n🧪 USING DIFFICULTY-BASED ORDERING:")
        
        ordered_dogs = self._order_by_difficulty(dogs_to_reassign)
        order_names = [dog['dog_name'] for dog in ordered_dogs]
        print(f"   Order: {' → '.join(order_names[:5])}{'...' if len(order_names) > 5 else ''}")
        
        # Run assignment with this ordering
        assignments = self._run_greedy_assignment_with_order(ordered_dogs)
        
        elapsed_time = time.time() - start_time
        
        # Show results
        print(f"\n🏆 SMART OPTIMIZATION RESULTS:")
        print(f"   ⏱️ Total processing time: {elapsed_time:.1f} seconds")
        print(f"   📊 Dogs assigned: {len(assignments)}/{num_dogs}")
        
        # Show the assignments with safety validation
        if assignments:
            print(f"\n🎉 ASSIGNMENTS (with safety validation):")
            
            # 🔒 SAFETY CHECK: Validate all assignments before returning
            validated_assignments = []
            for assignment in assignments:
                dog_id = assignment.get('dog_id', '')
                new_assignment = assignment.get('new_assignment', '')
                
                # Safety validations
                if not new_assignment:
                    print(f"   ❌ INVALID: Empty new_assignment for {dog_id}")
                    continue
                
                if new_assignment == dog_id:
                    print(f"   ❌ INVALID: new_assignment equals dog_id for {dog_id}")
                    continue
                
                if new_assignment.endswith('x') and new_assignment[:-1].isdigit():
                    print(f"   ❌ INVALID: new_assignment '{new_assignment}' looks like dog_id for {dog_id}")
                    continue
                
                if ':' not in new_assignment:
                    print(f"   ❌ INVALID: new_assignment '{new_assignment}' missing driver:group format for {dog_id}")
                    continue
                
                # Valid assignment
                validated_assignments.append(assignment)
                assignment_type = assignment.get('assignment_type', 'regular')
                type_indicator = "🎯" if assignment_type == 'regular' else "⚠️"
                
                print(f"      {type_indicator} {assignment['dog_name']} → {assignment['new_assignment']}")
                print(f"         Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
            
            print(f"\n🔒 SAFETY VALIDATION: {len(validated_assignments)}/{len(assignments)} assignments passed validation")
            
            return validated_assignments
        
        return []

    def write_results_to_sheets(self, reassignments):
        """FIXED VERSION: Write reassignment results back to Google Sheets with extensive safety checks"""
        try:
            print(f"\n📝 Writing {len(reassignments)} results to Google Sheets...")
            
            if not hasattr(self, 'sheets_client') or not self.sheets_client:
                print("❌ Google Sheets client not initialized")
                return False
            
            # 🔍 SAFETY: Pre-validation of reassignments data
            print(f"\n🔒 PRE-VALIDATION: Checking reassignment data structure...")
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
            
            if len(reassignments) > 3:
                print(f"   ... and {len(reassignments) - 3} more (all validated)")
            
            print(f"✅ Pre-validation passed!")
            
            # Extract sheet ID
            sheet_id = "1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0"
            
            # Open the spreadsheet
            spreadsheet = self.sheets_client.open_by_key(sheet_id)
            
            # Get the worksheet
            worksheet = None
            try:
                # Find worksheet with gid 267803750
                for ws in spreadsheet.worksheets():
                    if str(ws.id) == "267803750":
                        worksheet = ws
                        break
            except:
                pass
            
            # Fallback to common sheet names
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
                    print(f"📍 Found Dog ID column at index {i} ('{header}')")
                    break
            
            if dog_id_col is None:
                print("❌ Could not find 'Dog ID' column")
                return False
            
            # CRITICAL: Target Column H (Combined column) - index 7
            target_col = 7  # Column H = index 7 (0-based)
            target_col_letter = chr(ord('A') + target_col)  # Convert to letter (H)
            print(f"📍 Writing to Column {target_col_letter} (Combined) at index {target_col}")
            
            # Prepare batch updates
            updates = []
            updates_count = 0
            
            print(f"\n🔍 Processing {len(reassignments)} reassignments...")
            
            for assignment in reassignments:
                # 🔧 CRITICAL FIX: Extract the correct values
                dog_id = str(assignment.get('dog_id', '')).strip()
                new_assignment = str(assignment.get('new_assignment', '')).strip()
                
                print(f"  🔍 Processing: Dog ID '{dog_id}' → New Assignment '{new_assignment}'")
                
                # 🚨 FINAL VALIDATION before writing
                if not new_assignment:
                    print(f"  ❌ SKIPPING: Empty new_assignment")
                    continue
                
                if new_assignment == dog_id:
                    print(f"  ❌ SKIPPING: new_assignment equals dog_id (BUG DETECTED!)")
                    continue
                
                if new_assignment.endswith('x') and new_assignment[:-1].isdigit():
                    print(f"  ❌ SKIPPING: new_assignment looks like dog_id (BUG DETECTED!)")
                    continue
                
                if ':' not in new_assignment:
                    print(f"  ❌ SKIPPING: new_assignment missing driver:group format")
                    continue
                
                print(f"  ✅ VALID: Will write '{new_assignment}' for dog '{dog_id}'")
                
                # Find the row for this dog ID
                found = False
                for row_idx in range(1, len(all_data)):  # Skip header row
                    if dog_id_col < len(all_data[row_idx]):
                        current_dog_id = str(all_data[row_idx][dog_id_col]).strip()
                        
                        if current_dog_id == dog_id:
                            # Convert to A1 notation (row is 1-indexed, col is 1-indexed)
                            cell_address = gspread.utils.rowcol_to_a1(row_idx + 1, target_col + 1)
                            
                            # 🔧 CRITICAL: Write the NEW_ASSIGNMENT (driver:group), NOT the dog_id!
                            updates.append({
                                'range': cell_address,
                                'values': [[new_assignment]]  # ← This is "Driver:Group", NOT dog_id!
                            })
                            
                            updates_count += 1
                            print(f"     📍 {dog_id} → {new_assignment} (cell {cell_address})")
                            found = True
                            break
                
                if not found:
                    print(f"  ⚠️ Could not find row for dog ID: {dog_id}")
            
            if not updates:
                print("❌ No valid updates to make")
                return False
            
            # 🔍 FINAL SAFETY CHECK: Inspect what we're about to write
            print(f"\n🔒 FINAL SAFETY CHECK before writing to Google Sheets:")
            invalid_updates = 0
            for i, update in enumerate(updates[:5]):  # Show first 5
                value = update['values'][0][0]
                cell = update['range']
                print(f"   {i+1}. Cell {cell} ← '{value}'")
                
                # Check for invalid values
                if value.endswith('x') and value[:-1].isdigit():
                    print(f"      🚨 INVALID: Looks like dog_id!")
                    invalid_updates += 1
                elif ':' not in value:
                    print(f"      🚨 INVALID: Missing driver:group format!")
                    invalid_updates += 1
                else:
                    print(f"      ✅ VALID: Proper driver:group format")
            
            if len(updates) > 5:
                print(f"   ... and {len(updates) - 5} more updates")
            
            if invalid_updates > 0:
                print(f"\n❌ ABORTING: Found {invalid_updates} invalid updates! Not writing to prevent data corruption!")
                return False
            
            print(f"\n✅ All {len(updates)} updates passed final safety check!")
            
            # Execute batch update
            print(f"\n📤 Writing {len(updates)} updates to Google Sheets...")
            
            worksheet.batch_update(updates)
            
            print(f"✅ Successfully updated {updates_count} assignments in Google Sheets!")
            print(f"🔒 All entries written in proper 'Driver:Group' format (NO 'x' entries!)")
            print(f"📏 Used RELAXED constraints: exact ≤3.3mi, adjacent ≤0.66mi, parking ≤0.44mi")
            
            # Optional: Send Slack notification
            slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    slack_message = {
                        "text": f"🐕 Dog Reassignment Complete: Updated {updates_count} assignments (RELAXED constraints - 10% more distance)"
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
    print("🚀 Production Dog Reassignment System - FIXED & RELAXED VERSION")
    print("🔒 Enhanced with extensive safety checks and bug fixes")
    print("📏 RELAXED Distance Limits (+10%): Same groups ≤3.3mi, Adjacent ≤0.66mi, Parking ≤0.44mi")
    print("🐕 FILTERED: Skips Parking/Field entries, processes only real dogs")
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
    
    # Write results with enhanced safety
    if reassignments:
        print(f"\n🔒 SAFETY REMINDER: About to write {len(reassignments)} assignments to Google Sheets")
        print(f"🔍 Each assignment will be in format 'Driver:Group' (like 'Nate:2', 'Jon:3DD3')")
        print(f"❌ NO 'x' entries will be written (bug has been fixed!)")
        
        write_success = system.write_results_to_sheets(reassignments)
        if write_success:
            print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments with RELAXED constraints")
            print(f"✅ All entries written in proper 'Driver:Group' format")
            print(f"📏 Used 10% more relaxed distance limits for better assignment success")
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")

if __name__ == "__main__":
    main()
