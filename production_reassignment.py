# production_reassignment.py
# Complete dog reassignment system with dynamic constraint-based optimization for 20+ dogs

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
        self.ADJACENT_MATCH_MAX_DISTANCE = 0.6  # Adjacent group matches: 0.6 miles
        self.PARKING_EXACT_MAX_DISTANCE = 0.4  # Parking/field exact matches: 0.4 miles
        self.PARKING_ADJACENT_MAX_DISTANCE = 0.2  # Parking/field adjacent matches: 0.2 miles
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

    def _determine_group_match_type(self, needed_groups, available_groups):
        """Determine if groups are exact match, adjacent, or no match"""
        # Convert to sets for easier comparison
        needed_set = set(needed_groups)
        available_set = set(available_groups)
        
        # Exact match: same groups
        if needed_set == available_set:
            return "exact"
        
        # Adjacent match: check if groups are numerically adjacent
        # Group 1 is adjacent to Group 2
        # Group 2 is adjacent to Groups 1 and 3  
        # Group 3 is adjacent to Group 2
        
        for needed in needed_groups:
            for available in available_groups:
                # Check if any needed group is adjacent to any available group
                if abs(needed - available) == 1:
                    return "adjacent"
        
        # No compatibility
        return "none"
    
    def _check_driver_capacity(self, driver, needed_groups, num_dogs, current_driver_loads):
        """Check if driver has capacity and return detailed results"""
        if driver not in self.driver_capacities:
            return False, "Driver not found in capacity list"
        
        driver_capacity = self.driver_capacities[driver]
        current_load = current_driver_loads.get(driver, {'group1': 0, 'group2': 0, 'group3': 0})
        
        capacity_details = []
        can_handle_all = True
        
        for group in needed_groups:
            group_key = f'group{group}'
            current = current_load.get(group_key, 0)
            capacity = driver_capacity.get(group_key, 0)
            needed = current + num_dogs
            
            if needed <= capacity:
                capacity_details.append(f"G{group}:{needed}/{capacity}")
            else:
                capacity_details.append(f"G{group}:{needed}/{capacity}(OVER)")
                can_handle_all = False
        
        detail_str = ", ".join(capacity_details)
        return can_handle_all, detail_str
    
    def _calculate_match_score(self, distance, match_type, dog_type):
        """Calculate a score for ranking matches (lower is better)"""
        # Base score is distance
        score = distance
        
        # Add penalties for less preferred matches
        if match_type == "adjacent":
            score += 10  # Prefer exact matches over adjacent
            
        if dog_type == "parking_field":
            score += 100  # Strongly prefer regular dogs
            
        return score

    def _attempt_greedy_walk_assignment(self, target_dog, target_driver, current_driver_loads, dogs_going_today):
        """Attempt to make room for target_dog by moving one of target_driver's current dogs"""
        print(f"         🔄 ATTEMPTING GREEDY WALK for {target_driver}:")
        print(f"            Goal: Make room for {target_dog['dog_name']} (Groups: {target_dog['needed_groups']})")
        
        # Find all dogs currently assigned to target_driver
        driver_current_dogs = [dog for dog in dogs_going_today if dog['driver'] == target_driver]
        
        if not driver_current_dogs:
            print(f"            ❌ No current dogs found for {target_driver}")
            return False, None
        
        print(f"            📊 {target_driver} currently has {len(driver_current_dogs)} dogs:")
        for dog in driver_current_dogs:
            print(f"               - {dog['dog_name']} (Groups: {dog['groups']}, {dog['num_dogs']} dogs)")
        
        # For each of target_driver's current dogs, see if they have close alternatives
        moveable_dogs = []
        
        for current_dog in driver_current_dogs:
            print(f"\n            🔍 Checking if {current_dog['dog_name']} can be moved:")
            
            # Find alternative drivers within 0.5 miles
            close_alternatives = []
            
            for other_dog in dogs_going_today:
                if other_dog['driver'] == target_driver:
                    continue  # Skip same driver
                
                distance = self.get_distance(current_dog['dog_id'], other_dog['dog_id'])
                
                if distance <= 0.5:  # Within 0.5 miles
                    # Check group compatibility
                    match_type = self._determine_group_match_type(current_dog['groups'], other_dog['groups'])
                    
                    if match_type in ["exact", "adjacent"]:
                        # Check if the alternative driver has capacity
                        alt_driver = other_dog['driver']
                        can_handle, capacity_details = self._check_driver_capacity(
                            alt_driver, current_dog['groups'], current_dog['num_dogs'], current_driver_loads
                        )
                        
                        if can_handle:
                            close_alternatives.append({
                                'alternative_dog': other_dog,
                                'distance': distance,
                                'match_type': match_type,
                                'capacity_details': capacity_details
                            })
                            
                            print(f"               ✅ {other_dog['dog_name']} → {alt_driver} at {distance:.2f}mi ({match_type}, {capacity_details})")
                        else:
                            print(f"               ❌ {other_dog['dog_name']} → {alt_driver} at {distance:.2f}mi (no capacity: {capacity_details})")
                    else:
                        print(f"               ❌ {other_dog['dog_name']} → {other_dog['driver']} at {distance:.2f}mi (no group compatibility)")
            
            if close_alternatives:
                # Sort by distance (closest first) and match type preference
                close_alternatives.sort(key=lambda x: (x['distance'], 0 if x['match_type'] == 'exact' else 1))
                best_alternative = close_alternatives[0]
                
                moveable_dogs.append({
                    'current_dog': current_dog,
                    'best_alternative': best_alternative,
                    'move_distance': best_alternative['distance']
                })
                
                print(f"               🎯 MOVEABLE: Best option is {best_alternative['alternative_dog']['dog_name']} → {best_alternative['alternative_dog']['driver']} at {best_alternative['distance']:.2f}mi")
            else:
                print(f"               ❌ No close alternatives found")
        
        if not moveable_dogs:
            print(f"            ❌ No dogs can be moved from {target_driver}")
            return False, None
        
        # Sort moveable dogs by move distance (easiest moves first)
        moveable_dogs.sort(key=lambda x: x['move_distance'])
        
        print(f"\n            🎯 MOVEABLE DOGS RANKED:")
        for i, move in enumerate(moveable_dogs):
            print(f"               {i+1}. {move['current_dog']['dog_name']} → can move {move['move_distance']:.2f}mi to {move['best_alternative']['alternative_dog']['driver']}")
        
        # Try to move the easiest dog first
        best_move = moveable_dogs[0]
        dog_to_move = best_move['current_dog']
        alternative = best_move['best_alternative']
        new_driver = alternative['alternative_dog']['driver']
        
        print(f"\n            🔄 EXECUTING MOVE:")
        print(f"               Moving: {dog_to_move['dog_name']} from {target_driver} → {new_driver}")
        print(f"               Distance: {alternative['distance']:.2f}mi via {alternative['alternative_dog']['dog_name']}")
        print(f"               Match type: {alternative['match_type']}")
        
        # Create the move assignment (we'll need to reconstruct the assignment string)
        # For now, let's assume we can move the dog with the same groups
        groups_string = ''.join(map(str, dog_to_move['groups']))
        move_assignment = f"{new_driver}:{groups_string}"
        
        move_result = {
            'dog_id': dog_to_move['dog_id'],
            'dog_name': dog_to_move['dog_name'],
            'old_driver': target_driver,
            'new_driver': new_driver,
            'new_assignment': move_assignment,
            'distance': alternative['distance'],
            'reason': f"greedy_walk_move_to_make_room_for_{target_dog['dog_name']}",
            'groups': dog_to_move['groups'],
            'num_dogs': dog_to_move['num_dogs']
        }
        
        print(f"               📝 Move assignment: {move_assignment}")
        print(f"            ✅ GREEDY WALK SUCCESS!")
        
        return True, move_result

    def _attempt_assignment_with_greedy_walk(self, callout_dog, all_potential_matches, current_driver_loads, dogs_going_today):
        """Try to assign a dog, using greedy walk if capacity is the only issue"""
        
        if not all_potential_matches:
            return False, None, None

    def get_distance(self, dog1_id: str, dog2_id: str) -> float:
        
        # Sort potential matches by score (best first)
        all_potential_matches.sort(key=lambda x: x['score'])
        
        # Try each potential match
        for match in all_potential_matches:
            driver = match['driver']
            
            # Check capacity
            capacity_ok, capacity_details = self._check_driver_capacity(
                driver, callout_dog['needed_groups'], 
                callout_dog['num_dogs'], current_driver_loads
            )
            
            if capacity_ok:
                # Direct assignment works
                print(f"         ✅ DIRECT ASSIGNMENT: {match['dog_name']} → {driver}")
                print(f"            Capacity: {capacity_details}")
                
                new_assignment = f"{driver}:{callout_dog['full_assignment_string']}"
                
                assignment_result = {
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'new_assignment': new_assignment,
                    'driver': driver,
                    'distance': match['distance'],
                    'closest_dog': match['dog_name'],
                    'reason': match['reason'],
                    'match_type': match['match_type'],
                    'dog_type': match['dog_type'],
                    'score': match['score']
                }
                
                return True, assignment_result, None
            
            else:
                # Capacity is the issue - try greedy walk
                print(f"         ⚠️ CAPACITY ISSUE for {driver}: {capacity_details}")
                
                # Only attempt greedy walk for high-quality matches (score < 15)
                if match['score'] < 15.0:
                    print(f"         🎯 High-quality match (score: {match['score']:.2f}) - attempting greedy walk...")
                    
                    walk_success, move_result = self._attempt_greedy_walk_assignment(
                        callout_dog, driver, current_driver_loads, dogs_going_today
                    )
                    
                    if walk_success:
                        # Greedy walk succeeded - we can now assign the original dog
                        new_assignment = f"{driver}:{callout_dog['full_assignment_string']}"
                        
                        assignment_result = {
                            'dog_id': callout_dog['dog_id'],
                            'dog_name': callout_dog['dog_name'],
                            'new_assignment': new_assignment,
                            'driver': driver,
                            'distance': match['distance'],
                            'closest_dog': match['dog_name'],
                            'reason': f"greedy_walk_{match['reason']}",
                            'match_type': match['match_type'],
                            'dog_type': match['dog_type'],
                            'score': match['score']
                        }
                        
                        print(f"         🎉 GREEDY WALK ASSIGNMENT SUCCESS!")
                        print(f"            Original dog: {callout_dog['dog_name']} → {driver}")
                        print(f"            New assignment: {new_assignment}")
                        
                        return True, assignment_result, move_result
                    else:
                        print(f"         ❌ Greedy walk failed for {driver}")
                else:
                    print(f"         ⏭️ Low-quality match (score: {match['score']:.2f}) - skipping greedy walk")
        
        # No assignments worked
        return False, None, None
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

    def _run_greedy_assignment_with_order(self, ordered_dogs, verbose_debug=True):
        """Run assignment keeping the exact assignment strings, only changing driver names"""
        if verbose_debug:
            print("   🎯 Finding drivers with capacity - preserving exact assignment strings...")
            print("   🔍 DEBUGGING MODE: Will show detailed decision-making process")
        
        # Build list of all dogs currently going today (with drivers)
        dogs_going_today = []
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if combined and ':' in combined:
                driver = combined.split(':', 1)[0].strip()
                assignment_string = combined.split(':', 1)[1].strip()
                groups = self._extract_groups_for_capacity_check(assignment_string)
                
                dogs_going_today.append({
                    'dog_id': assignment['dog_id'],
                    'dog_name': assignment['dog_name'],
                    'driver': driver,
                    'groups': groups,
                    'num_dogs': assignment['num_dogs']
                })
        
        if verbose_debug:
            print(f"   📊 Found {len(dogs_going_today)} dogs currently going today")
        
        # Categorize by regular vs parking/field
        regular_dogs = [dog for dog in dogs_going_today if dog['dog_name'].lower() not in ['parking', 'field']]
        parking_field_dogs = [dog for dog in dogs_going_today if dog['dog_name'].lower() in ['parking', 'field']]
        
        if verbose_debug:
            print(f"   🎯 Regular dogs: {len(regular_dogs)}, ⚠️ Parking/field dogs: {len(parking_field_dogs)}")
        
        assignments = []
        current_driver_loads = {}
        
        # Initialize driver loads
        for driver in self.driver_capacities.keys():
            current_driver_loads[driver] = self.calculate_driver_load(driver)
            
        if verbose_debug:
            print(f"   📋 Initial driver loads:")
            for driver, load in current_driver_loads.items():
                print(f"      🚗 {driver}: G1:{load['group1']}, G2:{load['group2']}, G3:{load['group3']}")

        # Process each callout dog in the specified order
        for callout_dog in ordered_dogs:
            if verbose_debug:
                print(f"\n" + "="*60)
                print(f"   🐕 PROCESSING: {callout_dog['dog_name']} ({callout_dog['dog_id']})")
                print(f"       📋 Original: {callout_dog['original_callout']}")
                print(f"       🎯 Assignment string: '{callout_dog['full_assignment_string']}'")
                print(f"       📊 Needs capacity in groups: {callout_dog['needed_groups']}")
                print(f"       🐕 Physical dogs: {callout_dog['num_dogs']}")
            
            # Analyze all potential matches with detailed scoring
            all_potential_matches = []
            
            if verbose_debug:
                print(f"\n   🔍 ANALYZING POTENTIAL MATCHES:")
            
            # Check regular dogs first
            if verbose_debug:
                print(f"   🎯 CHECKING REGULAR DOGS ({len(regular_dogs)} available):")
            for going_dog in regular_dogs:
                distance = self.get_distance(callout_dog['dog_id'], going_dog['dog_id'])
                
                # Determine match type and distance limit
                match_type = self._determine_group_match_type(callout_dog['needed_groups'], going_dog['groups'])
                
                if match_type == "exact":
                    distance_limit = self.EXACT_MATCH_MAX_DISTANCE  # 3.0
                    type_desc = "same groups"
                elif match_type == "adjacent":
                    distance_limit = self.ADJACENT_MATCH_MAX_DISTANCE  # 0.6
                    type_desc = "adjacent groups"
                else:
                    distance_limit = 0  # No match
                    type_desc = "no group compatibility"
                
                if verbose_debug:
                    print(f"      📏 {going_dog['dog_name']} (Driver: {going_dog['driver']})")
                    print(f"         Distance: {distance:.2f}mi, Groups: {going_dog['groups']} → {callout_dog['needed_groups']}")
                    print(f"         Match type: {match_type} ({type_desc}), Limit: {distance_limit}mi")
                
                if distance <= distance_limit and match_type != "none":
                    # Check capacity
                    capacity_ok, capacity_details = self._check_driver_capacity(
                        going_dog['driver'], callout_dog['needed_groups'], 
                        callout_dog['num_dogs'], current_driver_loads
                    )
                    
                    if verbose_debug:
                        print(f"         Capacity check: {'✅ PASS' if capacity_ok else '❌ FAIL'} - {capacity_details}")
                    
                    if capacity_ok:
                        # Calculate score for ranking
                        score = self._calculate_match_score(distance, match_type, "regular")
                        
                        all_potential_matches.append({
                            'dog_id': going_dog['dog_id'],
                            'dog_name': going_dog['dog_name'],
                            'driver': going_dog['driver'],
                            'groups': going_dog['groups'],
                            'distance': distance,
                            'match_type': match_type,
                            'dog_type': 'regular',
                            'score': score,
                            'reason': f"regular_{match_type}_match"
                        })
                        
                        if verbose_debug:
                            print(f"         🎯 VALID MATCH! Score: {score:.2f}")
                    elif verbose_debug:
                        print(f"         ❌ Capacity insufficient")
                elif verbose_debug:
                    if match_type == "none":
                        print(f"         ❌ No group compatibility")
                    else:
                        print(f"         ❌ Distance {distance:.2f}mi > {distance_limit}mi limit")
            
            # Check parking/field dogs as backup
            if verbose_debug:
                print(f"\n   ⚠️ CHECKING PARKING/FIELD DOGS ({len(parking_field_dogs)} available):")
            for going_dog in parking_field_dogs:
                distance = self.get_distance(callout_dog['dog_id'], going_dog['dog_id'])
                
                # Determine match type and distance limit (stricter for parking/field)
                match_type = self._determine_group_match_type(callout_dog['needed_groups'], going_dog['groups'])
                
                if match_type == "exact":
                    distance_limit = self.PARKING_EXACT_MAX_DISTANCE  # 0.4
                    type_desc = "same groups (parking)"
                elif match_type == "adjacent":
                    distance_limit = self.PARKING_ADJACENT_MAX_DISTANCE  # 0.2
                    type_desc = "adjacent groups (parking)"
                else:
                    distance_limit = 0  # No match
                    type_desc = "no group compatibility"
                
                if verbose_debug:
                    print(f"      📏 {going_dog['dog_name']} (Driver: {going_dog['driver']})")
                    print(f"         Distance: {distance:.2f}mi, Groups: {going_dog['groups']} → {callout_dog['needed_groups']}")
                    print(f"         Match type: {match_type} ({type_desc}), Limit: {distance_limit}mi")
                
                if distance <= distance_limit and match_type != "none":
                    # Check capacity
                    capacity_ok, capacity_details = self._check_driver_capacity(
                        going_dog['driver'], callout_dog['needed_groups'], 
                        callout_dog['num_dogs'], current_driver_loads
                    )
                    
                    if verbose_debug:
                        print(f"         Capacity check: {'✅ PASS' if capacity_ok else '❌ FAIL'} - {capacity_details}")
                    
                    if capacity_ok:
                        # Calculate score for ranking (lower score for parking/field)
                        score = self._calculate_match_score(distance, match_type, "parking_field")
                        
                        all_potential_matches.append({
                            'dog_id': going_dog['dog_id'],
                            'dog_name': going_dog['dog_name'],
                            'driver': going_dog['driver'],
                            'groups': going_dog['groups'],
                            'distance': distance,
                            'match_type': match_type,
                            'dog_type': 'parking_field',
                            'score': score,
                            'reason': f"parking_field_{match_type}_match"
                        })
                        
                        if verbose_debug:
                            print(f"         ⚠️ VALID BACKUP! Score: {score:.2f}")
                    elif verbose_debug:
                        print(f"         ❌ Capacity insufficient")
                elif verbose_debug:
                    if match_type == "none":
                        print(f"         ❌ No group compatibility")
                    else:
                        print(f"         ❌ Distance {distance:.2f}mi > {distance_limit}mi limit")
            
            # Sort potential matches by score (lower is better)
            all_potential_matches.sort(key=lambda x: x['score'])
            
            if verbose_debug:
                print(f"\n   📊 MATCH RANKING SUMMARY:")
                print(f"      Found {len(all_potential_matches)} valid potential matches")
            
            if all_potential_matches and verbose_debug:
                print(f"      🏆 TOP 5 CANDIDATES:")
                for i, match in enumerate(all_potential_matches[:5]):
                    rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                    type_emoji = "🎯" if match['dog_type'] == 'regular' else "⚠️"
                    match_emoji = "🎯" if match['match_type'] == 'exact' else "📍"
                    
                    print(f"         {rank_emoji} {type_emoji}{match_emoji} {match['dog_name']} → Driver {match['driver']}")
                    print(f"            Score: {match['score']:.2f}, Distance: {match['distance']:.2f}mi, {match['match_type']} match")
            
            # Try to assign to the best available match (with greedy walk if needed)
            assigned = False
            move_result = None
            
            if all_potential_matches:
                if verbose_debug:
                    print(f"\n   🎯 ATTEMPTING ASSIGNMENT (with greedy walk support):")
                
                # Use greedy walk assignment logic
                assignment_success, assignment_result, move_result = self._attempt_assignment_with_greedy_walk(
                    callout_dog, all_potential_matches, current_driver_loads, dogs_going_today
                )
                
                if assignment_success:
                    # APPLY THE MOVE FIRST (if there was one)
                    if move_result:
                        if verbose_debug:
                            print(f"\n      🔄 APPLYING GREEDY WALK MOVE:")
                            print(f"         Moving: {move_result['dog_name']} from {move_result['old_driver']} → {move_result['new_driver']}")
                        
                        # Add the move to our assignments list so it gets written to sheets
                        # We need to track this as a special "move" assignment
                        move_assignment_record = {
                            'dog_id': move_result['dog_id'],
                            'dog_name': move_result['dog_name'],
                            'new_assignment': move_result['new_assignment'],
                            'driver': move_result['new_driver'],
                            'distance': move_result['distance'],
                            'closest_dog': move_result.get('closest_dog', 'greedy_walk'),
                            'reason': move_result['reason'],
                            'match_type': 'greedy_walk_move',
                            'dog_type': 'moved_existing',
                            'score': 0,  # Special score for moves
                            'is_greedy_walk_move': True,
                            'original_assignment': f"{move_result['old_driver']}:{''.join(map(str, move_result['groups']))}"
                        }
                        assignments.append(move_assignment_record)
                        
                        # Update driver loads for the move
                        # Remove from old driver
                        if move_result['old_driver'] in current_driver_loads:
                            for group in move_result['groups']:
                                group_key = f'group{group}'
                                current_driver_loads[move_result['old_driver']][group_key] -= move_result['num_dogs']
                        
                        # Add to new driver
                        if move_result['new_driver'] not in current_driver_loads:
                            current_driver_loads[move_result['new_driver']] = {'group1': 0, 'group2': 0, 'group3': 0}
                        for group in move_result['groups']:
                            group_key = f'group{group}'
                            current_driver_loads[move_result['new_driver']][group_key] += move_result['num_dogs']
                        
                        # Update dogs_going_today list
                        for i, dog in enumerate(dogs_going_today):
                            if dog['dog_id'] == move_result['dog_id']:
                                dogs_going_today[i]['driver'] = move_result['new_driver']
                                break
                        
                        # Update regular_dogs and parking_field_dogs lists
                        for dog_list in [regular_dogs, parking_field_dogs]:
                            for i, dog in enumerate(dog_list):
                                if dog['dog_id'] == move_result['dog_id']:
                                    dog_list[i]['driver'] = move_result['new_driver']
                                    break
                        
                        if verbose_debug:
                            print(f"         📊 Updated loads after move:")
                            print(f"            {move_result['old_driver']}: {current_driver_loads.get(move_result['old_driver'], {})}")
                            print(f"            {move_result['new_driver']}: {current_driver_loads.get(move_result['new_driver'], {})}")
                            print(f"         📝 Move will be tracked: Callout = original assignment, Combined = new assignment")                    
                    # NOW APPLY THE MAIN ASSIGNMENT
                    assignments.append(assignment_result)
                    
                    if verbose_debug:
                        print(f"\n      ✅ MAIN ASSIGNMENT:")
                        print(f"         {assignment_result['dog_name']} → {assignment_result['new_assignment']}")
                        print(f"         Distance: {assignment_result['distance']:.1f}mi via {assignment_result['closest_dog']}")
                        if move_result:
                            print(f"         🎯 GREEDY WALK: Moved {move_result['dog_name']} to make room")
                    
                    # Update driver loads for the main assignment
                    driver = assignment_result['driver']
                    if driver not in current_driver_loads:
                        current_driver_loads[driver] = {'group1': 0, 'group2': 0, 'group3': 0}
                    
                    for group in callout_dog['needed_groups']:
                        group_key = f'group{group}'
                        current_driver_loads[driver][group_key] += callout_dog['num_dogs']
                    
                    if verbose_debug:
                        print(f"         📊 Updated {driver} load: {current_driver_loads[driver]}")
                    
                    # Add to dogs going today list for future iterations
                    dogs_going_today.append({
                        'dog_id': callout_dog['dog_id'],
                        'dog_name': callout_dog['dog_name'],
                        'driver': driver,
                        'groups': callout_dog['needed_groups'],
                        'num_dogs': callout_dog['num_dogs']
                    })
                    
                    # Update the appropriate category
                    dog_type = assignment_result.get('dog_type', 'regular')
                    if dog_type == 'regular':
                        regular_dogs.append({
                            'dog_id': callout_dog['dog_id'],
                            'dog_name': callout_dog['dog_name'],
                            'driver': driver,
                            'groups': callout_dog['needed_groups'],
                            'num_dogs': callout_dog['num_dogs']
                        })
                    else:
                        parking_field_dogs.append({
                            'dog_id': callout_dog['dog_id'],
                            'dog_name': callout_dog['dog_name'],
                            'driver': driver,
                            'groups': callout_dog['needed_groups'],
                            'num_dogs': callout_dog['num_dogs']
                        })
                    
                    assigned = True
                    if verbose_debug:
                        print(f"      ✅ ASSIGNMENT COMPLETE!")
                        if move_result:
                            print(f"      📊 Total changes: 1 new assignment + 1 greedy walk move")
                        else:
                            print(f"      📊 Total changes: 1 direct assignment")
            
            if not assigned:
                if verbose_debug:
                    print(f"\n   ❌ NO VALID ASSIGNMENT FOUND")
                    print(f"      All potential drivers either:")
                    print(f"      - Too far away (>3mi for regular exact, >0.6mi for regular adjacent)")
                    print(f"      - Lack capacity in needed groups {callout_dog['needed_groups']} (even with greedy walk)")
                    print(f"      - No group compatibility")
                    print(f"      - Parking/field too far (>0.4mi exact, >0.2mi adjacent)")
        
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
            assignments = self._run_greedy_assignment_with_order(list(dog_order), verbose_debug=False)
            
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
        
        # Show detailed results
        self._show_detailed_assignment_results(best_assignments, "Optimal Permutation")
        
        return best_assignments

    def reassign_dogs_smart_optimization(self):
        """Smart dynamic constraint-based optimization for large numbers of callout dogs"""
        print("\n🔄 Starting SMART DYNAMIC OPTIMIZATION...")
        print("🎯 Using intelligent constraint-based selection (same groups ≤3mi, adjacent ≤0.6mi, parking ≤0.4mi/0.2mi)")
        
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
        elif num_dogs <= 12:
            print(f"🧠 Medium number of dogs - using hybrid approach (critical subset + greedy)")
            return self._hybrid_critical_subset_assignment(dogs_to_reassign)
        else:
            print(f"🚀 Large number of dogs - using dynamic constraint-based assignment")
            # For 20+ dogs, use adaptive debugging (full for first 3 iterations, summary after)
            return self._dynamic_constraint_assignment(dogs_to_reassign, debug_verbosity="adaptive")

    def _hybrid_critical_subset_assignment(self, dogs_to_reassign):
        """Hybrid approach: Try permutations on most critical dogs, greedy for rest"""
        print("\n🎯 HYBRID CRITICAL SUBSET APPROACH:")
        print("   Step 1: Identify most critical dogs for permutation testing")
        print("   Step 2: Try all permutations of critical subset")
        print("   Step 3: Greedily assign remaining dogs")
        
        start_time = time.time()
        
        # Step 1: Identify the most critical dogs (hardest to assign)
        critical_count = min(6, len(dogs_to_reassign))  # Max 6 for permutations
        difficulty_ordered = self._order_by_difficulty(dogs_to_reassign)
        critical_dogs = difficulty_ordered[:critical_count]
        remaining_dogs = difficulty_ordered[critical_count:]
        
        print(f"\n📊 CRITICAL SUBSET ANALYSIS:")
        print(f"   🎯 Critical dogs (permutation testing): {critical_count}")
        print(f"   🔄 Remaining dogs (greedy assignment): {len(remaining_dogs)}")
        
        print(f"\n   🏆 CRITICAL DOGS:")
        for i, dog in enumerate(critical_dogs):
            print(f"      {i+1}. {dog['dog_name']} - Groups: {dog['needed_groups']}")
        
        # Step 2: Try all permutations of critical dogs
        best_assignments = []
        best_cost = float('inf')
        best_critical_order = None
        
        num_critical_perms = 1
        for i in range(1, len(critical_dogs) + 1):
            num_critical_perms *= i
        
        print(f"\n🧪 TESTING {num_critical_perms} CRITICAL PERMUTATIONS:")
        
        for perm_idx, critical_order in enumerate(itertools.permutations(critical_dogs)):
            # Combine critical order + remaining dogs in difficulty order
            full_order = list(critical_order) + remaining_dogs
            
            # Run assignment
            assignments = self._run_greedy_assignment_with_order(full_order, verbose_debug=False)
            cost = self._calculate_permutation_cost(assignments, full_order)
            
            if perm_idx < 5 or cost < best_cost:  # Show first 5 and any improvements
                critical_names = [dog['dog_name'] for dog in critical_order]
                print(f"   📋 Permutation {perm_idx + 1}: {' → '.join(critical_names)} + {len(remaining_dogs)} more")
                print(f"      📊 Result: {len(assignments)}/{len(full_order)} assigned, cost: {cost:.1f}")
            
            if cost < best_cost:
                best_cost = cost
                best_assignments = assignments
                best_critical_order = [dog['dog_name'] for dog in critical_order]
                print(f"      🏆 NEW BEST!")
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🏆 HYBRID OPTIMIZATION RESULTS:")
        print(f"   ⏱️ Processing time: {elapsed_time:.1f} seconds")
        print(f"   🎯 Best critical order: {' → '.join(best_critical_order)}")
        print(f"   📊 Dogs assigned: {len(best_assignments)}/{len(dogs_to_reassign)}")
        print(f"   📏 Total cost: {best_cost:.1f} miles")
        
        # Show detailed results
        self._show_detailed_assignment_results(best_assignments, "Hybrid Critical Subset")
        
        return best_assignments

    def _dynamic_constraint_assignment(self, dogs_to_reassign, debug_verbosity="full"):
        """Dynamic constraint-based assignment: Pick next dog based on current state"""
        print("\n🚀 DYNAMIC CONSTRAINT-BASED ASSIGNMENT:")
        print("   🎯 Strategy: Always pick the most constrained remaining dog next")
        print("   🔄 Recalculate constraints after each assignment")
        
        if debug_verbosity == "full":
            print("   🔍 DEBUG MODE: Full detailed analysis for each assignment")
        elif debug_verbosity == "summary":
            print("   📊 SUMMARY MODE: Constraint analysis + assignment results only")
        
        start_time = time.time()
        
        # Initialize state
        remaining_dogs = dogs_to_reassign.copy()
        assignments = []
        iteration = 0
        
        # Track state for constraint calculation
        dogs_going_today = []
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if combined and ':' in combined:
                driver = combined.split(':', 1)[0].strip()
                assignment_string = combined.split(':', 1)[1].strip()
                groups = self._extract_groups_for_capacity_check(assignment_string)
                
                dogs_going_today.append({
                    'dog_id': assignment['dog_id'],
                    'dog_name': assignment['dog_name'],
                    'driver': driver,
                    'groups': groups,
                    'num_dogs': assignment['num_dogs']
                })
        
        current_driver_loads = {}
        for driver in self.driver_capacities.keys():
            current_driver_loads[driver] = self.calculate_driver_load(driver)
        
        print(f"\n🔄 DYNAMIC ASSIGNMENT ITERATIONS:")
        print(f"💡 TIP: Each iteration shows constraint analysis + detailed assignment debugging")
        
        while remaining_dogs:
            iteration += 1
            print(f"\n" + "="*50)
            print(f"   🔄 ITERATION {iteration}: {len(remaining_dogs)} dogs remaining")
            
            # Recalculate constraints for all remaining dogs based on current state
            constraint_scores = []
            
            for dog in remaining_dogs:
                # Count current valid options for this dog
                valid_options = 0
                closest_distance = float('inf')
                
                for going_dog in dogs_going_today:
                    distance = self.get_distance(dog['dog_id'], going_dog['dog_id'])
                    
                    # Check if within any distance limit
                    dog_name = going_dog['dog_name'].lower()
                    is_parking_field = dog_name in ['parking', 'field']
                    match_type = self._determine_group_match_type(dog['needed_groups'], going_dog['groups'])
                    
                    # Determine distance limit
                    if match_type == "exact":
                        limit = self.PARKING_EXACT_MAX_DISTANCE if is_parking_field else self.EXACT_MATCH_MAX_DISTANCE
                    elif match_type == "adjacent":
                        limit = self.PARKING_ADJACENT_MAX_DISTANCE if is_parking_field else self.ADJACENT_MATCH_MAX_DISTANCE
                    else:
                        continue  # No group compatibility
                    
                    if distance <= limit:
                        # Check capacity
                        driver = going_dog['driver']
                        can_handle, _ = self._check_driver_capacity(
                            driver, dog['needed_groups'], dog['num_dogs'], current_driver_loads
                        )
                        
                        if can_handle:
                            valid_options += 1
                            closest_distance = min(closest_distance, distance)
                
                # Calculate constraint score (higher = more constrained = higher priority)
                if valid_options == 0:
                    constraint_score = 1000  # Highest priority - no options left
                else:
                    # Fewer options = higher score, farther distance = higher score
                    constraint_score = (10 / valid_options) + (closest_distance * 2)
                
                constraint_scores.append({
                    'dog': dog,
                    'constraint_score': constraint_score,
                    'valid_options': valid_options,
                    'closest_distance': closest_distance
                })
            
            # Sort by constraint score (highest first = most constrained first)
            constraint_scores.sort(key=lambda x: x['constraint_score'], reverse=True)
            
            # Show top 5 most constrained
            print(f"   📊 CONSTRAINT RANKING (most constrained first):")
            for i, item in enumerate(constraint_scores[:5]):
                dog = item['dog']
                score = item['constraint_score']
                options = item['valid_options']
                distance = item['closest_distance']
                
                status = "🚨" if options == 0 else "⚠️" if options <= 2 else "📍"
                print(f"      {i+1}. {status} {dog['dog_name']} - Score: {score:.1f}")
                print(f"         Options: {options}, Closest: {distance:.2f}mi, Groups: {dog['needed_groups']}")
            
            # Pick the most constrained dog
            most_constrained = constraint_scores[0]['dog']
            print(f"\n   🎯 SELECTING MOST CONSTRAINED: {most_constrained['dog_name']}")
            print(f"      Constraint score: {constraint_scores[0]['constraint_score']:.1f}")
            print(f"      Valid options: {constraint_scores[0]['valid_options']}")
            
            # Try to assign this dog  
            print(f"\n   🎯 ATTEMPTING ASSIGNMENT:")
            print(f"      🐕 Dog: {most_constrained['dog_name']}")
            print(f"      📊 Groups needed: {most_constrained['needed_groups']}")
            print(f"      🎯 Assignment string: '{most_constrained['full_assignment_string']}'")
            
            # Use appropriate debug level (full debug for first few, summary for later iterations)
            verbose_debug = (iteration <= 3) if debug_verbosity == "adaptive" else (debug_verbosity == "full")
            
            # Enable detailed debugging for this assignment
            single_dog_assignment = self._run_greedy_assignment_with_order([most_constrained], verbose_debug=verbose_debug)
            
            if single_dog_assignment:
                # Success! Update state
                assignment = single_dog_assignment[0]
                assignments.append(assignment)
                
                print(f"      ✅ ASSIGNED: {assignment['new_assignment']}")
                print(f"         Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
                
                # Update dogs_going_today
                dogs_going_today.append({
                    'dog_id': most_constrained['dog_id'],
                    'dog_name': most_constrained['dog_name'],
                    'driver': assignment['driver'],
                    'groups': most_constrained['needed_groups'],
                    'num_dogs': most_constrained['num_dogs']
                })
                
                # Update driver loads
                for group in most_constrained['needed_groups']:
                    group_key = f'group{group}'
                    if assignment['driver'] not in current_driver_loads:
                        current_driver_loads[assignment['driver']] = {'group1': 0, 'group2': 0, 'group3': 0}
                    current_driver_loads[assignment['driver']][group_key] += most_constrained['num_dogs']
                
            else:
                print(f"      ❌ COULD NOT ASSIGN: {most_constrained['dog_name']}")
                print(f"         No valid drivers found with current constraints")
            
            # Remove from remaining dogs
            remaining_dogs.remove(most_constrained)
            
            print(f"      📊 Progress: {len(assignments)} assigned, {len(remaining_dogs)} remaining")
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🏆 DYNAMIC OPTIMIZATION RESULTS:")
        print(f"   ⏱️ Processing time: {elapsed_time:.1f} seconds")
        print(f"   🔄 Iterations: {iteration}")
        print(f"   📊 Dogs assigned: {len(assignments)}/{len(dogs_to_reassign)}")
        print(f"   📏 Total cost: {sum(a['distance'] for a in assignments):.1f} miles")
        
        # Show detailed results
        self._show_detailed_assignment_results(assignments, "Dynamic Constraint-Based")
        
        return assignments

    def _show_detailed_assignment_results(self, assignments, method_name):
        """Show detailed assignment results with quality breakdown"""
        if not assignments:
            print(f"\n❌ NO ASSIGNMENTS FOUND with {method_name}")
            return
        
        print(f"\n🎉 OPTIMAL ASSIGNMENTS ({method_name}):")
        
        # Group assignments by type for better organization
        perfect_matches = [a for a in assignments if a.get('match_type') == 'exact' and a.get('dog_type') == 'regular']
        good_matches = [a for a in assignments if a.get('match_type') == 'adjacent' and a.get('dog_type') == 'regular']
        backup_matches = [a for a in assignments if a.get('dog_type') == 'parking_field' and a.get('match_type') == 'exact']
        last_resort = [a for a in assignments if a.get('dog_type') == 'parking_field' and a.get('match_type') == 'adjacent']
        
        # Separate greedy walk assignments
        greedy_walk_assignments = [a for a in assignments if 'greedy_walk' in a.get('reason', '')]
        direct_assignments = [a for a in assignments if 'greedy_walk' not in a.get('reason', '')]
        
        if greedy_walk_assignments:
            print(f"\n      🔄 GREEDY WALK SUCCESSES (made room by moving existing dogs):")
            for assignment in greedy_walk_assignments:
                assignment_type = "🎯" if assignment.get('dog_type') == 'regular' else "⚠️"
                match_type = "exact" if assignment.get('match_type') == 'exact' else "adjacent"
                print(f"         {assignment_type} {assignment['dog_name']} → {assignment['new_assignment']}")
                print(f"            Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']} ({match_type})")
                print(f"            🔄 Required moving another dog to make capacity")
                if 'score' in assignment:
                    print(f"            Score: {assignment['score']:.2f}, Reason: {assignment.get('reason', 'N/A')}")
        
        if perfect_matches:
            perfect_direct = [a for a in perfect_matches if a not in greedy_walk_assignments]
            if perfect_direct:
                print(f"\n      🎯 PERFECT (regular exact matches - direct):")
                for assignment in perfect_direct:
                    print(f"         ✅ {assignment['dog_name']} → {assignment['new_assignment']}")
                    print(f"            Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
                    if 'score' in assignment:
                        print(f"            Score: {assignment['score']:.2f}, Reason: {assignment.get('reason', 'N/A')}")
        
        if good_matches:
            good_direct = [a for a in good_matches if a not in greedy_walk_assignments]
            if good_direct:
                print(f"\n      📍 GOOD (regular adjacent matches - direct):")
                for assignment in good_direct:
                    print(f"         ✅ {assignment['dog_name']} → {assignment['new_assignment']}")
                    print(f"            Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
                    if 'score' in assignment:
                        print(f"            Score: {assignment['score']:.2f}, Reason: {assignment.get('reason', 'N/A')}")
        
        if backup_matches:
            backup_direct = [a for a in backup_matches if a not in greedy_walk_assignments]
            if backup_direct:
                print(f"\n      🔄 BACKUP (parking/field exact matches ≤0.4mi - direct):")
                for assignment in backup_direct:
                    print(f"         ⚠️ {assignment['dog_name']} → {assignment['new_assignment']}")
                    print(f"            Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
                    if 'score' in assignment:
                        print(f"            Score: {assignment['score']:.2f}, Reason: {assignment.get('reason', 'N/A')}")
        
        if last_resort:
            last_resort_direct = [a for a in last_resort if a not in greedy_walk_assignments]
            if last_resort_direct:
                print(f"\n      🆘 LAST RESORT (parking/field adjacent matches ≤0.2mi - direct):")
                for assignment in last_resort_direct:
                    print(f"         🆘 {assignment['dog_name']} → {assignment['new_assignment']}")
                    print(f"            Distance: {assignment['distance']:.1f}mi via {assignment['closest_dog']}")
                    if 'score' in assignment:
                        print(f"            Score: {assignment['score']:.2f}, Reason: {assignment.get('reason', 'N/A')}")
        
        # Show overall summary
        print(f"\n      📊 ASSIGNMENT SUMMARY:")
        print(f"         🔄 Greedy walk successes: {len(greedy_walk_assignments)}")
        print(f"         🎯 Perfect direct matches: {len([a for a in perfect_matches if a not in greedy_walk_assignments])}")
        print(f"         📍 Good direct matches: {len([a for a in good_matches if a not in greedy_walk_assignments])}")
        print(f"         🔄 Backup direct matches: {len([a for a in backup_matches if a not in greedy_walk_assignments])}")  
        print(f"         🆘 Last resort direct: {len([a for a in last_resort if a not in greedy_walk_assignments])}")
        if assignments:
            print(f"         📏 Average distance: {sum(a['distance'] for a in assignments) / len(assignments):.2f}mi")
        
        if greedy_walk_assignments:
            print(f"\n      🎯 GREEDY WALK IMPACT:")
            print(f"         Without greedy walk: {len(direct_assignments)} assignments")
            print(f"         With greedy walk: {len(assignments)} assignments (+{len(greedy_walk_assignments)})")
            print(f"         Improvement: {len(greedy_walk_assignments) / len(assignments) * 100:.1f}% of assignments used greedy walk")

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
            
            # Column indices
            combined_col = 7   # Column H (Combined)
            callout_col = 10   # Column K (Callout)
            
            print(f"📍 Writing to Column H (Combined) at index {combined_col}")
            print(f"📍 Writing to Column K (Callout) at index {callout_col} for moved dogs")
            
            # Separate regular assignments from greedy walk moves
            regular_assignments = [a for a in reassignments if not a.get('is_greedy_walk_move', False)]
            greedy_walk_moves = [a for a in reassignments if a.get('is_greedy_walk_move', False)]
            
            print(f"\n🔍 Processing assignments:")
            print(f"   📊 Regular callout assignments: {len(regular_assignments)}")
            print(f"   🔄 Greedy walk moves: {len(greedy_walk_moves)}")
            
            # Prepare batch updates
            updates = []
            updates_count = 0
            
            # Process regular assignments (write to Combined column only)
            for assignment in regular_assignments:
                dog_id = str(assignment['dog_id']).strip()
                new_assignment = assignment['new_assignment']
                
                # Find the row for this dog ID
                found = False
                for row_idx in range(1, len(all_data)):  # Skip header row
                    if dog_id_col < len(all_data[row_idx]):
                        current_dog_id = str(all_data[row_idx][dog_id_col]).strip()
                        
                        if current_dog_id == dog_id:
                            # Convert to A1 notation (row is 1-indexed, col is 1-indexed)
                            cell_address = gspread.utils.rowcol_to_a1(row_idx + 1, combined_col + 1)
                            
                            updates.append({
                                'range': cell_address,
                                'values': [[new_assignment]]
                            })
                            
                            updates_count += 1
                            print(f"  ✅ {dog_id} → {new_assignment} (cell {cell_address}) [REGULAR]")
                            found = True
                            break
                
                if not found:
                    print(f"  ⚠️ Could not find row for dog ID: {dog_id}")
            
            # Process greedy walk moves (write to BOTH Callout and Combined columns)
            for move in greedy_walk_moves:
                dog_id = str(move['dog_id']).strip()
                new_assignment = move['new_assignment']
                original_assignment = move['original_assignment']
                
                # Find the row for this dog ID
                found = False
                for row_idx in range(1, len(all_data)):  # Skip header row
                    if dog_id_col < len(all_data[row_idx]):
                        current_dog_id = str(all_data[row_idx][dog_id_col]).strip()
                        
                        if current_dog_id == dog_id:
                            # Write original assignment to Callout column (K)
                            callout_cell_address = gspread.utils.rowcol_to_a1(row_idx + 1, callout_col + 1)
                            updates.append({
                                'range': callout_cell_address,
                                'values': [[original_assignment]]
                            })
                            
                            # Write new assignment to Combined column (H)
                            combined_cell_address = gspread.utils.rowcol_to_a1(row_idx + 1, combined_col + 1)
                            updates.append({
                                'range': combined_cell_address,
                                'values': [[new_assignment]]
                            })
                            
                            updates_count += 2  # Two cells updated
                            print(f"  🔄 {dog_id} MOVED:")
                            print(f"     Callout (K): {original_assignment} (cell {callout_cell_address})")
                            print(f"     Combined (H): {new_assignment} (cell {combined_cell_address})")
                            found = True
                            break
                
                if not found:
                    print(f"  ⚠️ Could not find row for moved dog ID: {dog_id}")
            
            if not updates:
                print("❌ No valid updates to make")
                return False
            
            # Execute batch update
            print(f"\n📤 Sending {len(updates)} cell updates to Google Sheets...")
            
            # Use batch_update for efficiency
            worksheet.batch_update(updates)
            
            assignment_count = len(regular_assignments) + len(greedy_walk_moves)
            print(f"✅ Successfully updated {assignment_count} assignments ({len(regular_assignments)} regular + {len(greedy_walk_moves)} moves) in Google Sheets!")
            
            # Optional: Send Slack notification if webhook is configured
            slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    slack_message = {
                        "text": f"🐕 Dog Reassignment Complete: Updated {assignment_count} assignments ({len(regular_assignments)} regular + {len(greedy_walk_moves)} greedy walk moves) using dynamic constraint-based optimization with greedy walk"
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
    print("🚀 Production Dog Reassignment System - DYNAMIC CONSTRAINT OPTIMIZATION")
    print("🎯 Smart approaches: ≤6 dogs=exhaustive, ≤12 dogs=hybrid, >12 dogs=dynamic")
    print("🎯 Group-based matching: same groups ≤3mi, adjacent ≤0.6mi, parking ≤0.4mi/0.2mi")
    print("🔄 Greedy walk: Makes room by moving existing dogs <0.5mi to alternative drivers")
    print("🔍 DEBUGGING: Full detailed analysis showing why each dog is selected")
    print("=" * 75)
    
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
            print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments using dynamic constraint-based optimization with greedy walk")
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")

if __name__ == "__main__":
    main()
