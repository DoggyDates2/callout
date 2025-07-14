# production_reassignment.py
# COMPLETE FIXED VERSION: Distance-first assignment with greedy walk optimization

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
        
        # DISTANCE LIMITS - RELAXED BY 10%
        self.MAX_DISTANCE = 3.3  # Hard limit: no assignments beyond 3.3 miles
        self.GREEDY_WALK_MAX_DISTANCE = 0.5  # For moving existing dogs to make space
        self.EXCLUSION_DISTANCE = 100.0  # Recognize 100+ as "do not assign" placeholders
        
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

    def _order_by_difficulty(self, dogs_to_reassign):
        """Order dogs by difficulty: multi-group and multi-dog first"""
        def difficulty_score(dog):
            score = 0
            
            # Multi-group assignments are harder (need capacity in multiple groups)
            if len(dog['needed_groups']) > 1:
                score += len(dog['needed_groups']) * 100  # 200 for 2&3, 300 for 1&2&3
            
            # Multi-dog households are harder (need more capacity)
            if dog['num_dogs'] > 1:
                score += dog['num_dogs'] * 50  # 100 for 2 dogs, 150 for 3 dogs
            
            return score
        
        sorted_dogs = sorted(dogs_to_reassign, key=difficulty_score, reverse=True)
        
        print("📋 Processing order (hardest first):")
        for i, dog in enumerate(sorted_dogs):
            groups_str = "&".join(map(str, dog['needed_groups']))
            print(f"   {i+1}. {dog['dog_name']} - Groups:{groups_str}, Dogs:{dog['num_dogs']} (difficulty: {difficulty_score(dog)})")
        
        return sorted_dogs

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

    def _run_greedy_assignment_with_order(self, ordered_dogs):
        """FIXED: Always choose closest driver with capacity, with greedy walk fallback"""
        print("   🎯 DISTANCE-FIRST assignment with greedy walk optimization")
        print("   📏 Priority: Closest driver wins")
        print("   🚶 Fallback: Move existing dogs to make space")
        
        assignments = []
        moves_made = []  # Track greedy walk moves
        
        # Build dynamic list of current assignments (will be updated as we assign)
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
        
        print(f"       📊 Starting with {len(current_assignments)} existing assignments")
        
        # Process each callout dog in difficulty order
        for callout_dog in ordered_dogs:
            print(f"\n   🐕 Processing {callout_dog['dog_name']} (Groups: {callout_dog['needed_groups']}, Dogs: {callout_dog['num_dogs']})")
            
            # Find ALL drivers with capacity, calculate distances to their dogs
            candidate_drivers = []
            
            for assignment in current_assignments:
                driver = assignment['driver']
                
                # Calculate distance to this dog
                distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                
                # Skip if too far or invalid
                if distance > self.MAX_DISTANCE or distance >= self.EXCLUSION_DISTANCE:
                    continue
                
                # Check if this driver has capacity in ALL needed groups
                current_load = self.calculate_driver_load(driver, current_assignments)
                has_capacity = True
                capacity_check = []
                
                for group in callout_dog['needed_groups']:
                    group_key = f'group{group}'
                    current = current_load.get(group_key, 0)
                    max_cap = self.driver_capacities.get(driver, {}).get(group_key, 0)
                    needed = callout_dog['num_dogs']
                    
                    if current + needed <= max_cap:
                        capacity_check.append(f"G{group}:✅")
                    else:
                        capacity_check.append(f"G{group}:❌")
                        has_capacity = False
                
                if has_capacity:
                    candidate_drivers.append({
                        'driver': driver,
                        'distance': distance,
                        'via_dog': assignment['dog_name'],
                        'via_dog_id': assignment['dog_id'],
                        'capacity_check': capacity_check
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
                
                print(f"       📏 Found {len(unique_candidates)} drivers with capacity (closest first):")
                for i, candidate in enumerate(unique_candidates[:3]):  # Show top 3
                    rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    print(f"         {rank} {candidate['driver']} - {candidate['distance']:.1f}mi via {candidate['via_dog']}")
                
                # Pick the closest driver
                winner = unique_candidates[0]
                driver = winner['driver']
                distance = winner['distance']
                
                print(f"       🏆 WINNER: {driver} (closest at {distance:.1f}mi)")
                
            else:
                # No drivers with available capacity - try greedy walk
                print(f"       🚫 No drivers with available capacity")
                
                greedy_move = self._try_greedy_walk(callout_dog, current_assignments)
                
                if not greedy_move:
                    print(f"       ❌ NO ASSIGNMENT POSSIBLE for {callout_dog['dog_name']}")
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
                
                print(f"         ✅ Moved {dog_to_move['dog_name']} from {original_driver} → {new_driver}")
                
                # Now assign the callout dog to the freed space
                driver = original_driver
                distance = self.get_distance(callout_dog['dog_id'], dog_to_move['dog_id'])
                
                print(f"         🎯 Space freed! Assigning {callout_dog['dog_name']} → {driver}")
            
            # Create the assignment
            new_assignment = f"{driver}:{callout_dog['full_assignment_string']}"
            
            print(f"         📝 {callout_dog['dog_name']} → {new_assignment}")
            print(f"         📏 Distance: {distance:.1f}mi")
            
            # Record the assignment
            assignments.append({
                'dog_id': callout_dog['dog_id'],
                'dog_name': callout_dog['dog_name'],
                'new_assignment': new_assignment,
                'driver': driver,
                'distance': distance,
                'assignment_type': 'regular',
                'match_type': 'closest_distance',
                'reason': f"closest_available_{distance:.1f}mi"
            })
            
            # Update current_assignments with the new dog
            current_assignments.append({
                'dog_id': callout_dog['dog_id'],
                'dog_name': callout_dog['dog_name'],
                'driver': driver,
                'needed_groups': callout_dog['needed_groups'],
                'num_dogs': callout_dog['num_dogs']
            })
            
            print(f"         📊 Total assignments now: {len(current_assignments)}")
        
        # Show greedy walk moves if any were made
        if moves_made:
            print(f"\n🚶 GREEDY WALK MOVES MADE ({len(moves_made)}):")
            for move in moves_made:
                print(f"   🔄 {move['dog_name']} moved from {move['from_driver']} → {move['to_driver']} ({move['distance']:.1f}mi)")
                print(f"      Reason: {move['reason']}")
        
        return assignments, moves_made

    def reassign_dogs_smart_optimization(self):
        """Main reassignment function with correct difficulty ordering and distance-first logic"""
        print("\n🔄 Starting SMART DISTANCE-FIRST ASSIGNMENT...")
        print("🎯 Order: Hardest dogs first (multi-group + multi-dog)")
        print("📏 Logic: Always choose closest driver with capacity")
        print("🚶 Fallback: Greedy walk to make space")
        print("🔄 State: Update immediately after each assignment")
        
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
            
            if any(available.values()):  # Only show drivers with availability
                print(f"   🚗 {driver}: G1:{available['group1']}, G2:{available['group2']}, G3:{available['group3']}")
        
        start_time = time.time()
        
        # Order by difficulty (hardest first)
        ordered_dogs = self._order_by_difficulty(dogs_to_reassign)
        
        # Run assignment with corrected logic
        assignments, moves_made = self._run_greedy_assignment_with_order(ordered_dogs)
        
        elapsed_time = time.time() - start_time
        
        # Show results
        print(f"\n🏆 ASSIGNMENT RESULTS:")
        print(f"   ⏱️ Processing time: {elapsed_time:.1f} seconds")
        print(f"   📊 Dogs assigned: {len(assignments)}/{num_dogs}")
        if moves_made:
            print(f"   🚶 Greedy walk moves: {len(moves_made)}")
        
        if assignments:
            print(f"\n🎉 SUCCESSFUL ASSIGNMENTS:")
            for assignment in assignments:
                print(f"      ✅ {assignment['dog_name']} → {assignment['new_assignment']}")
                print(f"         📏 {assignment['distance']:.1f}mi")
            
            # Validate assignments
            validated_assignments = []
            for assignment in assignments:
                new_assignment = assignment.get('new_assignment', '')
                if new_assignment and ':' in new_assignment and not new_assignment.endswith('x'):
                    validated_assignments.append(assignment)
                else:
                    print(f"   ❌ INVALID: {assignment.get('dog_name')} has bad assignment: {new_assignment}")
            
            print(f"\n🔒 VALIDATION: {len(validated_assignments)}/{len(assignments)} assignments are valid")
            
            # Store moves for potential writing to sheets
            self.greedy_moves_made = moves_made
            
            return validated_assignments
        
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
            
            success_msg = f"✅ Successfully updated {updates_count} assignments!"
            if hasattr(self, 'greedy_moves_made') and self.greedy_moves_made:
                success_msg += f" (including {len(self.greedy_moves_made)} greedy walk moves)"
            
            print(success_msg)
            print(f"🎯 Used distance-first logic with greedy walk optimization")
            
            # Send Slack notification
            slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    message = f"🐕 Dog Reassignment Complete: {updates_count} assignments updated using distance-first + greedy walk"
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
    print("🚀 Production Dog Reassignment System - DISTANCE-FIRST + GREEDY WALK")
    print("🎯 Hardest dogs first: Multi-group and multi-dog assignments")
    print("📏 Always choose closest driver with capacity")
    print("🚶 Greedy walk: Move existing dogs ≤0.5mi to make space")
    print("🔄 State updates: Immediate after each assignment")
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
            print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments")
            print(f"✅ Used distance-first logic with greedy walk optimization")
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")

if __name__ == "__main__":
    main()
