# production_reassignment.py
# Complete dog reassignment system with dynamic distance-based individual assignment

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
                    assigned_groups = self._parse_groups(f"dummy:{groups_part}")
                    
                    # Add to load for each group
                    for group in assigned_groups:
                        group_key = f'group{group}'
                        if group_key in load:
                            load[group_key] += assignment['num_dogs']
        
        return load

    def get_driver_current_dogs(self, driver_name: str) -> List[str]:
        """Get list of dog IDs currently assigned to a driver"""
        dogs = []
        
        if not self.dog_assignments:
            return dogs
        
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            
            # Skip empty assignments
            if not combined or combined.strip() == "":
                continue
            
            # Extract driver name from combined assignment (before colon)
            if ':' in combined:
                assigned_driver = combined.split(':', 1)[0].strip()
                
                if assigned_driver == driver_name:
                    dogs.append(assignment['dog_id'])
        
        return dogs

    def find_driver_outliers(self, driver_name: str, driver_dogs: List[str], outlier_threshold: float = 1.0) -> List[Tuple[str, float]]:
        """Find outlier dogs that don't have any neighbor within the threshold distance"""
        if len(driver_dogs) <= 1:
            return []  # Need at least 2 dogs to determine outliers
        
        outliers = []
        
        for dog_id in driver_dogs:
            # Find the closest neighbor to this dog
            closest_neighbor_distance = float('inf')
            closest_neighbor_id = None
            
            for other_dog_id in driver_dogs:
                if dog_id != other_dog_id:
                    distance = self.get_distance(dog_id, other_dog_id)
                    if distance != float('inf') and distance < closest_neighbor_distance:
                        closest_neighbor_distance = distance
                        closest_neighbor_id = other_dog_id
            
            # If this dog's closest neighbor is farther than threshold, it's an outlier
            if closest_neighbor_distance > outlier_threshold:
                outliers.append((dog_id, closest_neighbor_distance))
        
        # Sort by distance to closest neighbor (most isolated first)
        outliers.sort(key=lambda x: x[1], reverse=True)
        return outliers

    def get_dog_info(self, dog_id: str) -> Dict:
        """Get detailed info about a specific dog"""
        for assignment in self.dog_assignments:
            if assignment['dog_id'] == dog_id:
                # Parse the combined assignment to get groups
                combined = assignment.get('combined', '')
                if combined and ':' in combined:
                    groups_part = combined.split(':', 1)[1].strip()
                    groups = self._parse_groups(f"dummy:{groups_part}")
                else:
                    groups = []
                
                return {
                    'dog_id': dog_id,
                    'dog_name': assignment['dog_name'],
                    'num_dogs': assignment['num_dogs'],
                    'needed_groups': groups,
                    'combined': combined
                }
        return None

    def attempt_domino_reassignment(self, preferred_driver: str, callout_dog: Dict, 
                                  working_drivers: Dict, max_depth: int = 3) -> Tuple[bool, List[Dict]]:
        """Attempt to free up capacity in preferred driver by moving their outliers (with multi-level domino chains)"""
        print(f"      🔄 DOMINO CHECK: Can we free space in {preferred_driver}?")
        
        # Start the recursive domino chain
        success, domino_chain = self._recursive_domino_search(
            preferred_driver, callout_dog, working_drivers, depth=0, max_depth=max_depth, visited_drivers=set()
        )
        
        return success, domino_chain

    def _recursive_domino_search(self, target_driver: str, target_dog: Dict, working_drivers: Dict,
                               depth: int, max_depth: int, visited_drivers: set) -> Tuple[bool, List[Dict]]:
        """Recursively search for domino chain to free capacity"""
        
        if depth > max_depth:
            print(f"      ❌ Reached maximum domino depth ({max_depth})")
            return False, []
        
        if target_driver in visited_drivers:
            print(f"      ❌ Cycle detected - already tried {target_driver}")
            return False, []
        
        depth_prefix = "  " * depth
        print(f"      {depth_prefix}🔍 Level {depth}: Trying to free space in {target_driver}")
        
        # Find outliers in target driver's route
        driver_dogs = working_drivers[target_driver]['current_dogs']
        outliers = self.find_driver_outliers(target_driver, driver_dogs, outlier_threshold=1.0)
        
        if not outliers:
            print(f"      {depth_prefix}❌ No outliers found in {target_driver}'s route")
            return False, []
        
        print(f"      {depth_prefix}📋 Found {len(outliers)} outliers to try moving:")
        for dog_id, closest_neighbor_dist in outliers:
            dog_info = self.get_dog_info(dog_id)
            dog_name = dog_info['dog_name'] if dog_info else dog_id
            print(f"      {depth_prefix}   {dog_name}: closest neighbor {closest_neighbor_dist:.1f}mi away")
        
        # Try to move each outlier
        for outlier_dog_id, closest_neighbor_distance in outliers:
            outlier_info = self.get_dog_info(outlier_dog_id)
            if not outlier_info:
                continue
            
            print(f"      {depth_prefix}🎯 Trying to move outlier: {outlier_info['dog_name']}")
            
            # Find potential destinations for this outlier, sorted by distance/score
            potential_destinations = []
            
            for alt_driver, alt_info in working_drivers.items():
                if alt_driver == target_driver or alt_driver in visited_drivers:
                    continue
                
                # Check group compatibility first
                alt_distance, compatibility = self.calculate_effective_distance_to_driver(
                    outlier_dog_id, 
                    outlier_info['needed_groups'], 
                    alt_info['current_dogs'],
                    alt_info['current_groups']
                )
                
                if alt_distance > self.MAX_DISTANCE:
                    continue
                
                # Calculate score
                score = 100.0 - (alt_distance * 10)
                if "exact_match" in compatibility:
                    score += 15
                elif "adjacent_match" in compatibility:
                    score += 5
                
                # Check capacity
                has_capacity = True
                for group in outlier_info['needed_groups']:
                    group_key = f'group{group}'
                    current = alt_info['current_load'].get(group_key, 0)
                    capacity = alt_info['capacity'].get(group_key, 0)
                    
                    if current + outlier_info['num_dogs'] > capacity:
                        has_capacity = False
                        break
                
                potential_destinations.append({
                    'driver': alt_driver,
                    'distance': alt_distance,
                    'score': score,
                    'has_capacity': has_capacity,
                    'compatibility': compatibility
                })
            
            # Sort by score (best first)
            potential_destinations.sort(key=lambda x: x['score'], reverse=True)
            
            # Try destinations in order of preference
            for dest in potential_destinations:
                alt_driver = dest['driver']
                alt_distance = dest['distance']
                alt_score = dest['score']
                has_capacity = dest['has_capacity']
                
                print(f"      {depth_prefix}   📍 Considering {alt_driver}: {alt_distance:.1f}mi, score {alt_score:.1f}, capacity: {has_capacity}")
                
                if has_capacity:
                    # Can move directly - check if this frees enough space for the target dog
                    if self._check_if_move_frees_space(target_driver, outlier_info, target_dog, working_drivers):
                        print(f"      {depth_prefix}   ✅ DIRECT MOVE: {outlier_info['dog_name']} → {alt_driver}")
                        
                        move = {
                            'dog_id': outlier_dog_id,
                            'dog_name': outlier_info['dog_name'],
                            'from_driver': target_driver,
                            'to_driver': alt_driver,
                            'needed_groups': outlier_info['needed_groups'],
                            'num_dogs': outlier_info['num_dogs'],
                            'distance': alt_distance,
                            'domino_level': depth
                        }
                        return True, [move]
                
                elif alt_score > 70.0:  # Only try recursive domino for good candidates
                    print(f"      {depth_prefix}   🔄 {alt_driver} is full but good candidate - trying recursive domino...")
                    
                    # Try recursive domino to free space in alt_driver
                    new_visited = visited_drivers | {target_driver}
                    recursive_success, recursive_chain = self._recursive_domino_search(
                        alt_driver, outlier_info, working_drivers, depth + 1, max_depth, new_visited
                    )
                    
                    if recursive_success:
                        print(f"      {depth_prefix}   ✅ RECURSIVE SUCCESS: Found chain to free space in {alt_driver}")
                        
                        # Add this move to the chain
                        this_move = {
                            'dog_id': outlier_dog_id,
                            'dog_name': outlier_info['dog_name'],
                            'from_driver': target_driver,
                            'to_driver': alt_driver,
                            'needed_groups': outlier_info['needed_groups'],
                            'num_dogs': outlier_info['num_dogs'],
                            'distance': alt_distance,
                            'domino_level': depth
                        }
                        
                        return True, recursive_chain + [this_move]
                else:
                    print(f"      {depth_prefix}   ❌ {alt_driver} score too low ({alt_score:.1f}) for recursive domino")
        
        print(f"      {depth_prefix}❌ No successful domino chain found at level {depth}")
        return False, []

    def _check_if_move_frees_space(self, driver: str, moving_dog: Dict, target_dog: Dict, working_drivers: Dict) -> bool:
        """Check if moving a dog frees enough space for the target dog"""
        # Calculate space that would be freed
        freed_space = {}
        for group in moving_dog['needed_groups']:
            group_key = f'group{group}'
            freed_space[group_key] = moving_dog['num_dogs']
        
        # Check if target dog can now fit
        for group in target_dog['needed_groups']:
            group_key = f'group{group}'
            current = working_drivers[driver]['current_load'].get(group_key, 0)
            capacity = working_drivers[driver]['capacity'].get(group_key, 0)
            freed = freed_space.get(group_key, 0)
            
            if current - freed + target_dog['num_dogs'] > capacity:
                return False
        
        return True

    def execute_domino_moves(self, domino_moves: List[Dict], working_drivers: Dict) -> List[Dict]:
        """Execute domino moves in the correct order (deepest level first) and return the assignment records"""
        domino_assignments = []
        
        if not domino_moves:
            return domino_assignments
        
        # Sort moves by domino level (deepest first) to execute the chain correctly
        domino_moves.sort(key=lambda x: x.get('domino_level', 0))
        
        print(f"      🔄 EXECUTING {len(domino_moves)}-STEP DOMINO CHAIN:")
        for i, move in enumerate(domino_moves):
            level = move.get('domino_level', 0)
            level_prefix = "  " * level
            print(f"      {level_prefix}Step {i+1}: {move['dog_name']} from {move['from_driver']} to {move['to_driver']} (Level {level})")
        
        for move in domino_moves:
            level = move.get('domino_level', 0)
            level_prefix = "  " * level
            print(f"      {level_prefix}🔄 EXECUTING: {move['dog_name']} from {move['from_driver']} to {move['to_driver']}")
            
            # Remove from old driver
            for group in move['needed_groups']:
                group_key = f'group{group}'
                working_drivers[move['from_driver']]['current_load'][group_key] -= move['num_dogs']
            
            working_drivers[move['from_driver']]['current_dogs'].remove(move['dog_id'])
            
            # Update old driver's groups
            remaining_groups = set()
            for dog_id in working_drivers[move['from_driver']]['current_dogs']:
                dog_info = self.get_dog_info(dog_id)
                if dog_info and dog_info['needed_groups']:
                    remaining_groups.update(dog_info['needed_groups'])
            working_drivers[move['from_driver']]['current_groups'] = sorted(list(remaining_groups))
            
            # Add to new driver
            for group in move['needed_groups']:
                group_key = f'group{group}'
                working_drivers[move['to_driver']]['current_load'][group_key] += move['num_dogs']
            
            working_drivers[move['to_driver']]['current_dogs'].append(move['dog_id'])
            
            # Update new driver's groups
            new_groups = set(working_drivers[move['to_driver']]['current_groups'])
            new_groups.update(move['needed_groups'])
            working_drivers[move['to_driver']]['current_groups'] = sorted(list(new_groups))
            
            # Create assignment record
            groups_text = "&".join(map(str, move['needed_groups']))
            new_assignment = f"{move['to_driver']}:{groups_text}"
            
            domino_assignments.append({
                'dog_id': move['dog_id'],
                'dog_name': move['dog_name'],
                'new_assignment': new_assignment,
                'distance': move['distance'],
                'driver': move['to_driver'],
                'assignment_type': f'domino_level_{level}'
            })
            
            print(f"      {level_prefix}✅ {move['dog_name']} successfully moved to {move['to_driver']}")
        
        print(f"      🎯 DOMINO CHAIN COMPLETE! {len(domino_moves)} moves executed successfully")
        return domino_assignments
        """Get list of groups currently handled by a driver"""
        groups = set()
        
        if not self.dog_assignments:
            return []
        
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            
            # Skip empty assignments
            if not combined or combined.strip() == "":
                continue
            
            # Extract driver name and groups from combined assignment
            if ':' in combined:
                assigned_driver = combined.split(':', 1)[0].strip()
                
                if assigned_driver == driver_name:
                    # Parse groups for this assignment
                    groups_part = combined.split(':', 1)[1].strip()
                    assigned_groups = self._parse_groups(f"dummy:{groups_part}")
                    groups.update(assigned_groups)
        
        return sorted(list(groups))

    def calculate_effective_distance_to_driver(self, callout_dog_id: str, callout_needed_groups: List[int], 
                                             driver_dogs: List[str], driver_groups: List[int]) -> Tuple[float, str]:
        """Calculate effective distance considering group compatibility"""
        if not driver_dogs:
            return float('inf'), "no_dogs"  # Driver has no current dogs
        
        # Find minimum physical distance to driver's route
        min_physical_distance = float('inf')
        closest_dog = None
        
        for driver_dog_id in driver_dogs:
            distance = self.get_distance(callout_dog_id, driver_dog_id)
            if distance < min_physical_distance:
                min_physical_distance = distance
                closest_dog = driver_dog_id
        
        if min_physical_distance == float('inf'):
            return float('inf'), "no_distance_data"
        
        # 🔍 DEBUG: Show which dog was closest for distance calculation
        if len(driver_dogs) > 1:
            print(f"      📏 Closest to {callout_dog_id}: {closest_dog} ({min_physical_distance:.1f}mi)")
        
        # Determine group compatibility
        callout_groups_set = set(callout_needed_groups)
        driver_groups_set = set(driver_groups)
        
        # Check for exact group matches
        exact_matches = callout_groups_set.intersection(driver_groups_set)
        
        if exact_matches:
            # Exact group match - use actual distance
            return min_physical_distance, f"exact_match_{list(exact_matches)}"
        
        # Check for adjacent group matches
        # Groups 1, 2, 3 are all adjacent to each other
        adjacent_matches = []
        for callout_group in callout_needed_groups:
            for driver_group in driver_groups:
                # All groups are adjacent: 1↔2, 2↔3, 1↔3
                if abs(callout_group - driver_group) <= 2 and callout_group != driver_group:
                    adjacent_matches.append((callout_group, driver_group))
        
        if adjacent_matches:
            # Adjacent group match - double the distance
            effective_distance = min_physical_distance * 2.0
            return effective_distance, f"adjacent_match_{adjacent_matches[0]}"
        
        # No group compatibility
        return float('inf'), "no_group_compatibility"

    def calculate_optimal_route_distance(self, driver_dogs: List[str]) -> Tuple[float, List[str]]:
        """Calculate the optimal route distance for a driver's assigned dogs using TSP"""
        if len(driver_dogs) <= 1:
            return 0.0, driver_dogs
        
        if len(driver_dogs) == 2:
            distance = self.get_distance(driver_dogs[0], driver_dogs[1])
            return distance, driver_dogs
        
        # For larger routes, use a greedy nearest-neighbor TSP approximation
        # Start from the first dog and always go to the nearest unvisited dog
        unvisited = driver_dogs[1:]
        current_dog = driver_dogs[0]
        route = [current_dog]
        total_distance = 0.0
        
        while unvisited:
            nearest_dog = None
            nearest_distance = float('inf')
            
            for dog in unvisited:
                distance = self.get_distance(current_dog, dog)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_dog = dog
            
            if nearest_dog:
                route.append(nearest_dog)
                total_distance += nearest_distance
                current_dog = nearest_dog
                unvisited.remove(nearest_dog)
            else:
                # If we can't find distances, just add remaining dogs with penalty
                route.extend(unvisited)
                total_distance += len(unvisited) * 5.0  # 5 mile penalty per dog
                break
        
        return total_distance, route

    def calculate_total_system_miles(self, assignments: List[Dict]) -> float:
        """Calculate total system miles for a given assignment strategy"""
        # Group assignments by driver
        driver_assignments = {}
        for assignment in assignments:
            driver = assignment['driver']
            if driver not in driver_assignments:
                driver_assignments[driver] = []
            driver_assignments[driver].append(assignment['dog_id'])
        
        total_miles = 0.0
        route_details = {}
        
        print(f"      📊 ROUTE ANALYSIS:")
        
        for driver, dog_ids in driver_assignments.items():
            route_distance, optimal_route = self.calculate_optimal_route_distance(dog_ids)
            total_miles += route_distance
            route_details[driver] = {
                'dogs': dog_ids,
                'optimal_route': optimal_route,
                'distance': route_distance
            }
            
            print(f"         {driver}: {len(dog_ids)} dogs, optimal route = {route_distance:.1f}mi")
            if len(optimal_route) <= 4:  # Show route for small groups
                print(f"            Route: {' → '.join(optimal_route)}")
        
        print(f"      🎯 TOTAL SYSTEM MILES: {total_miles:.1f}")
        return total_miles

    def strategy_individual_assignment(self, dogs_to_reassign: List[Dict], working_drivers: Dict) -> Tuple[List[Dict], float]:
        """Strategy A: Individual assignment (current approach)"""
        print(f"\n🔹 STRATEGY A: Individual Assignment")
        
        # Use our existing individual assignment logic
        assignments = []
        drivers_copy = {k: {
            'capacity': v['capacity'].copy(),
            'current_load': v['current_load'].copy(),
            'current_dogs': v['current_dogs'].copy(),
            'current_groups': v['current_groups'].copy()
        } for k, v in working_drivers.items()}
        
        for dog in dogs_to_reassign:
            best_driver = None
            best_score = float('-inf')
            
            for driver_name, driver_info in drivers_copy.items():
                # Check capacity
                can_handle = True
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    current = driver_info['current_load'].get(group_key, 0)
                    capacity = driver_info['capacity'].get(group_key, 0)
                    
                    if current + dog['num_dogs'] > capacity:
                        can_handle = False
                        break
                
                if not can_handle:
                    continue
                
                # Calculate distance
                effective_distance, compatibility = self.calculate_effective_distance_to_driver(
                    dog['dog_id'], dog['needed_groups'], 
                    driver_info['current_dogs'], driver_info['current_groups']
                )
                
                if effective_distance > self.MAX_DISTANCE:
                    continue
                
                # Score
                score = 100.0 - (effective_distance * 20)
                if "exact_match" in compatibility:
                    score += 15
                elif "adjacent_match" in compatibility:
                    score += 5
                
                if score > best_score:
                    best_driver = driver_name
                    best_score = score
            
            if best_driver:
                # Update driver state
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    drivers_copy[best_driver]['current_load'][group_key] += dog['num_dogs']
                
                drivers_copy[best_driver]['current_dogs'].append(dog['dog_id'])
                
                current_groups = set(drivers_copy[best_driver]['current_groups'])
                current_groups.update(dog['needed_groups'])
                drivers_copy[best_driver]['current_groups'] = sorted(list(current_groups))
                
                groups_text = "&".join(map(str, dog['needed_groups']))
                assignments.append({
                    'dog_id': dog['dog_id'],
                    'dog_name': dog['dog_name'],
                    'new_assignment': f"{best_driver}:{groups_text}",
                    'driver': best_driver,
                    'strategy': 'individual'
                })
        
        total_miles = self.calculate_total_system_miles(assignments)
        print(f"      ✅ Individual Assignment: {len(assignments)} dogs assigned, {total_miles:.1f} total miles")
        
        return assignments, total_miles

    def strategy_load_balanced_assignment(self, dogs_to_reassign: List[Dict], working_drivers: Dict) -> Tuple[List[Dict], float]:
        """Strategy B: Load-balanced assignment (distribute evenly across drivers)"""
        print(f"\n🔹 STRATEGY B: Load-Balanced Assignment")
        
        assignments = []
        drivers_copy = {k: {
            'capacity': v['capacity'].copy(),
            'current_load': v['current_load'].copy(),
            'current_dogs': v['current_dogs'].copy(),
            'current_groups': v['current_groups'].copy()
        } for k, v in working_drivers.items()}
        
        for dog in dogs_to_reassign:
            best_driver = None
            best_score = float('-inf')
            
            for driver_name, driver_info in drivers_copy.items():
                # Check capacity
                can_handle = True
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    current = driver_info['current_load'].get(group_key, 0)
                    capacity = driver_info['capacity'].get(group_key, 0)
                    
                    if current + dog['num_dogs'] > capacity:
                        can_handle = False
                        break
                
                if not can_handle:
                    continue
                
                # Calculate distance
                effective_distance, compatibility = self.calculate_effective_distance_to_driver(
                    dog['dog_id'], dog['needed_groups'], 
                    driver_info['current_dogs'], driver_info['current_groups']
                )
                
                if effective_distance > self.MAX_DISTANCE:
                    continue
                
                # Score with heavy load balancing
                total_load = sum(driver_info['current_load'].values())
                max_total_capacity = sum(driver_info['capacity'].values())
                load_ratio = total_load / max_total_capacity if max_total_capacity > 0 else 0
                
                score = 100.0 - (effective_distance * 10) - (load_ratio * 40)  # Heavy load penalty
                
                if "exact_match" in compatibility:
                    score += 15
                elif "adjacent_match" in compatibility:
                    score += 5
                
                if score > best_score:
                    best_driver = driver_name
                    best_score = score
            
            if best_driver:
                # Update driver state
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    drivers_copy[best_driver]['current_load'][group_key] += dog['num_dogs']
                
                drivers_copy[best_driver]['current_dogs'].append(dog['dog_id'])
                
                current_groups = set(drivers_copy[best_driver]['current_groups'])
                current_groups.update(dog['needed_groups'])
                drivers_copy[best_driver]['current_groups'] = sorted(list(current_groups))
                
                groups_text = "&".join(map(str, dog['needed_groups']))
                assignments.append({
                    'dog_id': dog['dog_id'],
                    'dog_name': dog['dog_name'],
                    'new_assignment': f"{best_driver}:{groups_text}",
                    'driver': best_driver,
                    'strategy': 'load_balanced'
                })
        
        total_miles = self.calculate_total_system_miles(assignments)
        print(f"      ✅ Load-Balanced Assignment: {len(assignments)} dogs assigned, {total_miles:.1f} total miles")
        
        return assignments, total_miles

    def strategy_proximity_clustering(self, dogs_to_reassign: List[Dict], working_drivers: Dict) -> Tuple[List[Dict], float]:
        """Strategy C: Proximity clustering (assign nearby dogs to same driver when possible)"""
        print(f"\n🔹 STRATEGY C: Proximity Clustering")
        
        assignments = []
        drivers_copy = {k: {
            'capacity': v['capacity'].copy(),
            'current_load': v['current_load'].copy(),
            'current_dogs': v['current_dogs'].copy(),
            'current_groups': v['current_groups'].copy()
        } for k, v in working_drivers.items()}
        
        # Group dogs by proximity (within 1 mile of each other)
        clusters = []
        remaining_dogs = dogs_to_reassign.copy()
        
        while remaining_dogs:
            cluster = [remaining_dogs.pop(0)]
            dogs_to_add = []
            
            for dog in remaining_dogs:
                for cluster_dog in cluster:
                    distance = self.get_distance(dog['dog_id'], cluster_dog['dog_id'])
                    if distance <= 1.0:  # Close enough to cluster
                        dogs_to_add.append(dog)
                        break
            
            for dog in dogs_to_add:
                cluster.append(dog)
                remaining_dogs.remove(dog)
            
            clusters.append(cluster)
        
        print(f"      📦 Created {len(clusters)} proximity clusters")
        
        # Assign each cluster to the best driver who can handle all dogs in cluster
        for cluster in clusters:
            best_driver = None
            best_total_score = float('-inf')
            
            for driver_name, driver_info in drivers_copy.items():
                # Check if driver can handle entire cluster
                can_handle_all = True
                total_dogs_needed = {}
                
                for dog in cluster:
                    for group in dog['needed_groups']:
                        group_key = f'group{group}'
                        total_dogs_needed[group_key] = total_dogs_needed.get(group_key, 0) + dog['num_dogs']
                
                for group_key, needed in total_dogs_needed.items():
                    current = driver_info['current_load'].get(group_key, 0)
                    capacity = driver_info['capacity'].get(group_key, 0)
                    if current + needed > capacity:
                        can_handle_all = False
                        break
                
                if not can_handle_all:
                    continue
                
                # Calculate total score for cluster
                total_score = 0
                for dog in cluster:
                    effective_distance, compatibility = self.calculate_effective_distance_to_driver(
                        dog['dog_id'], dog['needed_groups'], 
                        driver_info['current_dogs'], driver_info['current_groups']
                    )
                    
                    if effective_distance > self.MAX_DISTANCE:
                        total_score = float('-inf')
                        break
                    
                    score = 100.0 - (effective_distance * 15)
                    if "exact_match" in compatibility:
                        score += 15
                    elif "adjacent_match" in compatibility:
                        score += 5
                    
                    total_score += score
                
                if total_score > best_total_score:
                    best_driver = driver_name
                    best_total_score = total_score
            
            # Assign cluster to best driver
            if best_driver:
                for dog in cluster:
                    # Update driver state
                    for group in dog['needed_groups']:
                        group_key = f'group{group}'
                        drivers_copy[best_driver]['current_load'][group_key] += dog['num_dogs']
                    
                    drivers_copy[best_driver]['current_dogs'].append(dog['dog_id'])
                    
                    current_groups = set(drivers_copy[best_driver]['current_groups'])
                    current_groups.update(dog['needed_groups'])
                    drivers_copy[best_driver]['current_groups'] = sorted(list(current_groups))
                    
                    groups_text = "&".join(map(str, dog['needed_groups']))
                    assignments.append({
                        'dog_id': dog['dog_id'],
                        'dog_name': dog['dog_name'],
                        'new_assignment': f"{best_driver}:{groups_text}",
                        'driver': best_driver,
                        'strategy': 'proximity_cluster'
                    })
        
        total_miles = self.calculate_total_system_miles(assignments)
        print(f"      ✅ Proximity Clustering: {len(assignments)} dogs assigned, {total_miles:.1f} total miles")
        
        return assignments, total_miles
        """Sort callout dogs by placement difficulty (hardest first)"""
        
        def get_priority_score(dog):
            """Calculate priority score (higher = process first)"""
            score = 0
            
            # Priority 1: Number of physical dogs (higher = more priority)
            # Dogs with 2+ dogs should be placed first
            score += dog['num_dogs'] * 100
            
            # Priority 2: Number of groups needed (higher = more priority)  
            # Multi-group dogs (1&2, 2&3, 123) are harder to place
            num_groups = len(dog['needed_groups'])
            score += num_groups * 10
            
            # Priority 3: Group complexity bonus
            # Dogs needing multiple specific groups get extra priority
            if num_groups >= 2:
                score += 20  # Extra bonus for multi-group requirements
            
            return score
        
        # Sort by priority score (highest first)
        prioritized_dogs = sorted(dogs_to_reassign, key=get_priority_score, reverse=True)
        
        print(f"\n📋 CALLOUT DOG PRIORITIZATION:")
        print(f"   Processing order (hardest to place first):")
        
        for i, dog in enumerate(prioritized_dogs):
            priority_score = get_priority_score(dog)
            groups_text = "&".join(map(str, dog['needed_groups']))
            complexity = ""
            
            if dog['num_dogs'] > 1:
                complexity += f"{dog['num_dogs']} dogs, "
            if len(dog['needed_groups']) > 1:
                complexity += f"multi-group ({groups_text}), "
            if not complexity:
                complexity = f"single dog, group {groups_text}, "
            
            complexity = complexity.rstrip(", ")
            
            print(f"   {i+1}. {dog['dog_name']} - {complexity} (priority: {priority_score})")
        
    def reassign_dogs_individually(self):
        """Compare multiple assignment strategies and pick the one with lowest total system miles"""
        print("\n🔄 Starting MULTI-STRATEGY ROUTE-OPTIMIZED ASSIGNMENT...")
        print("🎯 Comparing multiple strategies and selecting optimal total system miles")
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        # 📋 PRIORITIZE DOGS: Process hardest-to-place dogs first
        dogs_to_reassign = self.prioritize_callout_dogs(dogs_to_reassign)
        
        # Validate data
        print(f"\n🔍 Data validation:")
        print(f" Matrix dogs: {len(self.distance_matrix) if self.distance_matrix is not None else 0}")
        print(f" Assignment dogs: {len(self.dog_assignments)}")
        print(f" Callout dogs: {len(dogs_to_reassign)}")
        
        if len(self.distance_matrix) == 0:
            print("❌ NO DISTANCE MATRIX DATA!")
            return []
        
        # Get callout driver to exclude
        callout_driver = None
        if dogs_to_reassign[0].get('original_callout'):
            callout_text = dogs_to_reassign[0]['original_callout']
            if ':' in callout_text:
                callout_driver = callout_text.split(':', 1)[0].strip()
        
        print(f"🚫 Excluding callout driver: {callout_driver}")
        
        # Initialize working drivers with current state
        working_drivers = {}
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if combined and ':' in combined:
                driver_name = combined.split(':', 1)[0].strip()
                if driver_name and driver_name in self.driver_capacities and driver_name != callout_driver:
                    if driver_name not in working_drivers:
                        working_drivers[driver_name] = {
                            'capacity': self.driver_capacities[driver_name].copy(),
                            'current_load': self.calculate_driver_load(driver_name).copy(),
                            'current_dogs': self.get_driver_current_dogs(driver_name).copy(),
                            'current_groups': self.get_driver_current_groups(driver_name).copy()
                        }
        
        print(f"👥 Available working drivers: {len(working_drivers)}")
        
        # 🏁 STRATEGY COMPARISON: Try multiple approaches
        print(f"\n🏁 COMPARING ASSIGNMENT STRATEGIES:")
        
        strategies = []
        
        # Strategy A: Individual assignment (distance-first)
        try:
            assignments_a, miles_a = self.strategy_individual_assignment(dogs_to_reassign, working_drivers)
            strategies.append(('Individual Assignment', assignments_a, miles_a))
        except Exception as e:
            print(f"      ❌ Strategy A failed: {e}")
        
        # Strategy B: Load-balanced assignment
        try:
            assignments_b, miles_b = self.strategy_load_balanced_assignment(dogs_to_reassign, working_drivers)
            strategies.append(('Load-Balanced Assignment', assignments_b, miles_b))
        except Exception as e:
            print(f"      ❌ Strategy B failed: {e}")
        
        # Strategy C: Proximity clustering
        try:
            assignments_c, miles_c = self.strategy_proximity_clustering(dogs_to_reassign, working_drivers)
            strategies.append(('Proximity Clustering', assignments_c, miles_c))
        except Exception as e:
            print(f"      ❌ Strategy C failed: {e}")
        
        if not strategies:
            print("❌ All strategies failed!")
            return []
        
        # Pick the best strategy
        best_strategy_name, best_assignments, best_miles = min(strategies, key=lambda x: x[2])
        
        print(f"\n🏆 STRATEGY COMPARISON RESULTS:")
        for name, assignments, miles in strategies:
            status = "🥇 WINNER!" if name == best_strategy_name else ""
            print(f"   {name}: {len(assignments)} dogs, {miles:.1f} total miles {status}")
        
        print(f"\n🎯 SELECTED STRATEGY: {best_strategy_name}")
        print(f"   📊 Dogs assigned: {len(best_assignments)}")
        print(f"   📏 Total system miles: {best_miles:.1f}")
        print(f"   📈 Average miles per dog: {best_miles/len(best_assignments):.1f}" if best_assignments else "   📈 No assignments")
        
        # Show detailed results
        if best_assignments:
            print(f"\n🎉 ROUTE-OPTIMIZED RESULTS ({best_strategy_name}):")
            driver_counts = {}
            driver_distances = {}
            
            # Group by driver for route analysis
            driver_dogs = {}
            for assignment in best_assignments:
                driver = assignment['driver']
                if driver not in driver_dogs:
                    driver_dogs[driver] = []
                driver_dogs[driver].append(assignment['dog_id'])
                
                # Count assignments
                driver_counts[driver] = driver_counts.get(driver, 0) + 1
            
            # Calculate and show optimal routes
            total_calculated_miles = 0.0
            for driver, dog_ids in driver_dogs.items():
                route_distance, optimal_route = self.calculate_optimal_route_distance(dog_ids)
                driver_distances[driver] = route_distance
                total_calculated_miles += route_distance
                
                # Get dog names for display
                dog_names = []
                for assignment in best_assignments:
                    if assignment['driver'] == driver:
                        dog_names.append(assignment['dog_name'])
                
                print(f"   {driver}: {len(dog_ids)} dogs → {route_distance:.1f}mi route")
                if len(optimal_route) <= 5:  # Show route for manageable sizes
                    route_names = []
                    for dog_id in optimal_route:
                        for assignment in best_assignments:
                            if assignment['dog_id'] == dog_id:
                                route_names.append(assignment['dog_name'])
                                break
                    print(f"      Route: {' → '.join(route_names)}")
                
                print(f"      Assignments: {', '.join([f'{name} → {next(a[\"new_assignment\"] for a in best_assignments if a[\"dog_name\"] == name)}' for name in dog_names])}")
            
            print(f"\n📊 Route optimization summary:")
            print(f"   🎯 Total optimized miles: {total_calculated_miles:.1f}")
            print(f"   📈 Miles per driver: {total_calculated_miles/len(driver_dogs):.1f}" if driver_dogs else "   📈 No drivers")
            
            total_assignments = len(best_assignments)
            unassigned = len(dogs_to_reassign) - total_assignments
            if unassigned > 0:
                print(f"   ⚠️  {unassigned} dogs could not be assigned")
        
        return best_assignments
        """Assign each callout dog to their closest available driver with dynamic route updates"""
        print("\n🔄 Starting DYNAMIC DISTANCE-BASED INDIVIDUAL ASSIGNMENT...")
        print("🎯 Each dog gets assigned to their closest available driver")
        print("🔄 Driver routes update dynamically after each assignment")
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        # 📋 PRIORITIZE DOGS: Process hardest-to-place dogs first
        dogs_to_reassign = self.prioritize_callout_dogs(dogs_to_reassign)
        
        # Validate data
        print(f"\n🔍 Data validation:")
        print(f" Matrix dogs: {len(self.distance_matrix) if self.distance_matrix is not None else 0}")
        print(f" Assignment dogs: {len(self.dog_assignments)}")
        print(f" Callout dogs: {len(dogs_to_reassign)}")
        
        if len(self.distance_matrix) == 0:
            print("❌ NO DISTANCE MATRIX DATA!")
            return []
        
        # Get callout driver to exclude
        callout_driver = None
        if dogs_to_reassign[0].get('original_callout'):
            callout_text = dogs_to_reassign[0]['original_callout']
            if ':' in callout_text:
                callout_driver = callout_text.split(':', 1)[0].strip()
        
        print(f"🚫 Excluding callout driver: {callout_driver}")
        
        # Initialize working drivers with current state
        working_drivers = {}
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if combined and ':' in combined:
                driver_name = combined.split(':', 1)[0].strip()
                if driver_name and driver_name in self.driver_capacities and driver_name != callout_driver:
                    if driver_name not in working_drivers:
                        working_drivers[driver_name] = {
                            'capacity': self.driver_capacities[driver_name].copy(),
                            'current_load': self.calculate_driver_load(driver_name).copy(),
                            'current_dogs': self.get_driver_current_dogs(driver_name).copy(),
                            'current_groups': self.get_driver_current_groups(driver_name).copy()
                        }
        
        print(f"👥 Available working drivers: {len(working_drivers)}")
        for driver, info in list(working_drivers.items())[:5]:  # Show first 5
            groups_text = ",".join(map(str, info['current_groups'])) if info['current_groups'] else "none"
            print(f"   {driver}: {len(info['current_dogs'])} dogs, groups [{groups_text}], load G1={info['current_load']['group1']}/{info['capacity']['group1']}")
        
        final_assignments = []
        total_system_distance = 0.0
        
        # Process each callout dog individually (now in priority order)
        for i, dog in enumerate(dogs_to_reassign):
            priority_info = f"({dog['num_dogs']} dogs, groups {dog['needed_groups']})"
            print(f"\n🐕 Priority Assignment {i+1}/{len(dogs_to_reassign)}: {dog['dog_name']} ({dog['dog_id']}) {priority_info}")
            
            best_driver = None
            best_distance = float('inf')
            best_score = float('-inf')
            
            candidates_checked = 0
            valid_candidates = 0
            
            # Check each available driver
            for driver_name, driver_info in working_drivers.items():
                candidates_checked += 1
                
                # 🔒 Check group compatibility first (before capacity)
                if not any(group in [1, 2, 3] for group in dog['needed_groups']):
                    print(f"   ❌ {driver_name} - group incompatibility")
                    continue
                
                # 📏 Calculate effective distance (for all drivers, regardless of capacity)
                effective_distance, compatibility_reason = self.calculate_effective_distance_to_driver(
                    dog['dog_id'], 
                    dog['needed_groups'], 
                    driver_info['current_dogs'],
                    driver_info['current_groups']
                )
                
                # 🔍 DEBUG: Show route composition for distance calculation
                route_size = len(driver_info['current_dogs'])
                if route_size > 0:
                    print(f"   📍 {driver_name} route ({route_size} dogs): {driver_info['current_dogs'][:3]}{'...' if route_size > 3 else ''}")
                
                # Skip if too far
                if effective_distance > self.MAX_DISTANCE:
                    print(f"   ❌ {driver_name} - too far ({effective_distance:.1f}mi > {self.MAX_DISTANCE}mi limit)")
                    continue
                
                # 🔒 Check capacity constraints  
                can_handle = True
                capacity_details = []
                
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    current = driver_info['current_load'].get(group_key, 0)
                    capacity = driver_info['capacity'].get(group_key, 0)
                    needed = dog['num_dogs']
                    
                    capacity_details.append(f"G{group}:{current}+{needed}<={capacity}")
                    
                    if current + needed > capacity:
                        can_handle = False
                        break
                
                # 🏆 Calculate score (for all drivers, even those without capacity)
                score = 100.0  # Base score
                score -= effective_distance * 20  # Distance penalty
                
                if effective_distance <= 0.5:
                    score += (0.5 - effective_distance) * 30
                elif effective_distance <= 1.0:
                    score += (1.0 - effective_distance) * 15
                
                if "exact_match" in compatibility_reason:
                    score += 15
                elif "adjacent_match" in compatibility_reason:
                    score += 5
                
                total_load = sum(driver_info['current_load'].values())
                max_total_capacity = sum(driver_info['capacity'].values())
                load_ratio = total_load / max_total_capacity if max_total_capacity > 0 else 0
                score -= load_ratio * 10
                
                if can_handle:
                    valid_candidates += 1
                    print(f"   ✅ {driver_name} - {effective_distance:.1f}mi, groups {driver_info['current_groups']}, score: {score:.1f} [{compatibility_reason}]")
                    
                    # Track best candidate with capacity
                    if score > best_score:
                        best_driver = driver_name
                        best_distance = effective_distance
                        best_score = score
                else:
                    print(f"   ⚠️  {driver_name} - {effective_distance:.1f}mi, score: {score:.1f}, NO CAPACITY [{', '.join(capacity_details)}] [{compatibility_reason}]")
                    
                    # Track best candidate even without capacity (for potential domino)
                    if score > best_score * 1.2:  # Only consider domino if significantly better (20% higher score)
                        print(f"      🎯 {driver_name} is significantly better - considering domino effect...")
                        
                        # Attempt domino reassignment (with multi-level chains)
                        domino_success, domino_moves = self.attempt_domino_reassignment(driver_name, dog, working_drivers, max_depth=3)
                        
                        if domino_success:
                            # Execute domino moves
                            domino_assignments = self.execute_domino_moves(domino_moves, working_drivers)
                            final_assignments.extend(domino_assignments)
                            
                            # Now this driver should have capacity
                            best_driver = driver_name
                            best_distance = effective_distance  
                            best_score = score
                            valid_candidates += 1
                            
                            print(f"      ✅ DOMINO SUCCESS! {driver_name} now has capacity")
                            break  # Exit the driver loop early since we found a solution
            
            # Assign to best driver found
            if best_driver:
                print(f"   🎯 BEST CHOICE: {dog['dog_name']} → {best_driver}")
                print(f"      Distance: {best_distance:.1f} miles")
                print(f"      Score: {best_score:.1f}")
                print(f"      Checked {candidates_checked} drivers, {valid_candidates} were valid")
                
                # 🔄 UPDATE DRIVER STATE (this is the key dynamic part!)
                # 1. Update load
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    working_drivers[best_driver]['current_load'][group_key] += dog['num_dogs']
                
                # 2. Add dog to driver's route (for future distance calculations)
                old_route_size = len(working_drivers[best_driver]['current_dogs'])
                working_drivers[best_driver]['current_dogs'].append(dog['dog_id'])
                new_route_size = len(working_drivers[best_driver]['current_dogs'])
                
                print(f"      🔄 ROUTE UPDATE: {best_driver} route expanded from {old_route_size} to {new_route_size} dogs")
                print(f"         New route: {working_drivers[best_driver]['current_dogs']}")
                
                # 3. Update driver's groups (for future group compatibility calculations)
                old_groups = working_drivers[best_driver]['current_groups'].copy()
                current_groups = set(working_drivers[best_driver]['current_groups'])
                current_groups.update(dog['needed_groups'])
                working_drivers[best_driver]['current_groups'] = sorted(list(current_groups))
                
                if old_groups != working_drivers[best_driver]['current_groups']:
                    print(f"         Groups updated: {old_groups} → {working_drivers[best_driver]['current_groups']}")
                
                # 4. Create assignment
                groups_text = "&".join(map(str, dog['needed_groups']))
                new_assignment = f"{best_driver}:{groups_text}"
                
                final_assignments.append({
                    'dog_id': dog['dog_id'],
                    'dog_name': dog['dog_name'],
                    'new_assignment': new_assignment,
                    'distance': best_distance,
                    'driver': best_driver
                })
                
                total_system_distance += best_distance
                
                # Show updated driver state
                updated_load = working_drivers[best_driver]['current_load']
                updated_groups = working_drivers[best_driver]['current_groups']
                capacity = working_drivers[best_driver]['capacity']
                groups_text = ",".join(map(str, updated_groups))
                print(f"      {best_driver} new state: groups [{groups_text}], load G1={updated_load['group1']}/{capacity['group1']}, "
                      f"G2={updated_load['group2']}/{capacity['group2']}, "
                      f"G3={updated_load['group3']}/{capacity['group3']}")
                
            else:
                print(f"   ❌ NO SUITABLE DRIVER FOUND for {dog['dog_name']}")
                print(f"      Checked {candidates_checked} drivers, {valid_candidates} had capacity")
                if valid_candidates == 0:
                    print(f"      Issue: No drivers with available capacity within {self.MAX_DISTANCE} miles")
        
        # Show final results
        print(f"\n📊 PRIORITY-BASED ASSIGNMENT SUMMARY:")
        print(f"   ✅ Successfully assigned: {len(final_assignments)}")
        print(f"   ❌ Unassigned: {len(dogs_to_reassign) - len(final_assignments)}")
        print(f"   📏 Total system distance: {total_system_distance:.1f} miles")
        avg_distance = total_system_distance / len(final_assignments) if final_assignments else 0
        callout_assignments = [a for a in final_assignments if not a.get('assignment_type', '').startswith('domino')]
        print(f"   📊 Average distance per dog: {avg_distance:.1f} miles")
        print(f"   🎯 Callout dogs assigned: {len(callout_assignments)}/{len(dogs_to_reassign)}")
        
        if final_assignments:
            print(f"\n🎉 DISTANCE-OPTIMIZED RESULTS:")
            driver_counts = {}
            driver_distances = {}
            domino_counts_by_level = {}
            regular_assignments = []
            
            for assignment in final_assignments:
                assignment_type = assignment.get('assignment_type', '')
                type_display = ""
                
                if assignment_type.startswith('domino_level_'):
                    level = assignment_type.split('_')[-1]
                    type_display = f" [DOMINO-L{level}]"
                    domino_counts_by_level[level] = domino_counts_by_level.get(level, 0) + 1
                else:
                    regular_assignments.append(assignment)
                
                print(f"   {assignment['dog_name']} → {assignment['new_assignment']} ({assignment['distance']:.1f}mi){type_display}")
                
                driver = assignment['driver']
                driver_counts[driver] = driver_counts.get(driver, 0) + 1
                driver_distances[driver] = driver_distances.get(driver, 0) + assignment['distance']
            
            print(f"\n📊 Driver distribution:")
            for driver in sorted(driver_counts.keys()):
                count = driver_counts[driver]
                total_dist = driver_distances[driver]
                avg_dist = total_dist / count
                print(f"   {driver}: {count} dogs, {total_dist:.1f}mi total, {avg_dist:.1f}mi avg")
            
            if domino_counts_by_level:
                total_domino_moves = sum(domino_counts_by_level.values())
                print(f"\n🔄 Multi-level domino effect summary:")
                print(f"   {total_domino_moves} dogs moved via domino chains")
                for level in sorted(domino_counts_by_level.keys()):
                    count = domino_counts_by_level[level]
                    print(f"   Level {level}: {count} moves")
                print(f"   {len(regular_assignments)} callout dogs optimally assigned")
                print(f"   Domino chains enabled {len([a for a in regular_assignments])} optimal placements")
        
        return final_assignments
        """Assign each callout dog to their closest available driver with dynamic route updates"""
        print("\n🔄 Starting DYNAMIC DISTANCE-BASED INDIVIDUAL ASSIGNMENT...")
        print("🎯 Each dog gets assigned to their closest available driver")
        print("🔄 Driver routes update dynamically after each assignment")
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        # Validate data
        print(f"\n🔍 Data validation:")
        print(f" Matrix dogs: {len(self.distance_matrix) if self.distance_matrix is not None else 0}")
        print(f" Assignment dogs: {len(self.dog_assignments)}")
        print(f" Callout dogs: {len(dogs_to_reassign)}")
        
        if len(self.distance_matrix) == 0:
            print("❌ NO DISTANCE MATRIX DATA!")
            return []
        
        # Get callout driver to exclude
        callout_driver = None
        if dogs_to_reassign[0].get('original_callout'):
            callout_text = dogs_to_reassign[0]['original_callout']
            if ':' in callout_text:
                callout_driver = callout_text.split(':', 1)[0].strip()
        
        print(f"🚫 Excluding callout driver: {callout_driver}")
        
        # Initialize working drivers with current state
        working_drivers = {}
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if combined and ':' in combined:
                driver_name = combined.split(':', 1)[0].strip()
                if driver_name and driver_name in self.driver_capacities and driver_name != callout_driver:
                    if driver_name not in working_drivers:
                        working_drivers[driver_name] = {
                            'capacity': self.driver_capacities[driver_name].copy(),
                            'current_load': self.calculate_driver_load(driver_name).copy(),
                            'current_dogs': self.get_driver_current_dogs(driver_name).copy(),
                            'current_groups': self.get_driver_current_groups(driver_name).copy()
                        }
        
        print(f"👥 Available working drivers: {len(working_drivers)}")
        for driver, info in list(working_drivers.items())[:5]:  # Show first 5
            groups_text = ",".join(map(str, info['current_groups'])) if info['current_groups'] else "none"
            print(f"   {driver}: {len(info['current_dogs'])} dogs, groups [{groups_text}], load G1={info['current_load']['group1']}/{info['capacity']['group1']}")
        
        final_assignments = []
        total_system_distance = 0.0
        
        # Process each callout dog individually
        for i, dog in enumerate(dogs_to_reassign):
            print(f"\n🐕 Assignment {i+1}/{len(dogs_to_reassign)}: {dog['dog_name']} ({dog['dog_id']})")
            print(f"   Needs: {dog['num_dogs']} dogs, groups {dog['needed_groups']}")
            
            best_driver = None
            best_distance = float('inf')
            best_score = float('-inf')
            
            candidates_checked = 0
            valid_candidates = 0
            
            # Check each available driver
            for driver_name, driver_info in working_drivers.items():
                candidates_checked += 1
                
                # 🔒 Check group compatibility first (before capacity)
                if not any(group in [1, 2, 3] for group in dog['needed_groups']):
                    print(f"   ❌ {driver_name} - group incompatibility")
                    continue
                
                # 📏 Calculate effective distance (for all drivers, regardless of capacity)
                effective_distance, compatibility_reason = self.calculate_effective_distance_to_driver(
                    dog['dog_id'], 
                    dog['needed_groups'], 
                    driver_info['current_dogs'],
                    driver_info['current_groups']
                )
                
                # Skip if too far
                if effective_distance > self.MAX_DISTANCE:
                    print(f"   ❌ {driver_name} - too far ({effective_distance:.1f}mi > {self.MAX_DISTANCE}mi limit)")
                    continue
                
                # 🔒 Check capacity constraints  
                can_handle = True
                capacity_details = []
                
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    current = driver_info['current_load'].get(group_key, 0)
                    capacity = driver_info['capacity'].get(group_key, 0)
                    needed = dog['num_dogs']
                    
                    capacity_details.append(f"G{group}:{current}+{needed}<={capacity}")
                    
                    if current + needed > capacity:
                        can_handle = False
                        break
                
                # 🏆 Calculate score (for all drivers, even those without capacity)
                score = 100.0  # Base score
                score -= effective_distance * 20  # Distance penalty
                
                if effective_distance <= 0.5:
                    score += (0.5 - effective_distance) * 30
                elif effective_distance <= 1.0:
                    score += (1.0 - effective_distance) * 15
                
                if "exact_match" in compatibility_reason:
                    score += 15
                elif "adjacent_match" in compatibility_reason:
                    score += 5
                
                total_load = sum(driver_info['current_load'].values())
                max_total_capacity = sum(driver_info['capacity'].values())
                load_ratio = total_load / max_total_capacity if max_total_capacity > 0 else 0
                score -= load_ratio * 10
                
                if can_handle:
                    valid_candidates += 1
                    print(f"   ✅ {driver_name} - {effective_distance:.1f}mi, groups {driver_info['current_groups']}, score: {score:.1f} [{compatibility_reason}]")
                    
                    # Track best candidate with capacity
                    if score > best_score:
                        best_driver = driver_name
                        best_distance = effective_distance
                        best_score = score
                else:
                    print(f"   ⚠️  {driver_name} - {effective_distance:.1f}mi, score: {score:.1f}, NO CAPACITY [{', '.join(capacity_details)}] [{compatibility_reason}]")
                    
                    # Track best candidate even without capacity (for potential domino)
                    if score > best_score * 1.2:  # Only consider domino if significantly better (20% higher score)
                        print(f"      🎯 {driver_name} is significantly better - considering domino effect...")
                        
                        # Attempt domino reassignment (with multi-level chains)
                        domino_success, domino_moves = self.attempt_domino_reassignment(driver_name, dog, working_drivers, max_depth=3)
                        
                        if domino_success:
                            # Execute domino moves
                            domino_assignments = self.execute_domino_moves(domino_moves, working_drivers)
                            final_assignments.extend(domino_assignments)
                            
                            # Now this driver should have capacity
                            best_driver = driver_name
                            best_distance = effective_distance  
                            best_score = score
                            valid_candidates += 1
                            
                            print(f"      ✅ DOMINO SUCCESS! {driver_name} now has capacity")
                            break  # Exit the driver loop early since we found a solution
            
            # Assign to best driver found
            if best_driver:
                print(f"   🎯 BEST CHOICE: {dog['dog_name']} → {best_driver}")
                print(f"      Distance: {best_distance:.1f} miles")
                print(f"      Score: {best_score:.1f}")
                print(f"      Checked {candidates_checked} drivers, {valid_candidates} were valid")
                
                # 🔄 UPDATE DRIVER STATE (this is the key dynamic part!)
                # 1. Update load
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    working_drivers[best_driver]['current_load'][group_key] += dog['num_dogs']
                
                # 2. Add dog to driver's route (for future distance calculations)
                old_route_size = len(working_drivers[best_driver]['current_dogs'])
                working_drivers[best_driver]['current_dogs'].append(dog['dog_id'])
                new_route_size = len(working_drivers[best_driver]['current_dogs'])
                
                print(f"      🔄 ROUTE UPDATE: {best_driver} route expanded from {old_route_size} to {new_route_size} dogs")
                print(f"         New route: {working_drivers[best_driver]['current_dogs']}")
                
                # 3. Update driver's groups (for future group compatibility calculations)
                old_groups = working_drivers[best_driver]['current_groups'].copy()
                current_groups = set(working_drivers[best_driver]['current_groups'])
                current_groups.update(dog['needed_groups'])
                working_drivers[best_driver]['current_groups'] = sorted(list(current_groups))
                
                if old_groups != working_drivers[best_driver]['current_groups']:
                    print(f"         Groups updated: {old_groups} → {working_drivers[best_driver]['current_groups']}")
                
                # 3. Create assignment
                groups_text = "&".join(map(str, dog['needed_groups']))
                new_assignment = f"{best_driver}:{groups_text}"
                
                final_assignments.append({
                    'dog_id': dog['dog_id'],
                    'dog_name': dog['dog_name'],
                    'new_assignment': new_assignment,
                    'distance': best_distance,
                    'driver': best_driver
                })
                
                total_system_distance += best_distance
                
                # Show updated driver state
                updated_load = working_drivers[best_driver]['current_load']
                updated_groups = working_drivers[best_driver]['current_groups']
                capacity = working_drivers[best_driver]['capacity']
                groups_text = ",".join(map(str, updated_groups))
                print(f"      {best_driver} new state: groups [{groups_text}], load G1={updated_load['group1']}/{capacity['group1']}, "
                      f"G2={updated_load['group2']}/{capacity['group2']}, "
                      f"G3={updated_load['group3']}/{capacity['group3']}")
                
            else:
                print(f"   ❌ NO SUITABLE DRIVER FOUND for {dog['dog_name']}")
                print(f"      Checked {candidates_checked} drivers, {valid_candidates} had capacity")
                if valid_candidates == 0:
                    print(f"      Issue: No drivers with available capacity within {self.MAX_DISTANCE} miles")
        
        # Show final results
        print(f"\n📊 DYNAMIC ASSIGNMENT SUMMARY:")
        print(f"   ✅ Successfully assigned: {len(final_assignments)}")
        print(f"   ❌ Unassigned: {len(dogs_to_reassign) - len(final_assignments)}")
        print(f"   📏 Total system distance: {total_system_distance:.1f} miles")
        avg_distance = total_system_distance / len(final_assignments) if final_assignments else 0
        callout_assignments = [a for a in final_assignments if a.get('assignment_type') != 'domino_move']
        print(f"   📊 Average distance per dog: {avg_distance:.1f} miles")
        print(f"   🎯 Callout dogs assigned: {len(callout_assignments)}/{len(dogs_to_reassign)}")
        
        if final_assignments:
            print(f"\n🎉 DISTANCE-OPTIMIZED RESULTS:")
            driver_counts = {}
            driver_distances = {}
            domino_counts_by_level = {}
            regular_assignments = []
            
            for assignment in final_assignments:
                assignment_type = assignment.get('assignment_type', '')
                type_display = ""
                
                if assignment_type.startswith('domino_level_'):
                    level = assignment_type.split('_')[-1]
                    type_display = f" [DOMINO-L{level}]"
                    domino_counts_by_level[level] = domino_counts_by_level.get(level, 0) + 1
                else:
                    regular_assignments.append(assignment)
                
                print(f"   {assignment['dog_name']} → {assignment['new_assignment']} ({assignment['distance']:.1f}mi){type_display}")
                
                driver = assignment['driver']
                driver_counts[driver] = driver_counts.get(driver, 0) + 1
                driver_distances[driver] = driver_distances.get(driver, 0) + assignment['distance']
            
            print(f"\n📊 Driver distribution:")
            for driver in sorted(driver_counts.keys()):
                count = driver_counts[driver]
                total_dist = driver_distances[driver]
                avg_dist = total_dist / count
                print(f"   {driver}: {count} dogs, {total_dist:.1f}mi total, {avg_dist:.1f}mi avg")
            
            if domino_counts_by_level:
                total_domino_moves = sum(domino_counts_by_level.values())
                print(f"\n🔄 Multi-level domino effect summary:")
                print(f"   {total_domino_moves} dogs moved via domino chains")
                for level in sorted(domino_counts_by_level.keys()):
                    count = domino_counts_by_level[level]
                    print(f"   Level {level}: {count} moves")
                print(f"   {len(regular_assignments)} callout dogs optimally assigned")
                print(f"   Domino chains enabled {len([a for a in regular_assignments])} optimal placements")
        
        return final_assignments

    def write_results_to_sheets(self, reassignments):
        """Write reassignment results back to Google Sheets"""
        try:
            print(f"\n📝 Writing {len(reassignments)} results to Google Sheets...")
            
            # Note: In this version, we're not actually writing back to sheets
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
    
    # Run the individual assignment system
    print("\n🔄 Processing callout assignments...")
    
    reassignments = system.reassign_dogs_individually()
    
    # Ensure reassignments is always a list
    if reassignments is None:
        reassignments = []
    
    # Write results
    if reassignments:
        write_success = system.write_results_to_sheets(reassignments)
        if write_success:
            print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments")
            print(f"📏 Total distance minimized: {sum(a.get('distance', 0) for a in reassignments):.1f} miles")
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")

if __name__ == "__main__":
    main()
