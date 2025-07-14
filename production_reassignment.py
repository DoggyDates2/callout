# production_reassignment.py
# Complete dog reassignment system with multi-strategy route optimization

import pandas as pd
import numpy as np
import requests
import json
import os
from typing import Dict, List, Tuple
import gspread
from google.oauth2.service_account import Credentials

class DogReassignmentSystem:
    def __init__(self):
        """Initialize the dog reassignment system"""
        # Google Sheets URLs (CSV export format)
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        
        # STRICTER DISTANCE LIMITS
        self.MAX_DISTANCE = 2.0  # Reduced from 3.0 - stricter limit
        self.EXACT_MATCH_MAX_DISTANCE = 1.5  # Even stricter for exact group matches
        self.ADJACENT_MATCH_MAX_DISTANCE = 0.75  # Very strict for adjacent matches
        
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

    def get_driver_current_groups(self, driver_name: str) -> List[int]:
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
        """Calculate effective distance considering group compatibility - STRICTER VERSION"""
        if not driver_dogs:
            return float('inf'), "no_dogs"
        
        # Find minimum physical distance to driver's route
        min_physical_distance = float('inf')
        closest_dog = None
        all_distances = []  # DEBUG: Track all distances
        
        for driver_dog_id in driver_dogs:
            distance = self.get_distance(callout_dog_id, driver_dog_id)
            all_distances.append(f"{driver_dog_id}:{distance:.1f}mi")
            if distance < min_physical_distance:
                min_physical_distance = distance
                closest_dog = driver_dog_id
        
        if min_physical_distance == float('inf'):
            return float('inf'), "no_distance_data"
        
        # 🔍 DEBUG: Show ALL distances for this assignment consideration
        print(f"      📏 Distances from {callout_dog_id}: {', '.join(all_distances)}")
        print(f"      📏 Closest: {closest_dog} ({min_physical_distance:.1f}mi)")
        
        # Determine group compatibility with STRICTER distance limits
        callout_groups_set = set(callout_needed_groups)
        driver_groups_set = set(driver_groups)
        
        # Check for exact group matches
        exact_matches = callout_groups_set.intersection(driver_groups_set)
        
        if exact_matches:
            # Exact group match - use actual distance but with strict limit
            if min_physical_distance <= self.EXACT_MATCH_MAX_DISTANCE:
                print(f"      ✅ EXACT MATCH accepted: {min_physical_distance:.1f}mi ≤ {self.EXACT_MATCH_MAX_DISTANCE}mi")
                return min_physical_distance, f"exact_match_{list(exact_matches)}"
            else:
                print(f"      ❌ EXACT MATCH rejected: {min_physical_distance:.1f}mi > {self.EXACT_MATCH_MAX_DISTANCE}mi")
                return float('inf'), f"exact_match_too_far_{min_physical_distance:.1f}mi"
        
        # Check for adjacent group matches with even stricter limits
        adjacent_matches = []
        for callout_group in callout_needed_groups:
            for driver_group in driver_groups:
                if abs(callout_group - driver_group) <= 2 and callout_group != driver_group:
                    adjacent_matches.append((callout_group, driver_group))
        
        if adjacent_matches:
            # Adjacent group match - double the distance and apply strict limit
            effective_distance = min_physical_distance * 2.0
            if effective_distance <= self.ADJACENT_MATCH_MAX_DISTANCE:
                print(f"      ✅ ADJACENT MATCH accepted: {effective_distance:.1f}mi ≤ {self.ADJACENT_MATCH_MAX_DISTANCE}mi")
                return effective_distance, f"adjacent_match_{adjacent_matches[0]}"
            else:
                print(f"      ❌ ADJACENT MATCH rejected: {effective_distance:.1f}mi > {self.ADJACENT_MATCH_MAX_DISTANCE}mi")
                return float('inf'), f"adjacent_match_too_far_{effective_distance:.1f}mi"
        
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
        """Strategy A: Individual assignment - DISTANCE PRIORITY VERSION"""
        print(f"\n🔹 STRATEGY A: Distance-Priority Individual Assignment")
        
        assignments = []
        drivers_copy = {k: {
            'capacity': v['capacity'].copy(),
            'current_load': v['current_load'].copy(),
            'current_dogs': v['current_dogs'].copy(),
            'current_groups': v['current_groups'].copy()
        } for k, v in working_drivers.items()}
        
        for dog in dogs_to_reassign:
            print(f"\n🐕 Processing {dog['dog_name']} ({dog['dog_id']}) - Groups: {dog['needed_groups']}")
            
            best_driver = None
            best_distance = float('inf')
            best_compatibility = ""
            candidates = []
            
            for driver_name, driver_info in drivers_copy.items():
                print(f"\n   🚗 Checking driver: {driver_name}")
                
                # Check capacity first
                can_handle = True
                for group in dog['needed_groups']:
                    group_key = f'group{group}'
                    current = driver_info['current_load'].get(group_key, 0)
                    capacity = driver_info['capacity'].get(group_key, 0)
                    
                    if current + dog['num_dogs'] > capacity:
                        print(f"      ❌ Capacity exceeded for group {group}: {current + dog['num_dogs']} > {capacity}")
                        can_handle = False
                        break
                
                if not can_handle:
                    continue
                
                # Calculate distance with detailed logging
                effective_distance, compatibility = self.calculate_effective_distance_to_driver(
                    dog['dog_id'], dog['needed_groups'], 
                    driver_info['current_dogs'], driver_info['current_groups']
                )
                
                print(f"      📊 Result: {effective_distance:.1f}mi ({compatibility})")
                
                if effective_distance == float('inf'):
                    continue
                
                candidates.append({
                    'driver': driver_name,
                    'distance': effective_distance,
                    'compatibility': compatibility
                })
                
                # DISTANCE FIRST: Pick the closest driver
                if effective_distance < best_distance:
                    best_driver = driver_name
                    best_distance = effective_distance
                    best_compatibility = compatibility
            
            print(f"\n   📋 All candidates for {dog['dog_name']}:")
            for candidate in sorted(candidates, key=lambda x: x['distance']):
                print(f"      - {candidate['driver']}: {candidate['distance']:.1f}mi ({candidate['compatibility']})")
            
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
                    'strategy': 'distance_priority'
                })
                
                print(f"   ✅ ASSIGNED: {dog['dog_name']} → {best_driver} ({best_distance:.1f}mi, {best_compatibility})")
            else:
                print(f"   ❌ NO VALID ASSIGNMENT for {dog['dog_name']}")
        
        total_miles = self.calculate_total_system_miles(assignments)
        print(f"      ✅ Distance-Priority Assignment: {len(assignments)} dogs assigned, {total_miles:.1f} total miles")
        
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

    def prioritize_callout_dogs(self, dogs_to_reassign: List[Dict]) -> List[Dict]:
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
        
        return prioritized_dogs

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
            strategies.append(('Distance-Priority Assignment', assignments_a, miles_a))
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
                
                # Show assignments for this driver
                assignment_list = []
                for name in dog_names:
                    for assignment in best_assignments:
                        if assignment['dog_name'] == name:
                            assignment_list.append(f"{name} → {assignment['new_assignment']}")
                            break
                print(f"      Assignments: {', '.join(assignment_list)}")
            
            print(f"\n📊 Route optimization summary:")
            print(f"   🎯 Total optimized miles: {total_calculated_miles:.1f}")
            print(f"   📈 Miles per driver: {total_calculated_miles/len(driver_dogs):.1f}" if driver_dogs else "   📈 No drivers")
            
            total_assignments = len(best_assignments)
            unassigned = len(dogs_to_reassign) - total_assignments
            if unassigned > 0:
                print(f"   ⚠️  {unassigned} dogs could not be assigned")
        
        return best_assignments

    def write_results_to_sheets(self, reassignments):
        """Write reassignment results back to Google Sheets"""
        try:
            print(f"\n📝 Writing {len(reassignments)} results to Google Sheets...")
            
            if not hasattr(self, 'sheets_client') or not self.sheets_client:
                print("❌ Google Sheets client not initialized")
                return False
            
            # Extract sheet ID from your existing MAP_SHEET_URL
            # "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
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
            print(f"📋 Sheet has {len(all_data)} rows and columns: {header_row[:10]}...")  # Show first 10 columns
            
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
                        "text": f"🐕 Dog Reassignment Complete: Updated {updates_count} assignments in Google Sheets"
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
    print("🚀 Production Dog Reassignment System")
    print("=" * 50)
    
    # Initialize system
    system = DogReassignmentSystem()
    
    # Setup Google Sheets client for WRITING (reading still uses CSV URLs)
    if not system.setup_google_sheets_client():
        print("❌ Failed to setup Google Sheets client for writing")
        return
    
    # Load all data (your existing code)
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
    
    # Run the multi-strategy assignment system
    print("\n🔄 Processing callout assignments...")
    
    reassignments = system.reassign_dogs_individually()
    
    # Ensure reassignments is always a list
    if reassignments is None:
        reassignments = []
    
    # Write results (NOW ACTUALLY WRITES!)
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
