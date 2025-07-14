# Global Network Reshuffling Optimizer
# Strategic reallocation of entire driver network to optimize for callout dogs

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

class GlobalReshufflingOptimizer:
    def __init__(self):
        """Initialize the global reshuffling optimizer"""
        # Google Sheets URLs (CSV export format)
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        
        # DISTANCE LIMITS - PRACTICAL THRESHOLDS
        self.PREFERRED_DISTANCE = 0.5  # Ideal assignments: under 0.5 miles
        self.ACCEPTABLE_DISTANCE = 0.8  # Acceptable for reshuffling: under 0.8 miles
        self.MAX_DISTANCE = 1.0  # Backup assignments: 0.5-1.0 miles
        self.ABSOLUTE_MAX_DISTANCE = 3.3  # Hard limit: emergency only
        self.RESHUFFLING_TOLERANCE = 0.3  # Allow existing dogs to move up to 0.3mi farther for global benefit
        self.EXCLUSION_DISTANCE = 100.0  # Recognize 100+ as "do not assign" placeholders
        
        # Data containers
        self.distance_matrix = None
        self.dog_assignments = None
        self.driver_capacities = None
        self.sheets_client = None

    def setup_google_sheets_client(self):
        """Setup Google Sheets API client using service account credentials"""
        try:
            service_account_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            if not service_account_json:
                print("❌ GOOGLE_SERVICE_ACCOUNT_JSON environment variable not found")
                return False
            
            credentials_dict = json.loads(service_account_json)
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
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
            response = requests.get(self.DISTANCE_MATRIX_URL)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), index_col=0)
            
            dog_ids = [col for col in df.columns if 'x' in str(col).lower()]
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
            response = requests.get(self.MAP_SHEET_URL)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            assignments = []
            
            for _, row in df.iterrows():
                try:
                    dog_name = row.iloc[1] if len(row) > 1 else ""
                    combined = row.iloc[7] if len(row) > 7 else ""
                    group = row.iloc[8] if len(row) > 8 else ""
                    dog_id = row.iloc[9] if len(row) > 9 else ""
                    callout = row.iloc[10] if len(row) > 10 else ""
                    num_dogs = row.iloc[5] if len(row) > 5 else 1
                    
                    if not dog_id or pd.isna(dog_id):
                        continue
                    
                    try:
                        num_dogs = int(float(num_dogs)) if not pd.isna(num_dogs) else 1
                        num_dogs = max(1, num_dogs)
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
            print("👥 Loading driver capacities...")
            response = requests.get(self.MAP_SHEET_URL)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            capacities = {}
            
            for _, row in df.iterrows():
                try:
                    driver_name = row.iloc[17] if len(row) > 17 else ""
                    group1_cap = row.iloc[20] if len(row) > 20 else 0
                    group2_cap = row.iloc[21] if len(row) > 21 else 0
                    group3_cap = row.iloc[22] if len(row) > 22 else 0
                    
                    if not driver_name or pd.isna(driver_name) or driver_name == "":
                        continue
                    
                    try:
                        group1_cap = int(float(group1_cap)) if not pd.isna(group1_cap) else 0
                        group2_cap = int(float(group2_cap)) if not pd.isna(group2_cap) else 0
                        group3_cap = int(float(group3_cap)) if not pd.isna(group3_cap) else 0
                    except:
                        continue
                    
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
            if (not assignment['combined'] or assignment['combined'].strip() == "") and \
               (assignment['callout'] and assignment['callout'].strip() != ""):
                
                dog_name = str(assignment['dog_name']).lower().strip()
                if any(keyword in dog_name for keyword in ['parking', 'field', 'admin', 'office']):
                    continue
                
                callout_text = assignment['callout'].strip()
                if ':' not in callout_text:
                    continue
                
                original_driver = callout_text.split(':', 1)[0].strip()
                full_assignment_string = callout_text.split(':', 1)[1].strip()
                needed_groups = self._extract_groups_for_capacity_check(full_assignment_string)
                
                if needed_groups:
                    dogs_to_reassign.append({
                        'dog_id': assignment['dog_id'],
                        'dog_name': assignment['dog_name'],
                        'num_dogs': assignment['num_dogs'],
                        'needed_groups': needed_groups,
                        'full_assignment_string': full_assignment_string,
                        'original_callout': assignment['callout'],
                        'original_driver': original_driver
                    })
        
        print(f"🚨 Found {len(dogs_to_reassign)} dogs that need drivers assigned")
        return dogs_to_reassign

    def _extract_groups_for_capacity_check(self, assignment_string):
        """Extract group numbers for capacity checking"""
        try:
            group_digits = re.findall(r'[123]', assignment_string)
            groups = sorted(list(set(int(digit) for digit in group_digits)))
            return groups
        except Exception as e:
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

    def build_current_network_state(self):
        """Build a complete picture of the current driver network"""
        print("\n🕸️ BUILDING CURRENT NETWORK STATE...")
        
        network = {}
        
        # Initialize all drivers
        for driver_name in self.driver_capacities.keys():
            network[driver_name] = {
                'capacity': self.driver_capacities[driver_name].copy(),
                'current_dogs': [],
                'current_load': {'group1': 0, 'group2': 0, 'group3': 0}
            }
        
        # Add current assignments
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            
            if not combined or combined.strip() == "" or ':' not in combined:
                continue
            
            driver = combined.split(':', 1)[0].strip()
            assignment_string = combined.split(':', 1)[1].strip()
            groups = self._extract_groups_for_capacity_check(assignment_string)
            
            if driver in network:
                dog_record = {
                    'dog_id': assignment['dog_id'],
                    'dog_name': assignment['dog_name'],
                    'needed_groups': groups,
                    'num_dogs': assignment['num_dogs'],
                    'assignment_string': assignment_string
                }
                
                network[driver]['current_dogs'].append(dog_record)
                
                # Update load
                for group in groups:
                    group_key = f'group{group}'
                    if group_key in network[driver]['current_load']:
                        network[driver]['current_load'][group_key] += assignment['num_dogs']
        
        # Calculate available capacity
        for driver_name, data in network.items():
            data['available_capacity'] = {}
            for group in [1, 2, 3]:
                group_key = f'group{group}'
                total_cap = data['capacity'].get(group_key, 0)
                current_load = data['current_load'].get(group_key, 0)
                data['available_capacity'][group_key] = max(0, total_cap - current_load)
        
        print(f"📊 Network state: {len(network)} drivers with current assignments")
        return network

    def identify_high_value_drivers(self, callout_dogs, network_state):
        """Identify drivers who are close to multiple callout dogs but lack capacity"""
        print("\n🎯 IDENTIFYING HIGH-VALUE DRIVERS FOR RESHUFFLING...")
        
        high_value_drivers = []
        
        for driver_name, driver_data in network_state.items():
            if not driver_data['current_dogs']:  # Skip drivers with no current dogs
                continue
            
            # Calculate how many callout dogs this driver could theoretically serve
            close_callouts = []
            
            for callout_dog in callout_dogs:
                # Find minimum distance to any of this driver's current dogs
                min_distance = float('inf')
                closest_via_dog = None
                
                for current_dog in driver_data['current_dogs']:
                    distance = self.get_distance(callout_dog['dog_id'], current_dog['dog_id'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_via_dog = current_dog['dog_name']
                
                # Check if this driver has the right group capabilities
                can_serve_groups = True
                for group in callout_dog['needed_groups']:
                    group_key = f'group{group}'
                    if driver_data['capacity'].get(group_key, 0) == 0:
                        can_serve_groups = False
                        break
                
                if min_distance <= self.ACCEPTABLE_DISTANCE and can_serve_groups:
                    close_callouts.append({
                        'callout_dog': callout_dog,
                        'distance': min_distance,
                        'via_dog': closest_via_dog,
                        'capacity_needed': callout_dog['num_dogs']
                    })
            
            # Check current capacity constraints
            capacity_shortfall = {}
            for close_callout in close_callouts:
                for group in close_callout['callout_dog']['needed_groups']:
                    group_key = f'group{group}'
                    needed = close_callout['capacity_needed']
                    available = driver_data['available_capacity'].get(group_key, 0)
                    
                    if group_key not in capacity_shortfall:
                        capacity_shortfall[group_key] = 0
                    
                    if needed > available:
                        capacity_shortfall[group_key] += needed - available
            
            # This is a high-value driver if they're close to callouts but lack capacity
            if close_callouts and any(shortfall > 0 for shortfall in capacity_shortfall.values()):
                high_value_drivers.append({
                    'driver_name': driver_name,
                    'close_callouts': close_callouts,
                    'capacity_shortfall': capacity_shortfall,
                    'current_dogs': driver_data['current_dogs'].copy(),
                    'value_score': len(close_callouts) * 100 - sum(c['distance'] for c in close_callouts)
                })
        
        # Sort by value score (more close callouts + shorter distances = higher value)
        high_value_drivers.sort(key=lambda x: x['value_score'], reverse=True)
        
        print(f"🎯 Found {len(high_value_drivers)} high-value drivers for reshuffling:")
        for i, driver_info in enumerate(high_value_drivers[:5]):  # Show top 5
            driver_name = driver_info['driver_name']
            close_count = len(driver_info['close_callouts'])
            avg_distance = np.mean([c['distance'] for c in driver_info['close_callouts']])
            
            print(f"   {i+1}. {driver_name}: {close_count} close callouts, avg {avg_distance:.1f}mi")
            
            for callout_info in driver_info['close_callouts'][:2]:  # Show top 2 callouts
                callout_name = callout_info['callout_dog']['dog_name']
                distance = callout_info['distance']
                via_dog = callout_info['via_dog']
                print(f"      - {callout_name} ({distance:.1f}mi via {via_dog})")
        
        return high_value_drivers

    def find_reshuffling_opportunities(self, high_value_drivers, network_state):
        """Find opportunities to move current dogs to create capacity for callouts"""
        print("\n🔄 ANALYZING RESHUFFLING OPPORTUNITIES...")
        
        reshuffling_scenarios = []
        
        for driver_info in high_value_drivers[:3]:  # Focus on top 3 high-value drivers
            driver_name = driver_info['driver_name']
            print(f"\n🔍 Analyzing {driver_name} for reshuffling opportunities...")
            
            # Find moveable dogs (single group, single dog assignments preferred)
            moveable_dogs = []
            
            for current_dog in driver_info['current_dogs']:
                # Prefer simple assignments for moving (easier to relocate)
                complexity_score = len(current_dog['needed_groups']) + (current_dog['num_dogs'] - 1)
                
                moveable_dogs.append({
                    'dog': current_dog,
                    'complexity': complexity_score,
                    'groups_freed': current_dog['needed_groups'],
                    'capacity_freed': current_dog['num_dogs']
                })
            
            # Sort by complexity (simpler dogs are easier to move)
            moveable_dogs.sort(key=lambda x: x['complexity'])
            
            print(f"   📋 {driver_name} has {len(moveable_dogs)} potentially moveable dogs:")
            for j, moveable in enumerate(moveable_dogs[:3]):  # Show top 3
                dog = moveable['dog']
                print(f"      {j+1}. {dog['dog_name']} - Groups:{dog['needed_groups']}, Dogs:{dog['num_dogs']}")
            
            # For each moveable dog, find alternative drivers
            for moveable_dog_info in moveable_dogs[:5]:  # Try top 5 moveable dogs
                moveable_dog = moveable_dog_info['dog']
                
                alternative_placements = []
                
                for alt_driver_name, alt_driver_data in network_state.items():
                    if alt_driver_name == driver_name:  # Skip same driver
                        continue
                    
                    # Find minimum distance to alternative driver's dogs
                    min_distance = float('inf')
                    closest_alt_dog = None
                    
                    for alt_dog in alt_driver_data['current_dogs']:
                        distance = self.get_distance(moveable_dog['dog_id'], alt_dog['dog_id'])
                        if distance < min_distance:
                            min_distance = distance
                            closest_alt_dog = alt_dog['dog_name']
                    
                    # Check if alternative driver has capacity and right groups
                    can_accept = True
                    for group in moveable_dog['needed_groups']:
                        group_key = f'group{group}'
                        available = alt_driver_data['available_capacity'].get(group_key, 0)
                        needed = moveable_dog['num_dogs']
                        
                        if available < needed:
                            can_accept = False
                            break
                    
                    # Only consider moves within acceptable distance increase
                    original_distance = self.get_distance(moveable_dog['dog_id'], 
                                                        driver_info['current_dogs'][0]['dog_id'])  # Approx current
                    distance_increase = min_distance - original_distance
                    
                    if can_accept and min_distance <= self.ACCEPTABLE_DISTANCE and distance_increase <= self.RESHUFFLING_TOLERANCE:
                        alternative_placements.append({
                            'alt_driver': alt_driver_name,
                            'distance': min_distance,
                            'distance_increase': distance_increase,
                            'via_dog': closest_alt_dog
                        })
                
                if alternative_placements:
                    # Sort by distance increase (prefer minimal disruption)
                    alternative_placements.sort(key=lambda x: x['distance_increase'])
                    best_alt = alternative_placements[0]
                    
                    # Calculate benefit: what callouts could we now serve?
                    capacity_freed = {}
                    for group in moveable_dog['needed_groups']:
                        group_key = f'group{group}'
                        capacity_freed[group_key] = moveable_dog['num_dogs']
                    
                    potential_callouts_served = []
                    for callout_info in driver_info['close_callouts']:
                        callout_dog = callout_info['callout_dog']
                        can_now_serve = True
                        
                        for group in callout_dog['needed_groups']:
                            group_key = f'group{group}'
                            if capacity_freed.get(group_key, 0) < callout_dog['num_dogs']:
                                can_now_serve = False
                                break
                        
                        if can_now_serve:
                            potential_callouts_served.append(callout_info)
                    
                    if potential_callouts_served:  # Only consider if it helps callouts
                        reshuffling_scenarios.append({
                            'source_driver': driver_name,
                            'target_driver': best_alt['alt_driver'],
                            'dog_to_move': moveable_dog,
                            'move_distance': best_alt['distance'],
                            'distance_increase': best_alt['distance_increase'],
                            'via_dog': best_alt['via_dog'],
                            'capacity_freed': capacity_freed,
                            'callouts_enabled': potential_callouts_served,
                            'benefit_score': len(potential_callouts_served) * 1000 - best_alt['distance_increase'] * 100
                        })
        
        # Sort scenarios by benefit score
        reshuffling_scenarios.sort(key=lambda x: x['benefit_score'], reverse=True)
        
        print(f"\n🎯 Found {len(reshuffling_scenarios)} viable reshuffling scenarios:")
        for i, scenario in enumerate(reshuffling_scenarios[:3]):  # Show top 3
            dog_name = scenario['dog_to_move']['dog_name']
            source = scenario['source_driver']
            target = scenario['target_driver']
            distance = scenario['move_distance']
            increase = scenario['distance_increase']
            callouts_count = len(scenario['callouts_enabled'])
            
            print(f"   {i+1}. Move {dog_name}: {source} → {target} (+{increase:.1f}mi to {distance:.1f}mi)")
            print(f"      Enables {callouts_count} callout assignments in {source}")
            
            for callout_info in scenario['callouts_enabled'][:2]:
                callout_name = callout_info['callout_dog']['dog_name']
                callout_distance = callout_info['distance']
                print(f"        - {callout_name} ({callout_distance:.1f}mi)")
        
        return reshuffling_scenarios

    def execute_global_reshuffling(self, callout_dogs):
        """Execute the global network reshuffling optimization"""
        print("\n🌐 STARTING GLOBAL NETWORK RESHUFFLING OPTIMIZATION...")
        print("🎯 Goal: Strategically reallocate network to maximize good callout assignments")
        print("🔄 Method: Move existing dogs to create capacity where callouts need it most")
        print("=" * 80)
        
        # Build current network state
        network_state = self.build_current_network_state()
        
        # Identify high-value drivers (close to callouts but lack capacity)
        high_value_drivers = self.identify_high_value_drivers(callout_dogs, network_state)
        
        if not high_value_drivers:
            print("❌ No high-value drivers found for reshuffling")
            return [], []
        
        # Find reshuffling opportunities
        reshuffling_scenarios = self.find_reshuffling_opportunities(high_value_drivers, network_state)
        
        if not reshuffling_scenarios:
            print("❌ No viable reshuffling scenarios found")
            return [], []
        
        # Execute the best reshuffling moves
        print(f"\n🚀 EXECUTING NETWORK RESHUFFLING...")
        
        executed_moves = []
        updated_network = deepcopy(network_state)
        
        # Execute top 3 reshuffling scenarios (to avoid over-complicating)
        for scenario in reshuffling_scenarios[:3]:
            source_driver = scenario['source_driver']
            target_driver = scenario['target_driver']
            dog_to_move = scenario['dog_to_move']
            
            print(f"\n🔄 Moving {dog_to_move['dog_name']}: {source_driver} → {target_driver}")
            print(f"   📏 Distance: {scenario['move_distance']:.1f}mi (+{scenario['distance_increase']:.1f}mi)")
            print(f"   🎯 Enables {len(scenario['callouts_enabled'])} callout assignments")
            
            # Update network state
            # Remove from source driver
            updated_network[source_driver]['current_dogs'] = [
                dog for dog in updated_network[source_driver]['current_dogs']
                if dog['dog_id'] != dog_to_move['dog_id']
            ]
            
            # Update source driver load and available capacity
            for group in dog_to_move['needed_groups']:
                group_key = f'group{group}'
                updated_network[source_driver]['current_load'][group_key] -= dog_to_move['num_dogs']
                updated_network[source_driver]['available_capacity'][group_key] += dog_to_move['num_dogs']
            
            # Add to target driver
            updated_network[target_driver]['current_dogs'].append(dog_to_move)
            
            # Update target driver load and available capacity
            for group in dog_to_move['needed_groups']:
                group_key = f'group{group}'
                updated_network[target_driver]['current_load'][group_key] += dog_to_move['num_dogs']
                updated_network[target_driver]['available_capacity'][group_key] -= dog_to_move['num_dogs']
            
            # Record the move
            executed_moves.append({
                'dog_name': dog_to_move['dog_name'],
                'dog_id': dog_to_move['dog_id'],
                'from_driver': source_driver,
                'to_driver': target_driver,
                'distance': scenario['move_distance'],
                'distance_increase': scenario['distance_increase'],
                'reason': f"create_capacity_for_{len(scenario['callouts_enabled'])}_callouts",
                'callouts_enabled': [c['callout_dog']['dog_name'] for c in scenario['callouts_enabled']]
            })
        
        print(f"\n✅ Executed {len(executed_moves)} strategic moves")
        
        # Now assign callouts to the freed capacity
        print(f"\n🎯 ASSIGNING CALLOUTS TO FREED CAPACITY...")
        
        callout_assignments = []
        
        for callout_dog in callout_dogs:
            best_assignment = None
            best_distance = float('inf')
            
            # Check all drivers for capacity (prioritizing those we just freed up)
            for driver_name, driver_data in updated_network.items():
                # Check if driver has capacity for this callout
                has_capacity = True
                for group in callout_dog['needed_groups']:
                    group_key = f'group{group}'
                    available = driver_data['available_capacity'].get(group_key, 0)
                    needed = callout_dog['num_dogs']
                    
                    if available < needed:
                        has_capacity = False
                        break
                
                if not has_capacity:
                    continue
                
                # Find distance to closest dog of this driver
                min_distance = float('inf')
                closest_dog = None
                
                for current_dog in driver_data['current_dogs']:
                    distance = self.get_distance(callout_dog['dog_id'], current_dog['dog_id'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_dog = current_dog['dog_name']
                
                if min_distance < best_distance and min_distance <= self.ABSOLUTE_MAX_DISTANCE:
                    best_assignment = {
                        'driver': driver_name,
                        'distance': min_distance,
                        'via_dog': closest_dog
                    }
                    best_distance = min_distance
            
            if best_assignment:
                driver = best_assignment['driver']
                distance = best_assignment['distance']
                
                # Create assignment
                new_assignment = f"{driver}:{callout_dog['full_assignment_string']}"
                
                # Determine quality
                if distance <= self.PREFERRED_DISTANCE:
                    quality = "GOOD"
                elif distance <= self.MAX_DISTANCE:
                    quality = "BACKUP"
                else:
                    quality = "EMERGENCY"
                
                callout_assignments.append({
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'new_assignment': new_assignment,
                    'driver': driver,
                    'distance': distance,
                    'quality': quality,
                    'via_dog': best_assignment['via_dog']
                })
                
                # Update network state for next assignment
                for group in callout_dog['needed_groups']:
                    group_key = f'group{group}'
                    updated_network[driver]['available_capacity'][group_key] -= callout_dog['num_dogs']
                
                print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.1f}mi, {quality})")
            else:
                print(f"   ❌ {callout_dog['dog_name']} - still no viable assignment")
        
        return callout_assignments, executed_moves

    def write_results_to_sheets(self, callout_assignments, reshuffling_moves):
        """Write both callout assignments and reshuffling moves to Google Sheets"""
        try:
            print(f"\n📝 Writing {len(callout_assignments)} callout assignments and {len(reshuffling_moves)} reshuffling moves...")
            
            if not hasattr(self, 'sheets_client') or not self.sheets_client:
                print("❌ Google Sheets client not initialized")
                return False
            
            sheet_id = "1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0"
            spreadsheet = self.sheets_client.open_by_key(sheet_id)
            
            # Find worksheet
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
                        break
                    except:
                        continue
            
            if not worksheet:
                print("❌ Could not find worksheet")
                return False
            
            # Get all data
            all_data = worksheet.get_all_values()
            header_row = all_data[0]
            
            # Find Dog ID column
            dog_id_col = None
            for i, header in enumerate(header_row):
                if 'dog id' in str(header).lower().strip():
                    dog_id_col = i
                    break
            
            if dog_id_col is None:
                print("❌ Could not find Dog ID column")
                return False
            
            target_col = 7  # Column H (Combined)
            updates = []
            
            # Process callout assignments
            print(f"\n🔍 Processing {len(callout_assignments)} callout assignments...")
            for assignment in callout_assignments:
                dog_id = str(assignment.get('dog_id', '')).strip()
                new_assignment = str(assignment.get('new_assignment', '')).strip()
                
                for row_idx in range(1, len(all_data)):
                    if dog_id_col < len(all_data[row_idx]):
                        current_dog_id = str(all_data[row_idx][dog_id_col]).strip()
                        
                        if current_dog_id == dog_id:
                            cell_address = gspread.utils.rowcol_to_a1(row_idx + 1, target_col + 1)
                            updates.append({'range': cell_address, 'values': [[new_assignment]]})
                            print(f"  ✅ {dog_id} → {new_assignment}")
                            break
            
            # Process reshuffling moves
            print(f"\n🔍 Processing {len(reshuffling_moves)} reshuffling moves...")
            for move in reshuffling_moves:
                dog_id = str(move.get('dog_id', '')).strip()
                to_driver = move.get('to_driver', '')
                
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
                                updates.append({'range': cell_address, 'values': [[new_combined]]})
                                print(f"  🔄 {dog_id} moved: {move['from_driver']} → {to_driver}")
                            break
            
            if updates:
                print(f"\n📤 Writing {len(updates)} updates to Google Sheets...")
                worksheet.batch_update(updates)
                print("✅ Successfully updated all assignments!")
                return True
            else:
                print("❌ No valid updates to make")
                return False
                
        except Exception as e:
            print(f"❌ Error writing to sheets: {e}")
            return False

def main():
    """Main function to run the global reshuffling optimizer"""
    print("🌐 Global Network Reshuffling Optimizer")
    print("🎯 Strategic reallocation of entire driver network for optimal callout assignments")
    print("🔄 Moves existing dogs to create capacity where callouts need it most")
    print("📊 Maximizes good assignments (≤0.5mi) through network-wide optimization")
    print("=" * 80)
    
    optimizer = GlobalReshufflingOptimizer()
    
    # Setup and load data
    if not optimizer.setup_google_sheets_client():
        return
    
    print("\n⬇️ Loading data from Google Sheets...")
    if not all([
        optimizer.load_distance_matrix(),
        optimizer.load_dog_assignments(),
        optimizer.load_driver_capacities()
    ]):
        print("❌ Failed to load required data")
        return
    
    # Get callout dogs
    callout_dogs = optimizer.get_dogs_to_reassign()
    if not callout_dogs:
        print("✅ No callouts detected - all dogs have drivers assigned!")
        return
    
    # Execute global reshuffling optimization
    callout_assignments, reshuffling_moves = optimizer.execute_global_reshuffling(callout_dogs)
    
    # Show results summary
    print(f"\n🏆 GLOBAL RESHUFFLING RESULTS:")
    if reshuffling_moves:
        print(f"   🔄 Strategic moves: {len(reshuffling_moves)}")
        for move in reshuffling_moves:
            print(f"      - {move['dog_name']}: {move['from_driver']} → {move['to_driver']} (+{move['distance_increase']:.1f}mi)")
            print(f"        Enabled: {', '.join(move['callouts_enabled'])}")
    
    if callout_assignments:
        good_count = len([a for a in callout_assignments if a.get('quality') == 'GOOD'])
        backup_count = len([a for a in callout_assignments if a.get('quality') == 'BACKUP'])
        emergency_count = len([a for a in callout_assignments if a.get('quality') == 'EMERGENCY'])
        
        print(f"   📊 Callout assignments: {len(callout_assignments)}/{len(callout_dogs)}")
        print(f"   💚 GOOD (≤0.5mi): {good_count}")
        print(f"   🟡 BACKUP (0.5-1.0mi): {backup_count}")
        print(f"   🚨 EMERGENCY (>1.0mi): {emergency_count}")
        
        improvement_msg = f"   🎯 Network reshuffling "
        if good_count > len(callout_dogs) * 0.6:
            improvement_msg += "achieved excellent results!"
        elif good_count > len(callout_dogs) * 0.4:
            improvement_msg += "achieved good results!"
        else:
            improvement_msg += "provided some improvement, but more drivers may be needed."
        
        print(improvement_msg)
        
        # Write results
        if optimizer.write_results_to_sheets(callout_assignments, reshuffling_moves):
            print(f"\n🎉 SUCCESS! Global network reshuffling complete.")
        else:
            print(f"\n❌ Failed to write results to Google Sheets")
    else:
        print("❌ No successful callout assignments after reshuffling")

if __name__ == "__main__":
    main()
