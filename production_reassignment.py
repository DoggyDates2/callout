# production_reassignment_smart.py
"""
Smart radius-based assignment with intelligent cascading
Always moves the OUTLIER dog to maintain route coherence
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, List, Optional, Tuple
from io import StringIO
import re


class SmartDogReassignment:
    """Intelligent reassignment focusing on proximity and outlier detection"""
    
    def __init__(self):
        # Published Google Sheets URLs - NO AUTH NEEDED FOR READING!
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSh-t6fIfgsli9D79KZeXM_-V5fO3zam6T_Bcp94d-IoRucxWusl6vbtT-WSaqFimHw7ABd76YGcKGV/pub?gid=988585160&single=true&output=csv"
        
        # MAP SHEET - Now published too! 
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT_9o2CJOgNpwtLPaqZkYBYVHtNJo5D-0qfeRqtdW-9yYV9cp5TMOvI5YTR8Xp3GcGhOU25mGBTHEdF/pub?gid=267803750&single=true&output=csv"
        
        # For writing results (always needs auth even if sheet is public)
        self.MAP_SHEET_ID = "1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0"
        
        # Configuration
        self.MAX_SEARCH_RADIUS = 3.5  # miles
        self.ADJACENT_GROUP_PENALTY = 1.5  # Adjacent groups get 1.5x distance penalty
        
        # Data storage
        self.distance_matrix = None
        self.current_state = []  # Live state that updates after each assignment
        self.driver_capacities = {}
        self.called_out_drivers = set()
        self.assignments_made = []
        self.cascading_moves = []
    
    def load_data(self):
        """Load all required data with robust error handling"""
        print("📊 Loading data...")
        
        # Load distance matrix
        try:
            response = requests.get(self.DISTANCE_MATRIX_URL, timeout=30)
            response.raise_for_status()
            
            # Check if we got CSV or HTML (common error)
            if 'html' in response.text[:100].lower():
                print("❌ Got HTML instead of CSV - check if sheet is published correctly")
                return False
            
            df = pd.read_csv(StringIO(response.text), index_col=0)
            dog_ids = [col for col in df.columns if 'x' in str(col).lower()]
            
            if not dog_ids:
                print("❌ No dog IDs found in distance matrix")
                return False
                
            self.distance_matrix = df.loc[df.index.isin(dog_ids), dog_ids]
            print(f"✅ Distance matrix: {self.distance_matrix.shape}")
            
        except requests.RequestException as e:
            print(f"❌ Network error loading distance matrix: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to parse distance matrix: {e}")
            return False
        
        # Load map data
        try:
            response = requests.get(self.MAP_SHEET_URL, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            print(f"📋 Map sheet has {len(df)} rows, {len(df.columns)} columns")
            
            # Build current state from assignments
            for idx, row in df.iterrows():
                try:
                    # Safely access columns
                    if len(row) < 11:  # Need at least 11 columns
                        continue
                    
                    dog_id = str(row.iloc[9]) if not pd.isna(row.iloc[9]) else ""
                    if not dog_id or 'x' not in dog_id.lower():
                        continue
                    
                    dog_name = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else "Unknown"
                    combined = str(row.iloc[7]) if not pd.isna(row.iloc[7]) else ""
                    callout = str(row.iloc[10]) if not pd.isna(row.iloc[10]) else ""
                    
                    # Safe conversion to int
                    try:
                        num_dogs = int(float(str(row.iloc[5]))) if not pd.isna(row.iloc[5]) else 1
                        num_dogs = max(1, num_dogs)  # Ensure at least 1
                    except:
                        num_dogs = 1
                
                    # If has assignment, add to current state
                    if combined and ':' in combined:
                        parts = combined.split(':', 1)
                        driver = parts[0].strip()
                        groups = self._extract_groups(parts[1])
                        
                        if groups and driver:  # Valid assignment
                            self.current_state.append({
                                'dog_id': dog_id,
                                'dog_name': dog_name,
                                'driver': driver,
                                'groups': groups,
                                'num_dogs': num_dogs,
                                'is_callout': False
                            })
                    
                    # If callout, mark for reassignment
                    elif callout and ':' in callout:
                        # Skip non-dog entries
                        if any(skip in dog_name.lower() for skip in ['parking', 'field', 'admin', 'office']):
                            continue
                            
                        parts = callout.split(':', 1)
                        original_driver = parts[0].strip()
                        groups = self._extract_groups(parts[1])
                        
                        if groups and original_driver:  # Valid callout
                            self.current_state.append({
                                'dog_id': dog_id,
                                'dog_name': dog_name,
                                'driver': None,  # Needs assignment
                                'groups': groups,
                                'num_dogs': num_dogs,
                                'is_callout': True,
                                'original': callout,
                                'assignment_string': parts[1].strip()
                            })
                            self.called_out_drivers.add(original_driver)
                
                except Exception as e:
                    # Skip problematic rows
                    continue
            
            # Load driver capacities with error handling
            self.driver_capacities = {}
            for idx, row in df.iterrows():
                try:
                    if len(row) < 23:  # Need at least 23 columns for capacities
                        continue
                        
                    driver = str(row.iloc[17]) if not pd.isna(row.iloc[17]) else ""
                    if not driver:
                        continue
                    
                    # Safe conversion to int
                    try:
                        g1 = int(float(str(row.iloc[20]))) if not pd.isna(row.iloc[20]) else 0
                        g2 = int(float(str(row.iloc[21]))) if not pd.isna(row.iloc[21]) else 0
                        g3 = int(float(str(row.iloc[22]))) if not pd.isna(row.iloc[22]) else 0
                    except:
                        g1 = g2 = g3 = 0
                    
                    if g1 > 0 or g2 > 0 or g3 > 0:
                        self.driver_capacities[driver] = {
                            'group1': max(0, g1),  # Ensure non-negative
                            'group2': max(0, g2),
                            'group3': max(0, g3)
                        }
                except:
                    continue
            
            callout_count = sum(1 for d in self.current_state if d['is_callout'])
            print(f"✅ Loaded {len(self.current_state)} dogs ({callout_count} callouts)")
            print(f"✅ {len(self.driver_capacities)} drivers with capacity")
            if self.called_out_drivers:
                print(f"🚫 Called out: {', '.join(sorted(self.called_out_drivers))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load map data: {e}")
            return False
    
    def _extract_groups(self, text):
        """Extract group numbers from assignment string"""
        return sorted(list(set(int(d) for d in re.findall(r'[123]', text))))
    
    def _get_distance(self, dog1_id, dog2_id):
        """Get distance between two dogs"""
        try:
            if dog1_id in self.distance_matrix.index and dog2_id in self.distance_matrix.columns:
                dist = self.distance_matrix.loc[dog1_id, dog2_id]
                return float(dist) if not pd.isna(dist) else float('inf')
        except:
            pass
        return float('inf')
    
    def _groups_compatible(self, dog_groups, driver_groups):
        """Check if driver's groups can handle dog's groups"""
        # Direct match - driver has all needed groups
        if all(g in driver_groups for g in dog_groups):
            return True, 1.0  # No penalty
        
        # Adjacent groups - more flexible
        # Group 2 can handle 1 or 3, Groups 1 and 3 can handle 2
        extended_groups = set(driver_groups)
        if 1 in driver_groups:
            extended_groups.add(2)
        if 2 in driver_groups:
            extended_groups.update([1, 3])
        if 3 in driver_groups:
            extended_groups.add(2)
        
        if all(g in extended_groups for g in dog_groups):
            return True, self.ADJACENT_GROUP_PENALTY  # Penalty for adjacent
        
        return False, float('inf')
    
    def _get_driver_load(self, driver):
        """Calculate current load for a driver"""
        load = {'group1': 0, 'group2': 0, 'group3': 0}
        for dog in self.current_state:
            if dog['driver'] == driver:
                for group in dog['groups']:
                    load[f'group{group}'] += dog['num_dogs']
        return load
    
    def _can_driver_accept(self, driver, dog):
        """Check if driver has capacity for dog"""
        load = self._get_driver_load(driver)
        capacity = self.driver_capacities.get(driver, {})
        
        for group in dog['groups']:
            current = load.get(f'group{group}', 0)
            max_cap = capacity.get(f'group{group}', 0)
            if current + dog['num_dogs'] > max_cap:
                return False
        return True
    
    def _find_outlier_dog(self, driver, groups_needed):
        """Find the outlier dog in driver's route that should be moved"""
        driver_dogs = [d for d in self.current_state if d['driver'] == driver and not d.get('is_callout', False)]
        
        # Need at least 2 dogs to find an outlier
        if len(driver_dogs) <= 1:
            return None
        
        # Calculate average distance for each dog to all others
        outlier_scores = []
        for dog in driver_dogs:
            # Skip if dog doesn't help free the needed groups
            if not any(g in groups_needed for g in dog['groups']):
                continue
            
            # Skip if this dog isn't in our distance matrix
            if dog['dog_id'] not in self.distance_matrix.index:
                continue
            
            # Calculate average distance to other dogs in this driver's route
            distances = []
            for other_dog in driver_dogs:
                if other_dog['dog_id'] != dog['dog_id']:
                    dist = self._get_distance(dog['dog_id'], other_dog['dog_id'])
                    if dist < 50:  # Skip placeholder distances
                        distances.append(dist)
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                outlier_scores.append((dog, avg_distance))
        
        if not outlier_scores:
            # Fallback: just pick any dog that has needed groups
            for dog in driver_dogs:
                if any(g in groups_needed for g in dog['groups']):
                    print(f"   ⚠️ No clear outlier, picking {dog['dog_name']}")
                    return dog
            return None
        
        # Return the dog with highest average distance (biggest outlier)
        outlier_scores.sort(key=lambda x: x[1], reverse=True)
        outlier_dog = outlier_scores[0][0]
        
        print(f"   🎯 Outlier in {driver}'s route: {outlier_dog['dog_name']} (avg dist: {outlier_scores[0][1]:.1f} mi)")
        return outlier_dog
    
    def _find_best_driver_for_dog(self, dog, exclude_drivers=None):
        """Find the closest available driver for a dog"""
        exclude = exclude_drivers or []
        best_option = None
        best_score = float('inf')
        
        # Get all unique drivers
        drivers = set(d['driver'] for d in self.current_state if d['driver'])
        
        for driver in drivers:
            if driver in exclude or driver in self.called_out_drivers:
                continue
            
            # Get driver's groups
            driver_groups = set()
            for d in self.current_state:
                if d['driver'] == driver:
                    driver_groups.update(d['groups'])
            
            # Check group compatibility
            compatible, penalty = self._groups_compatible(dog['groups'], list(driver_groups))
            if not compatible:
                continue
            
            # Check capacity
            if not self._can_driver_accept(driver, dog):
                continue
            
            # Find closest dog in this driver's route
            min_distance = float('inf')
            for other_dog in self.current_state:
                if other_dog['driver'] == driver:
                    dist = self._get_distance(dog['dog_id'], other_dog['dog_id'])
                    if dist < min_distance:
                        min_distance = dist
            
            # Score combines distance and group penalty
            score = min_distance * penalty
            
            if score < best_score:
                best_score = score
                best_option = {
                    'driver': driver,
                    'distance': min_distance,
                    'score': score,
                    'is_adjacent': penalty > 1
                }
        
        return best_option
    
    def _execute_cascading_move(self, blocked_driver, callout_dog):
        """Execute a smart cascading move to free space"""
        print(f"\n   🔄 Cascading needed: {blocked_driver} is full")
        
        # Find the outlier dog to move
        outlier = self._find_outlier_dog(blocked_driver, callout_dog['groups'])
        if not outlier:
            print(f"   ❌ No suitable dog to move from {blocked_driver}")
            return False
        
        # Find best new driver for the outlier
        new_home = self._find_best_driver_for_dog(outlier, exclude_drivers=[blocked_driver])
        if not new_home:
            print(f"   ❌ No driver can accept {outlier['dog_name']}")
            return False
        
        # Execute the move
        print(f"   ✅ Moving {outlier['dog_name']}: {blocked_driver} → {new_home['driver']} ({new_home['distance']:.1f} mi)")
        
        # Update state
        for dog in self.current_state:
            if dog['dog_id'] == outlier['dog_id']:
                dog['driver'] = new_home['driver']
                break
        
        # Record the move
        self.cascading_moves.append({
            'dog_name': outlier['dog_name'],
            'dog_id': outlier['dog_id'],
            'from_driver': blocked_driver,
            'to_driver': new_home['driver'],
            'distance': new_home['distance'],
            'reason': f"Free space for {callout_dog['dog_name']}"
        })
        
        return True
    
    def run_assignments(self):
        """Main assignment algorithm with smart cascading"""
        print("\n🚀 Running Smart Assignment Algorithm")
        print("=" * 60)
        
        # Configuration
        CAPACITY_BYPASS_THRESHOLD = 0.3  # If driver with capacity is within 0.3 mi of closest, use them instead
        
        # Get callout dogs
        callouts = [d for d in self.current_state if d['is_callout']]
        if not callouts:
            print("✅ No callouts to process!")
            return []
        
        print(f"📋 Processing {len(callouts)} callout dogs...")
        print(f"📏 Cascading bypass: Will use driver with capacity if within {CAPACITY_BYPASS_THRESHOLD} mi of closest")
        
        # Process each callout dog
        for i, callout_dog in enumerate(callouts, 1):
            print(f"\n{i}. {callout_dog['dog_name']} (needs groups {callout_dog['groups']}):")
            
            # Find ALL potential drivers, sorted by distance
            all_options = []
            drivers = set(d['driver'] for d in self.current_state if d['driver'])
            
            for driver in drivers:
                if driver in self.called_out_drivers:
                    continue
                
                # Get driver's groups
                driver_groups = set()
                for d in self.current_state:
                    if d['driver'] == driver:
                        driver_groups.update(d['groups'])
                
                # Check compatibility
                compatible, penalty = self._groups_compatible(callout_dog['groups'], list(driver_groups))
                if not compatible:
                    continue
                
                # Find closest distance
                min_distance = float('inf')
                for other_dog in self.current_state:
                    if other_dog['driver'] == driver:
                        dist = self._get_distance(callout_dog['dog_id'], other_dog['dog_id'])
                        if dist < min_distance:
                            min_distance = dist
                
                if min_distance < self.MAX_SEARCH_RADIUS:
                    has_capacity = self._can_driver_accept(driver, callout_dog)
                    all_options.append({
                        'driver': driver,
                        'distance': min_distance,
                        'score': min_distance * penalty,
                        'has_capacity': has_capacity,
                        'is_adjacent': penalty > 1
                    })
            
            # Sort by score (distance * penalty)
            all_options.sort(key=lambda x: x['score'])
            
            if not all_options:
                print(f"   ❌ No compatible drivers found within {self.MAX_SEARCH_RADIUS} miles")
                self.assignments_made.append({
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'driver': 'UNASSIGNED',
                    'distance': float('inf'),
                    'quality': 'FAILED',
                    'new_assignment': f"UNASSIGNED:{callout_dog['assignment_string']}",
                    'original': callout_dog['original']
                })
                continue
            
            # SMART LOGIC: Check if we should bypass cascading
            assigned = False
            closest_option = all_options[0]
            
            # If closest doesn't have capacity, check if there's one with capacity nearby
            if not closest_option['has_capacity']:
                # Find closest driver WITH capacity
                closest_with_capacity = None
                for option in all_options:
                    if option['has_capacity']:
                        closest_with_capacity = option
                        break  # Already sorted by distance
                
                # Check if it's within threshold of the absolute closest
                if closest_with_capacity:
                    distance_diff = closest_with_capacity['distance'] - closest_option['distance']
                    
                    if distance_diff <= CAPACITY_BYPASS_THRESHOLD:
                        # Use the one with capacity to avoid cascading
                        print(f"   📊 Closest: {closest_option['driver']} ({closest_option['distance']:.1f} mi) - FULL")
                        print(f"   ✨ Using {closest_with_capacity['driver']} ({closest_with_capacity['distance']:.1f} mi) - has capacity")
                        print(f"   💡 Avoiding cascade (only {distance_diff:.1f} mi farther)")
                        
                        # Assign to driver with capacity
                        callout_dog['driver'] = closest_with_capacity['driver']
                        self.assignments_made.append({
                            'dog_id': callout_dog['dog_id'],
                            'dog_name': callout_dog['dog_name'],
                            'driver': closest_with_capacity['driver'],
                            'distance': closest_with_capacity['distance'],
                            'quality': 'GOOD' if closest_with_capacity['distance'] <= 1.5 else 'BACKUP',
                            'new_assignment': f"{closest_with_capacity['driver']}:{callout_dog['assignment_string']}",
                            'original': callout_dog['original']
                        })
                        assigned = True
                    else:
                        # Too far, need to cascade
                        print(f"   📊 Closest: {closest_option['driver']} ({closest_option['distance']:.1f} mi) - FULL")
                        print(f"   📊 Next with capacity: {closest_with_capacity['driver']} ({closest_with_capacity['distance']:.1f} mi)")
                        print(f"   🔄 Need cascade ({distance_diff:.1f} mi difference > {CAPACITY_BYPASS_THRESHOLD} mi threshold)")
                else:
                    print(f"   📊 Closest: {closest_option['driver']} ({closest_option['distance']:.1f} mi) - FULL")
                    print(f"   🔄 No drivers with capacity found - must cascade")
            else:
                # Closest has capacity - direct assignment
                print(f"   ✅ Direct assignment to {closest_option['driver']} ({closest_option['distance']:.1f} mi)")
                callout_dog['driver'] = closest_option['driver']
                self.assignments_made.append({
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'driver': closest_option['driver'],
                    'distance': closest_option['distance'],
                    'quality': 'GOOD' if closest_option['distance'] <= 1.5 else 'BACKUP',
                    'new_assignment': f"{closest_option['driver']}:{callout_dog['assignment_string']}",
                    'original': callout_dog['original']
                })
                assigned = True
            
            # If not assigned yet, try cascading to the closest
            if not assigned:
                if self._execute_cascading_move(closest_option['driver'], callout_dog):
                    # Now there's space, assign the dog
                    print(f"   ✅ After cascading, assigned to {closest_option['driver']} ({closest_option['distance']:.1f} mi)")
                    
                    callout_dog['driver'] = closest_option['driver']
                    self.assignments_made.append({
                        'dog_id': callout_dog['dog_id'],
                        'dog_name': callout_dog['dog_name'],
                        'driver': closest_option['driver'],
                        'distance': closest_option['distance'],
                        'quality': 'CASCADED',
                        'new_assignment': f"{closest_option['driver']}:{callout_dog['assignment_string']}",
                        'original': callout_dog['original']
                    })
                    assigned = True
                else:
                    print(f"   ❌ Cascading failed - could not make space")
                    self.assignments_made.append({
                        'dog_id': callout_dog['dog_id'],
                        'dog_name': callout_dog['dog_name'],
                        'driver': 'UNASSIGNED',
                        'distance': float('inf'),
                        'quality': 'FAILED',
                        'new_assignment': f"UNASSIGNED:{callout_dog['assignment_string']}",
                        'original': callout_dog['original']
                    })
        
        # Print summary
        print("\n" + "="*60)
        print("📊 ASSIGNMENT SUMMARY:")
        print(f"   Total processed: {len(self.assignments_made)}")
        print(f"   Cascading moves: {len(self.cascading_moves)}")
        
        quality_counts = {}
        for a in self.assignments_made:
            q = a['quality']
            quality_counts[q] = quality_counts.get(q, 0) + 1
        
        for quality, count in quality_counts.items():
            print(f"   {quality}: {count}")
        
        if self.cascading_moves:
            print(f"\n🔄 Cascading moves made:")
            for move in self.cascading_moves:
                print(f"   - {move['dog_name']}: {move['from_driver']} → {move['to_driver']} ({move['distance']:.1f} mi)")
        
        return self.assignments_made
    
    def setup_sheets_client(self):
        """Setup Google Sheets client for writing results"""
        try:
            creds_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            if not creds_json:
                print("⚠️ No credentials for writing results")
                return False
            
            creds = Credentials.from_service_account_info(
                json.loads(creds_json),
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.sheets_client = gspread.authorize(creds)
            return True
        except Exception as e:
            print(f"⚠️ Sheets client error: {e}")
            return False
    
    def write_results(self):
        """Write results back to Google Sheets"""
        if not hasattr(self, 'sheets_client'):
            if not self.setup_sheets_client():
                return False
        
        try:
            print("\n📝 Writing results to Google Sheets...")
            
            sheet = self.sheets_client.open_by_key(self.MAP_SHEET_ID)
            ws = sheet.get_worksheet(0)
            data = ws.get_all_values()
            
            updates = []
            
            # Update main assignments
            for result in self.assignments_made:
                for i, row in enumerate(data[1:], 1):
                    if len(row) > 9 and row[9] == result['dog_id']:
                        # Column H (combined)
                        updates.append({
                            'range': f'H{i+1}',
                            'values': [[result['new_assignment']]]
                        })
                        # Column K (notes)
                        updates.append({
                            'range': f'K{i+1}',
                            'values': [[f"Reassigned: {result['original']}"]]
                        })
                        break
            
            # Update cascading moves
            for move in self.cascading_moves:
                for i, row in enumerate(data[1:], 1):
                    if len(row) > 9 and row[9] == move['dog_id']:
                        # Update the driver in combined column
                        if len(row) > 7 and row[7] and ':' in row[7]:
                            old_combined = row[7]
                            assignment_part = old_combined.split(':', 1)[1]
                            new_combined = f"{move['to_driver']}:{assignment_part}"
                            updates.append({
                                'range': f'H{i+1}',
                                'values': [[new_combined]]
                            })
                            updates.append({
                                'range': f'K{i+1}',
                                'values': [[f"Cascaded: {move['from_driver']} → {move['to_driver']}"]]
                            })
                        break
            
            if updates:
                ws.batch_update(updates)
                print(f"✅ Updated {len(updates)//2} assignments")
                return True
                
        except Exception as e:
            print(f"❌ Write error: {e}")
            return False


def test_system():
    """Test function to validate data loading and system setup"""
    print("🧪 Running system tests...\n")
    
    system = SmartDogReassignment()
    
    # Test 1: Data loading
    print("Test 1: Loading data...")
    if not system.load_data():
        print("❌ Data loading failed")
        return False
    
    # Test 2: Check distance matrix
    print("\nTest 2: Distance matrix validation...")
    if system.distance_matrix is not None:
        print(f"  ✅ Matrix shape: {system.distance_matrix.shape}")
        
        # Check for valid distances
        valid_count = 0
        total_count = 0
        for i in range(min(5, len(system.distance_matrix))):
            for j in range(min(5, len(system.distance_matrix.columns))):
                val = system.distance_matrix.iloc[i, j]
                total_count += 1
                if not pd.isna(val) and val < 50:
                    valid_count += 1
        
        print(f"  ✅ Sample valid distances: {valid_count}/{total_count}")
    else:
        print("  ❌ No distance matrix loaded")
    
    # Test 3: Check state
    print("\nTest 3: State validation...")
    callouts = [d for d in system.current_state if d.get('is_callout', False)]
    assigned = [d for d in system.current_state if d.get('driver')]
    
    print(f"  ✅ Total dogs: {len(system.current_state)}")
    print(f"  ✅ Assigned dogs: {len(assigned)}")
    print(f"  ✅ Callout dogs: {len(callouts)}")
    print(f"  ✅ Drivers with capacity: {len(system.driver_capacities)}")
    
    # Test 4: Sample callout dogs
    if callouts:
        print("\nTest 4: Sample callout dogs:")
        for dog in callouts[:3]:
            print(f"  - {dog['dog_name']} needs groups {dog['groups']}")
    
    # Test 5: Sample capacities
    print("\nTest 5: Sample driver capacities:")
    for driver, cap in list(system.driver_capacities.items())[:3]:
        print(f"  - {driver}: G1={cap['group1']}, G2={cap['group2']}, G3={cap['group3']}")
    
    print("\n✅ All tests passed!")
    return True


def main():
    """Main entry point with error handling"""
    print("🐕 Smart Dog Reassignment System")
    print("🎯 Strategy: Assign to closest driver, move outliers when needed")
    print("=" * 60)
    
    # Run tests first in debug mode
    if os.environ.get('DEBUG'):
        print("🧪 Debug mode - running tests first...\n")
        if not test_system():
            print("❌ Tests failed - exiting")
            return
        print("\n" + "="*60 + "\n")
    
    system = SmartDogReassignment()
    
    # Load data
    if not system.load_data():
        print("❌ Failed to load data")
        print("\n💡 Troubleshooting tips:")
        print("1. Check if both Google Sheets are published to web")
        print("2. Try opening these URLs in your browser:")
        print(f"   - Distance: {system.DISTANCE_MATRIX_URL[:100]}...")
        print(f"   - Map: {system.MAP_SHEET_URL[:100]}...")
        print("3. Run with DEBUG=1 for more details")
        return
    
    # Run assignments
    results = system.run_assignments()
    
    # Write results if any
    if results:
        if system.setup_sheets_client():
            system.write_results()
        else:
            print("\n⚠️ Could not write results (no credentials)")
            print("Results computed but not written to sheets")
        print("\n✨ Complete!")
    else:
        print("\n✅ No assignments needed")


if __name__ == "__main__":
    main()
