# bulletproof_capacity_assignment.py
# BULLETPROOF VERSION: Aggressive capacity validation that PREVENTS violations

import pandas as pd
import numpy as np
import requests
import json
import os
from typing import Dict, List, Tuple
import gspread
from google.oauth2.service_account import Credentials
import re
from copy import deepcopy

class BulletproofCapacitySystem:
    def __init__(self):
        """Initialize with bulletproof capacity tracking"""
        # Google Sheets URLs
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        
        # TIGHT DISTANCE THRESHOLDS
        self.MAX_DIRECT_DISTANCE = 0.5
        self.MAX_ADJACENT_DISTANCE = 0.3
        self.MAX_CASCADE_DISTANCE = 0.6
        
        # Data containers
        self.distance_matrix = None
        self.dog_assignments = None
        self.driver_capacities = None
        self.sheets_client = None
        
        # BULLETPROOF CAPACITY TRACKING
        self.current_driver_loads = {}  # Real-time capacity tracking
        self.assignment_log = []  # Every assignment for debugging

    def setup_google_sheets_client(self):
        """Setup Google Sheets API client"""
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
        """Load distance matrix"""
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
        """Load dog assignments"""
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
                        'callout': str(callout) if not pd.isna(callout) else "",
                        'num_dogs': num_dogs
                    })
                    
                except Exception as e:
                    continue
            
            self.dog_assignments = assignments
            print(f"✅ Loaded {len(assignments)} assignments")
            return True
            
        except Exception as e:
            print(f"❌ Error loading dog assignments: {e}")
            return False

    def load_driver_capacities(self):
        """Load driver capacities"""
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
                    
                    if not driver_name or pd.isna(driver_name):
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

    def initialize_capacity_tracking(self):
        """BULLETPROOF: Initialize real-time capacity tracking"""
        print("\n🔒 INITIALIZING BULLETPROOF CAPACITY TRACKING...")
        
        # Initialize all drivers to zero load
        self.current_driver_loads = {}
        for driver in self.driver_capacities.keys():
            self.current_driver_loads[driver] = {
                'group1': 0,
                'group2': 0,
                'group3': 0
            }
        
        # Process existing assignments
        existing_assignments = 0
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if combined and ':' in combined:
                driver = combined.split(':', 1)[0].strip()
                assignment_part = combined.split(':', 1)[1].strip()
                groups = self.extract_groups_from_assignment(assignment_part)
                
                if driver in self.current_driver_loads:
                    for group in groups:
                        group_key = f'group{group}'
                        self.current_driver_loads[driver][group_key] += assignment['num_dogs']
                    existing_assignments += 1
                    
                    # Log existing assignment
                    self.assignment_log.append({
                        'dog_id': assignment['dog_id'],
                        'dog_name': assignment['dog_name'],
                        'driver': driver,
                        'groups': groups,
                        'num_dogs': assignment['num_dogs'],
                        'action': 'EXISTING'
                    })
        
        # Validate initial state
        violations = self.validate_all_capacities("INITIAL STATE")
        if violations:
            print(f"🚨 WARNING: {len(violations)} capacity violations in existing assignments!")
            for violation in violations:
                print(f"   {violation['driver']} - Group {violation['group']}: {violation['current']}/{violation['capacity']}")
        else:
            print(f"✅ Initial state valid: {existing_assignments} existing assignments")
        
        # Show capacity summary
        print(f"\n📊 CAPACITY SUMMARY:")
        for driver, load in self.current_driver_loads.items():
            capacity = self.driver_capacities[driver]
            print(f"   {driver}:")
            for group_key in ['group1', 'group2', 'group3']:
                current = load[group_key]
                max_cap = capacity[group_key]
                if max_cap > 0:
                    print(f"     {group_key}: {current}/{max_cap} ({max_cap-current} available)")

    def extract_groups_from_assignment(self, assignment_string):
        """Extract group numbers from assignment string"""
        try:
            group_digits = re.findall(r'[123]', assignment_string)
            groups = sorted(list(set(int(digit) for digit in group_digits)))
            return groups
        except:
            return []

    def get_distance(self, dog1_id: str, dog2_id: str) -> float:
        """Get distance between two dogs"""
        try:
            if self.distance_matrix is None:
                return float('inf')
            
            if dog1_id in self.distance_matrix.index and dog2_id in self.distance_matrix.columns:
                distance = self.distance_matrix.loc[dog1_id, dog2_id]
                return float(distance) if not pd.isna(distance) else float('inf')
            
            return float('inf')
        except:
            return float('inf')

    def check_group_compatibility(self, dog_groups, target_groups, distance):
        """Check if groups are compatible within distance limits"""
        dog_set = set(dog_groups)
        target_set = set(target_groups)
        
        # Perfect match - same groups
        if dog_set.intersection(target_set):
            return distance <= self.MAX_DIRECT_DISTANCE
        
        # Adjacent groups
        adjacent_pairs = [(1, 2), (2, 3), (2, 1), (3, 2)]
        for dog_group in dog_set:
            for target_group in target_set:
                if (dog_group, target_group) in adjacent_pairs:
                    return distance <= self.MAX_ADJACENT_DISTANCE
        
        return False

    def validate_all_capacities(self, context=""):
        """BULLETPROOF: Validate all driver capacities"""
        violations = []
        
        for driver, load in self.current_driver_loads.items():
            capacity = self.driver_capacities.get(driver, {})
            
            for group_key in ['group1', 'group2', 'group3']:
                current = load.get(group_key, 0)
                max_cap = capacity.get(group_key, 0)
                
                if current > max_cap:
                    violations.append({
                        'driver': driver,
                        'group': int(group_key[-1]),
                        'current': current,
                        'capacity': max_cap,
                        'overage': current - max_cap,
                        'context': context
                    })
        
        return violations

    def can_assign_dog_to_driver(self, dog_groups, num_dogs, driver):
        """BULLETPROOF: Check if a dog can be assigned to a driver"""
        if driver not in self.current_driver_loads:
            return False, "Driver not found"
        
        current_load = self.current_driver_loads[driver]
        capacity = self.driver_capacities.get(driver, {})
        
        # Check each group the dog needs
        for group in dog_groups:
            group_key = f'group{group}'
            current = current_load.get(group_key, 0)
            max_cap = capacity.get(group_key, 0)
            
            if current + num_dogs > max_cap:
                return False, f"Group {group}: {current} + {num_dogs} > {max_cap}"
        
        return True, "OK"

    def assign_dog_to_driver(self, dog_id, dog_name, dog_groups, num_dogs, driver, reason=""):
        """BULLETPROOF: Assign a dog and update capacity tracking"""
        # Double-check capacity before assignment
        can_assign, reason_text = self.can_assign_dog_to_driver(dog_groups, num_dogs, driver)
        
        if not can_assign:
            print(f"🚨 BULLETPROOF BLOCK: Cannot assign {dog_name} to {driver}: {reason_text}")
            return False
        
        # Make the assignment
        for group in dog_groups:
            group_key = f'group{group}'
            self.current_driver_loads[driver][group_key] += num_dogs
        
        # Log the assignment
        self.assignment_log.append({
            'dog_id': dog_id,
            'dog_name': dog_name,
            'driver': driver,
            'groups': dog_groups,
            'num_dogs': num_dogs,
            'action': f'ASSIGNED - {reason}'
        })
        
        print(f"✅ BULLETPROOF ASSIGN: {dog_name} → {driver} (groups: {dog_groups}, {num_dogs} dogs)")
        
        # Validate state after assignment
        violations = self.validate_all_capacities(f"After assigning {dog_name}")
        if violations:
            print(f"🚨 CRITICAL ERROR: Assignment created violations!")
            for violation in violations:
                print(f"   {violation['driver']} - Group {violation['group']}: {violation['current']}/{violation['capacity']}")
            return False
        
        return True

    def get_callout_dogs(self):
        """Get dogs that need reassignment"""
        callout_dogs = []
        
        for assignment in self.dog_assignments:
            combined_blank = (not assignment['combined'] or assignment['combined'].strip() == "")
            callout_has_content = (assignment['callout'] and assignment['callout'].strip() != "")
            
            if combined_blank and callout_has_content:
                # Filter out non-dogs
                dog_name = str(assignment.get('dog_name', '')).lower().strip()
                if any(keyword in dog_name for keyword in ['parking', 'field', 'admin', 'office']):
                    continue
                
                callout_text = assignment['callout'].strip()
                if ':' not in callout_text:
                    continue
                
                original_driver = callout_text.split(':', 1)[0].strip()
                full_assignment_string = callout_text.split(':', 1)[1].strip()
                needed_groups = self.extract_groups_from_assignment(full_assignment_string)
                
                if needed_groups:
                    callout_dogs.append({
                        'dog_id': assignment['dog_id'],
                        'dog_name': assignment['dog_name'],
                        'num_dogs': assignment['num_dogs'],
                        'needed_groups': needed_groups,
                        'full_assignment_string': full_assignment_string,
                        'original_driver': original_driver
                    })
        
        return callout_dogs

    def find_closest_available_drivers(self, callout_dog):
        """Find closest drivers who ACTUALLY have capacity available"""
        available_candidates = []
        booked_candidates = []
        
        # Get all dogs currently assigned to each driver
        driver_dogs = {}
        for log_entry in self.assignment_log:
            if log_entry['action'] != 'REMOVED':
                driver = log_entry['driver']
                if driver not in driver_dogs:
                    driver_dogs[driver] = []
                driver_dogs[driver].append(log_entry)
        
        # Check each driver - CAPACITY FIRST, then distance
        for driver in self.driver_capacities.keys():
            # Step 1: Check capacity FIRST
            can_assign, reason = self.can_assign_dog_to_driver(
                callout_dog['needed_groups'], 
                callout_dog['num_dogs'], 
                driver
            )
            
            # Skip drivers with no capacity
            if not can_assign:
                booked_candidates.append({
                    'driver': driver,
                    'can_assign': False,
                    'reason': reason,
                    'distance': float('inf')
                })
                continue
            
            # Step 2: Find closest distance to this AVAILABLE driver
            if driver not in driver_dogs or not driver_dogs[driver]:
                # Driver has capacity but no assigned dogs - can't calculate distance
                continue
                
            closest_distance = float('inf')
            closest_via_dog = None
            
            for assigned_dog in driver_dogs[driver]:
                distance = self.get_distance(callout_dog['dog_id'], assigned_dog['dog_id'])
                if distance < closest_distance and distance < 100:
                    closest_distance = distance
                    closest_via_dog = assigned_dog
            
            if closest_via_dog is None:
                continue
            
            # Step 3: Check group compatibility
            if not self.check_group_compatibility(
                callout_dog['needed_groups'], 
                closest_via_dog['groups'], 
                closest_distance
            ):
                continue
            
            # This driver has capacity AND is compatible
            available_candidates.append({
                'driver': driver,
                'distance': closest_distance,
                'via_dog': closest_via_dog['dog_name'],
                'can_assign': True,
                'reason': "Available"
            })
        
        # Sort available drivers by distance (closest first)
        available_candidates.sort(key=lambda x: x['distance'])
        
        # Sort booked drivers by driver name for consistent reporting
        booked_candidates.sort(key=lambda x: x['driver'])
        
        # Return available first, then booked for reporting
        return available_candidates + booked_candidates

    def process_callouts_bulletproof(self):
        """BULLETPROOF: Process all callouts with guaranteed capacity compliance"""
        print("\n🔒 BULLETPROOF CALLOUT PROCESSING")
        print("🎯 Guaranteed capacity compliance with aggressive validation")
        print("📏 Only considers drivers who ACTUALLY have capacity available")
        print("=" * 60)
        
        callout_dogs = self.get_callout_dogs()
        
        if not callout_dogs:
            print("✅ No callouts found!")
            return []
        
        print(f"🎯 Processing {len(callout_dogs)} callout dogs:")
        for dog in callout_dogs:
            print(f"   - {dog['dog_name']} (groups: {dog['needed_groups']}, {dog['num_dogs']} dogs)")
        
        assignments_made = []
        
        # Process each callout dog
        for callout_dog in callout_dogs:
            print(f"\n🎯 Processing: {callout_dog['dog_name']}")
            
            # Find available drivers (capacity first, then distance)
            candidates = self.find_closest_available_drivers(callout_dog)
            
            # Separate available from booked
            available = [c for c in candidates if c['can_assign']]
            booked = [c for c in candidates if not c['can_assign']]
            
            print(f"   ✅ AVAILABLE drivers with capacity ({len(available)}):")
            if available:
                for i, candidate in enumerate(available[:3]):  # Show top 3 available
                    print(f"     {i+1}. {candidate['driver']} - {candidate['distance']:.2f}mi via {candidate['via_dog']}")
            else:
                print(f"     ❌ NO DRIVERS HAVE CAPACITY!")
            
            if booked:
                print(f"   📋 Booked drivers (no capacity) - {len(booked)} total:")
                for i, candidate in enumerate(booked[:3]):  # Show top 3 booked
                    print(f"     {i+1}. {candidate['driver']} - {candidate['reason']}")
            
            # Try to assign to the closest available driver
            assigned = False
            if available:
                best_candidate = available[0]  # Closest available driver
                
                success = self.assign_dog_to_driver(
                    callout_dog['dog_id'],
                    callout_dog['dog_name'],
                    callout_dog['needed_groups'],
                    callout_dog['num_dogs'],
                    best_candidate['driver'],
                    f"CALLOUT - {best_candidate['distance']:.2f}mi"
                )
                
                if success:
                    assignments_made.append({
                        'dog_id': callout_dog['dog_id'],
                        'dog_name': callout_dog['dog_name'],
                        'new_assignment': f"{best_candidate['driver']}:{callout_dog['full_assignment_string']}",
                        'driver': best_candidate['driver'],
                        'distance': best_candidate['distance']
                    })
                    assigned = True
                    print(f"   ✅ ASSIGNED to closest available driver: {best_candidate['driver']} ({best_candidate['distance']:.2f}mi)")
            
            if not assigned:
                print(f"   ❌ Could not assign {callout_dog['dog_name']} - no drivers have capacity")
                assignments_made.append({
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'new_assignment': f"UNASSIGNED:{callout_dog['full_assignment_string']}",
                    'driver': 'UNASSIGNED',
                    'distance': float('inf')
                })
        
        # Final validation
        print(f"\n🔒 FINAL BULLETPROOF VALIDATION:")
        violations = self.validate_all_capacities("FINAL STATE")
        if violations:
            print(f"🚨 CRITICAL ERROR: {len(violations)} violations detected!")
            for violation in violations:
                print(f"   {violation['driver']} - Group {violation['group']}: {violation['current']}/{violation['capacity']}")
            print(f"🚨 ABORTING - BULLETPROOF SYSTEM DETECTED VIOLATIONS!")
            return []
        else:
            assigned_count = len([a for a in assignments_made if a['driver'] != 'UNASSIGNED'])
            unassigned_count = len([a for a in assignments_made if a['driver'] == 'UNASSIGNED'])
            
            print(f"✅ BULLETPROOF SUCCESS: No capacity violations detected")
            print(f"📊 Results: {assigned_count} assigned, {unassigned_count} unassigned (no capacity)")
            
            if assigned_count > 0:
                avg_distance = sum(a['distance'] for a in assignments_made if a['distance'] != float('inf')) / assigned_count
                print(f"📏 Average assignment distance: {avg_distance:.2f}mi")
        
        return assignments_made

    def write_results_to_sheets(self, assignments):
        """Write results to Google Sheets"""
        try:
            if not assignments:
                print("✅ No assignments to write")
                return True
            
            print(f"\n📝 Writing {len(assignments)} bulletproof assignments to Google Sheets...")
            
            if not self.sheets_client:
                print("❌ Google Sheets client not initialized")
                return False
            
            # Open spreadsheet
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
                worksheet = spreadsheet.worksheets()[0]
            
            # Get all data
            all_data = worksheet.get_all_values()
            header_row = all_data[0]
            
            # Find Dog ID column
            dog_id_col = None
            for i, header in enumerate(header_row):
                if 'dog id' in str(header).lower():
                    dog_id_col = i
                    break
            
            if dog_id_col is None:
                print("❌ Could not find Dog ID column")
                return False
            
            target_col = 7  # Column H (Combined)
            updates = []
            
            # Process each assignment
            for assignment in assignments:
                dog_id = str(assignment['dog_id']).strip()
                new_assignment = str(assignment['new_assignment']).strip()
                
                # Find the row for this dog
                for row_idx in range(1, len(all_data)):
                    if dog_id_col < len(all_data[row_idx]):
                        current_dog_id = str(all_data[row_idx][dog_id_col]).strip()
                        
                        if current_dog_id == dog_id:
                            cell_address = gspread.utils.rowcol_to_a1(row_idx + 1, target_col + 1)
                            
                            updates.append({
                                'range': cell_address,
                                'values': [[new_assignment]]
                            })
                            
                            print(f"  ✅ {dog_id} → {new_assignment}")
                            break
            
            if updates:
                worksheet.batch_update(updates)
                print(f"\n✅ BULLETPROOF SUCCESS: {len(updates)} assignments written with zero violations!")
                
                # Send Slack notification
                slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
                if slack_webhook:
                    try:
                        message = f"🔒 Bulletproof Dog Assignment: {len(updates)} assignments with ZERO capacity violations"
                        slack_message = {"text": message}
                        requests.post(slack_webhook, json=slack_message, timeout=10)
                        print("📱 Bulletproof success notification sent")
                    except:
                        pass
                
                return True
            else:
                print("❌ No valid updates to make")
                return False
            
        except Exception as e:
            print(f"❌ Error writing bulletproof assignments: {e}")
            return False


def main():
    """Main function for bulletproof capacity-aware assignment"""
    print("🔒 BULLETPROOF CAPACITY-AWARE DOG ASSIGNMENT SYSTEM")
    print("🎯 Zero tolerance for capacity violations")
    print("📊 Real-time capacity tracking with aggressive validation")
    print("✅ Checks capacity FIRST, then finds closest available driver")
    print("🚫 Ignores booked drivers completely - only considers available drivers")
    print("✅ Guaranteed capacity compliance or assignment rejection")
    print("=" * 70)
    
    system = BulletproofCapacitySystem()
    
    # Setup
    if not system.setup_google_sheets_client():
        return
    
    # Load data
    print("\n⬇️ Loading data...")
    if not system.load_distance_matrix():
        return
    if not system.load_dog_assignments():
        return
    if not system.load_driver_capacities():
        return
    
    # Initialize bulletproof capacity tracking
    system.initialize_capacity_tracking()
    
    # Process callouts with bulletproof capacity checking
    assignments = system.process_callouts_bulletproof()
    
    if assignments:
        # Write results
        success = system.write_results_to_sheets(assignments)
        if success:
            valid_assignments = len([a for a in assignments if a['driver'] != 'UNASSIGNED'])
            total_callouts = len(assignments)
            print(f"\n🎉 BULLETPROOF SUCCESS! {valid_assignments}/{total_callouts} callouts assigned with ZERO violations")
            
            if valid_assignments < total_callouts:
                unassigned = total_callouts - valid_assignments
                print(f"⚠️ {unassigned} dogs remain unassigned due to insufficient capacity")
                print("💡 Consider: Adding driver capacity or moving existing assignments")
        else:
            print(f"\n❌ Bulletproof assignments failed to write to sheets")
    else:
        print(f"\n⚠️ No assignments made - either no callouts or capacity validation failed")


if __name__ == "__main__":
    main()
