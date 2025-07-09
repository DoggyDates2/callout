# production_reassignment.py
# Your EXACT working logic from Streamlit, adapted for GitHub Actions → Google Sheets
# All algorithms preserved exactly as you built them
# FIXED: Driver capacities now loaded from columns R:W on map sheet

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Set
import json
import os
from datetime import datetime
import re
from collections import defaultdict

# Google Sheets API
import gspread
from google.oauth2.service_account import Credentials

class ProductionDogReassignmentSystem:
    """
    Production version with your EXACT working logic
    Only the I/O changed (GitHub + Google Sheets instead of Streamlit + file uploads)
    FIXED: Driver capacities loaded from same sheet as map data (columns R:W)
    """
    
    def __init__(self):
        # Updated URLs with correct GIDs
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        # Driver capacities are now on the same sheet as map data (columns R:W)
        
        # For writing results back
        self.MAP_SPREADSHEET_ID = "1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0"
        
        # Data storage
        self.distance_matrix = {}
        self.dog_assignments = {}
        self.driver_capacities = {}
        self.callout_dogs = []
        
        print("🚀 Production Dog Reassignment System Initialized")
    
    def setup_google_sheets_client(self):
        """Setup Google Sheets API client using service account credentials"""
        try:
            # DEBUG: Print all environment variables to see what's available
            print("🔍 DEBUG: Available environment variables:")
            for key, value in os.environ.items():
                if 'GOOGLE' in key or 'TEST' in key or 'SECRET' in key:
                    # Show first/last 20 chars for security
                    safe_value = value[:20] + "..." + value[-20:] if len(value) > 40 else value[:10] + "..."
                    print(f"   {key}: {safe_value}")
            
            credentials_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            test_secret = os.environ.get('TEST_SECRET')
            
            print(f"🔍 GOOGLE_SERVICE_ACCOUNT_JSON exists: {credentials_json is not None}")
            print(f"🔍 GOOGLE_SERVICE_ACCOUNT_JSON length: {len(credentials_json) if credentials_json else 0}")
            print(f"🔍 TEST_SECRET value: {test_secret}")
            
            if not credentials_json:
                print("❌ Error: GOOGLE_SERVICE_ACCOUNT_JSON environment variable not found")
                return None
            
            credentials_dict = json.loads(credentials_json)
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            client = gspread.authorize(credentials)
            print("✅ Google Sheets client setup successful")
            return client
        
        except Exception as e:
            print(f"❌ Error setting up Google Sheets client: {e}")
            return None
    
    def load_distance_matrix(self):
        """Load distance matrix from Google Sheets CSV export"""
        try:
            print("📊 Loading distance matrix...")
            response = requests.get(self.DISTANCE_MATRIX_URL)
            response.raise_for_status()
            
            # DEBUG: Print first 500 characters of response
            print(f"🔍 Distance matrix CSV preview:")
            print(response.text[:500])
            print("..." if len(response.text) > 500 else "")
            
            # Parse CSV content
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"📊 Distance matrix shape: {df.shape}")
            print(f"📊 Distance matrix columns: {list(df.columns)}")
            
            # DEBUG: Print first few rows
            if len(df) > 0:
                print(f"🔍 First 3 rows of distance matrix:")
                print(df.head(3))
            else:
                print("⚠️ Distance matrix DataFrame is empty!")
            
            # Handle the case where first column contains row Dog IDs and other columns contain column Dog IDs
            distance_matrix = {}
            
            for index, row in df.iterrows():
                row_dog_id = row.iloc[0]  # First column has the row Dog ID
                if pd.isna(row_dog_id):
                    continue
                    
                # Skip invalid row dog IDs
                if not str(row_dog_id).strip():
                    continue
                
                row_dog_id = str(row_dog_id).strip()
                
                for col_idx in range(1, len(row)):
                    col_dog_id = df.columns[col_idx]  # Column name is the column Dog ID
                    distance = row.iloc[col_idx]
                    
                    if pd.isna(distance) or pd.isna(col_dog_id):
                        continue
                    
                    try:
                        distance = float(distance)
                        if distance > 0:  # Only include positive distances
                            distance_matrix[(row_dog_id, str(col_dog_id))] = distance
                    except (ValueError, TypeError):
                        continue
            
            self.distance_matrix = distance_matrix
            print(f"✅ Loaded distance matrix with {len(distance_matrix)} entries")
            
            # Get unique dog IDs from matrix for validation
            matrix_dogs = set()
            for (row_dog, col_dog) in distance_matrix.keys():
                matrix_dogs.add(row_dog)
                matrix_dogs.add(col_dog)
            
            print(f"📊 Matrix dogs: {len(matrix_dogs)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading distance matrix: {e}")
            return False
    
    def load_dog_assignments(self):
        """Load dog assignments and detect callouts from blank Combined column but content in Callout column"""
        try:
            print("🐕 Loading dog assignments...")
            response = requests.get(self.MAP_SHEET_URL)
            response.raise_for_status()
            
            # DEBUG: Print first 500 characters of response
            print(f"🔍 Map sheet CSV preview:")
            print(response.text[:500])
            print("..." if len(response.text) > 500 else "")
            
            # Parse CSV content
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"🐕 Map sheet shape: {df.shape}")
            print(f"🐕 Map sheet columns: {list(df.columns)}")
            
            # DEBUG: Print first few rows
            if len(df) > 0:
                print(f"🔍 First 3 rows of map sheet:")
                print(df.head(3))
            else:
                print("⚠️ Map sheet DataFrame is empty!")
            
            dog_assignments = {}
            callout_dogs = []
            
            for index, row in df.iterrows():
                try:
                    # Column positions based on screenshot:
                    # A=Address(0), B=Dog Name(1), ... G=Name(6), H=Combined(7), I=Group(8), J=Dog ID(9), K=Callout(10)
                    
                    # Get Dog ID from Column J (index 9)
                    if len(row) <= 9:
                        continue
                    dog_id = row.iloc[9]
                    if pd.isna(dog_id) or str(dog_id).strip() == '':
                        continue
                    
                    dog_id = str(dog_id).strip()
                    
                    # Get other data
                    dog_name = row.iloc[1] if len(row) > 1 else ''  # Column B
                    address = row.iloc[0] if len(row) > 0 else ''   # Column A
                    num_dogs = row.iloc[4] if len(row) > 4 else 1   # Column E
                    
                    # Handle num_dogs
                    try:
                        num_dogs = int(float(num_dogs)) if not pd.isna(num_dogs) else 1
                    except (ValueError, TypeError):
                        num_dogs = 1
                    
                    # New callout detection logic:
                    # Combined column (H) blank + Callout column (K) has content = callout
                    combined_col = row.iloc[7] if len(row) > 7 else ''   # Column H (Combined)
                    callout_col = row.iloc[10] if len(row) > 10 else ''  # Column K (Callout)
                    
                    # Simple callout detection
                    combined_is_blank = pd.isna(combined_col) or str(combined_col).strip() == ''
                    callout_has_content = not (pd.isna(callout_col) or str(callout_col).strip() == '')
                    
                    is_callout = combined_is_blank and callout_has_content
                    
                    if is_callout:
                        print(f"🚨 Callout detected: Dog {dog_id} needs assignment for {callout_col}")
                        
                        # Parse callout to get groups (e.g., "Nate:1&2" → groups [1,2])
                        callout_str = str(callout_col).strip()
                        if ':' in callout_str:
                            _, groups_str = callout_str.split(':', 1)
                            groups = self._parse_groups(groups_str)
                        else:
                            # Fallback - try to parse as groups directly
                            groups = self._parse_groups(callout_str)
                        
                        callout_dogs.append({
                            'dog_id': dog_id,
                            'dog_name': dog_name,
                            'address': address,
                            'num_dogs': num_dogs,
                            'original_callout': callout_str,
                            'needed_groups': groups
                        })
                    else:
                        # Regular assignment
                        if not combined_is_blank:
                            # Parse existing assignment from Combined column
                            combined_str = str(combined_col).strip()
                            if ':' in combined_str:
                                driver, groups_str = combined_str.split(':', 1)
                                groups = self._parse_groups(groups_str)
                            else:
                                driver = combined_str
                                groups = []
                            
                            dog_assignments[dog_id] = {
                                'dog_name': dog_name,
                                'address': address,
                                'num_dogs': num_dogs,
                                'current_driver': driver.strip(),
                                'current_groups': groups,
                                'assignment_str': combined_str
                            }
                
                except Exception as e:
                    print(f"⚠️ Error processing row {index}: {e}")
                    continue
            
            self.dog_assignments = dog_assignments
            self.callout_dogs = callout_dogs
            
            print(f"✅ Loaded {len(dog_assignments)} regular assignments")
            print(f"🚨 Found {len(callout_dogs)} callouts that need drivers assigned:")
            for callout in callout_dogs:
                print(f"   - {callout['dog_name']} ({callout['dog_id']}) needs replacement for {callout['original_callout']}")
            
            # Get dog IDs for validation
            assignment_dogs = set(dog_assignments.keys())
            callout_dog_ids = set([c['dog_id'] for c in callout_dogs])
            all_dog_ids = assignment_dogs.union(callout_dog_ids)
            
            print(f"🐕 Assignment dogs: {len(assignment_dogs)}")
            print(f"🚨 Callout dogs: {len(callout_dog_ids)}")
            print(f"📊 Total dogs: {len(all_dog_ids)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dog assignments: {e}")
            return False
    
    def load_driver_capacities(self):
        """Load driver capacities from columns R:W on the map sheet"""
        try:
            print("👥 Loading driver capacities from map sheet columns R:W...")
            response = requests.get(self.MAP_SHEET_URL)
            response.raise_for_status()
            
            # Parse CSV content
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            print(f"👥 Driver data shape: {df.shape}")
            print(f"👥 Driver data columns: {list(df.columns)}")
            
            # DEBUG: Check if we have enough columns
            print(f"🔍 DataFrame has {len(df.columns)} columns, need at least 23 for column W")
            
            # DEBUG: Print column indices we're trying to access
            if len(df) > 0:
                print(f"🔍 Sample row to check column access:")
                sample_row = df.iloc[0]
                print(f"   Row length: {len(sample_row)}")
                if len(sample_row) > 17:
                    print(f"   Column R (17): {sample_row.iloc[17] if not pd.isna(sample_row.iloc[17]) else 'NaN'}")
                if len(sample_row) > 20:
                    print(f"   Column U (20): {sample_row.iloc[20] if not pd.isna(sample_row.iloc[20]) else 'NaN'}")
                if len(sample_row) > 21:
                    print(f"   Column V (21): {sample_row.iloc[21] if not pd.isna(sample_row.iloc[21]) else 'NaN'}")
                if len(sample_row) > 22:
                    print(f"   Column W (22): {sample_row.iloc[22] if not pd.isna(sample_row.iloc[22]) else 'NaN'}")
            
            driver_capacities = {}
            
            for index, row in df.iterrows():
                try:
                    # Column positions: R=Driver(17), U=Group1(20), V=Group2(21), W=Group3(22)
                    if len(row) <= 22:
                        continue
                        
                    driver_name = row.iloc[17]  # Column R (Driver)
                    if pd.isna(driver_name) or str(driver_name).strip() == '':
                        continue
                    
                    driver_name = str(driver_name).strip()
                    
                    # Get capacities for each group
                    group1_cap = row.iloc[20] if len(row) > 20 else 9  # Column U (Group 1)
                    group2_cap = row.iloc[21] if len(row) > 21 else 9  # Column V (Group 2)  
                    group3_cap = row.iloc[22] if len(row) > 22 else 9  # Column W (Group 3)
                    
                    # Convert to integers with fallback
                    try:
                        group1_cap = int(float(group1_cap)) if not pd.isna(group1_cap) else 9
                    except (ValueError, TypeError):
                        group1_cap = 9
                    
                    try:
                        group2_cap = int(float(group2_cap)) if not pd.isna(group2_cap) else 9
                    except (ValueError, TypeError):
                        group2_cap = 9
                    
                    try:
                        group3_cap = int(float(group3_cap)) if not pd.isna(group3_cap) else 9
                    except (ValueError, TypeError):
                        group3_cap = 9
                    
                    driver_capacities[driver_name] = {
                        'group1': group1_cap,
                        'group2': group2_cap,
                        'group3': group3_cap
                    }
                    
                except Exception as e:
                    print(f"⚠️ Error processing driver row {index}: {e}")
                    continue
            
            self.driver_capacities = driver_capacities
            print(f"✅ Loaded capacities for {len(driver_capacities)} drivers")
            
            # Debug: Show some driver capacities
            for driver_name, caps in list(driver_capacities.items())[:5]:
                print(f"   - {driver_name}: Group1={caps['group1']}, Group2={caps['group2']}, Group3={caps['group3']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading driver capacities: {e}")
            return False
    
    def _parse_groups(self, group_str: str) -> List[int]:
        """Parse group string into list of integers (your exact logic)"""
        if not group_str or pd.isna(group_str):
            return []
        
        group_str = str(group_str).strip()
        
        # Handle special case like "2XX2" -> [2]
        if 'XX' in group_str and any(c.isdigit() for c in group_str):
            digits = [int(c) for c in group_str if c.isdigit()]
            return sorted(set(digits))
        
        # Extract all digits from the string
        digits = [int(c) for c in group_str if c.isdigit()]
        return sorted(set(digits))
    
    def get_dogs_to_reassign(self):
        """Find dogs that need driver assignment (callouts detected from blank Combined column but content in Callout column)"""
        return self.callout_dogs
    
    def get_nearby_dogs(self, dog_id, max_distance=3.0):
        """Get nearby dogs within max_distance (your exact logic)"""
        nearby_dogs = []
        
        for (row_dog, col_dog), distance in self.distance_matrix.items():
            if row_dog == dog_id and distance > 0 and distance <= max_distance:
                nearby_dogs.append((col_dog, distance))
            elif col_dog == dog_id and distance > 0 and distance <= max_distance:
                nearby_dogs.append((row_dog, distance))
        
        return nearby_dogs
    
    def is_adjacent_group(self, group, nearby_groups):
        """Check if group is adjacent to any nearby groups (your exact logic)"""
        adjacent_map = {1: [2], 2: [1, 3], 3: [2]}
        adjacent_to_group = adjacent_map.get(group, [])
        return any(adj_group in nearby_groups for adj_group in adjacent_to_group)
    
    def calculate_driver_load(self, driver_name: str) -> Dict[str, int]:
        """Calculate current load for a driver across all groups"""
        load = {'group1': 0, 'group2': 0, 'group3': 0}
        
        for dog_data in self.dog_assignments.values():
            if dog_data['current_driver'] == driver_name:
                num_dogs = dog_data['num_dogs']
                for group in dog_data['current_groups']:
                    group_key = f'group{group}'
                    if group_key in load:
                        load[group_key] += num_dogs
        
        return load
    
    def find_best_reassignment(self, dog_data: Dict, max_iterations=5) -> Dict:
        """Find best reassignment for a callout dog (your exact algorithm)"""
        dog_id = dog_data['dog_id']
        needed_groups = dog_data['needed_groups']
        num_dogs = dog_data['num_dogs']
        
        print(f"🔍 Finding replacement for {dog_data['dog_name']} ({dog_id}) - needs groups {needed_groups}")
        
        for iteration in range(max_iterations):
            print(f"   Iteration {iteration + 1}:")
            
            # Get nearby dogs
            nearby_dogs = self.get_nearby_dogs(dog_id, max_distance=5.0)
            
            if not nearby_dogs:
                print(f"   No nearby dogs found for {dog_id}")
                continue
            
            best_candidate = None
            best_score = -1
            
            for nearby_dog_id, distance in nearby_dogs:
                if nearby_dog_id not in self.dog_assignments:
                    continue
                
                nearby_dog_data = self.dog_assignments[nearby_dog_id]
                candidate_driver = nearby_dog_data['current_driver']
                nearby_groups = nearby_dog_data['current_groups']
                
                # Check if this driver has capacity for the needed groups
                current_load = self.calculate_driver_load(candidate_driver)
                driver_capacity = self.driver_capacities.get(candidate_driver, {'group1': 9, 'group2': 9, 'group3': 9})
                
                # Check capacity constraints
                capacity_ok = True
                for group in needed_groups:
                    group_key = f'group{group}'
                    current = current_load.get(group_key, 0)
                    capacity = driver_capacity.get(group_key, 9)
                    if current + num_dogs > capacity:
                        capacity_ok = False
                        break
                
                if not capacity_ok:
                    continue
                
                # Calculate score with iteration-based thresholds (your exact logic)
                same_group_threshold = 0.5 + (iteration * 0.5)
                adjacent_group_threshold = 0.25 + (iteration * 0.25)
                
                score = 0
                
                # Same group match (highest priority)
                for group in needed_groups:
                    if group in nearby_groups:
                        score += 10
                        break
                
                # Adjacent group match
                if score == 0:  # Only if no same group match
                    for group in needed_groups:
                        if self.is_adjacent_group(group, nearby_groups):
                            score += 5
                            break
                
                # Apply thresholds
                if score >= 10 and score < same_group_threshold * 10:
                    continue
                elif score >= 5 and score < adjacent_group_threshold * 10:
                    continue
                
                # Distance penalty
                score -= distance * 2
                
                # Load penalty
                total_load = sum(current_load.values())
                load_penalty = total_load * 0.1
                final_score = score - load_penalty
                
                if final_score > best_score:
                    best_score = final_score
                    best_candidate = {
                        'to_driver': candidate_driver,
                        'to_groups': needed_groups,
                        'distance': distance,
                        'score': final_score
                    }
            
            if best_candidate:
                print(f"   ✅ Found assignment: {best_candidate['to_driver']} (score: {best_candidate['score']:.2f})")
                return best_candidate
            
            print(f"   No suitable candidates in iteration {iteration + 1}")
        
        print(f"   ❌ No assignment found after {max_iterations} iterations")
        return None
    
    def _get_reassignment_priority(self, dog_data: Dict) -> int:
        """Calculate priority for reassignment (your exact logic)"""
        num_dogs = dog_data['num_dogs']
        num_groups = len(dog_data['needed_groups'])
        
        if num_dogs == 1 and num_groups == 1:
            return 1  # Highest priority
        elif num_dogs == 1 and num_groups > 1:
            return 2
        elif num_dogs > 1 and num_groups == 1:
            return 3
        else:
            return 4  # Lowest priority
    
    def reassign_dogs(self):
        """Main reassignment algorithm (your exact logic)"""
        print("\n🔄 Starting dog reassignment process...")
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts found - all dogs have drivers assigned!")
            return []
        
        print(f"🚨 Found {len(dogs_to_reassign)} callouts to process")
        
        # Sort by priority (your exact logic)
        dogs_to_reassign.sort(key=lambda x: self._get_reassignment_priority(x))
        
        successful_reassignments = []
        max_iterations = 5
        iteration = 0
        
        while dogs_to_reassign and iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Reassignment iteration {iteration}")
            
            dogs_assigned_this_iteration = []
            
            for dog_data in dogs_to_reassign:
                reassignment = self.find_best_reassignment(dog_data)
                
                if reassignment:
                    # Create full reassignment record
                    full_reassignment = {
                        'dog_id': dog_data['dog_id'],
                        'dog_name': dog_data['dog_name'],
                        'address': dog_data['address'],
                        'num_dogs': dog_data['num_dogs'],
                        'original_callout': dog_data['original_callout'],
                        'to_driver': reassignment['to_driver'],
                        'to_groups': reassignment['to_groups'],
                        'distance': reassignment['distance'],
                        'score': reassignment['score']
                    }
                    
                    successful_reassignments.append(full_reassignment)
                    dogs_assigned_this_iteration.append(dog_data)
                    
                    print(f"✅ Assigned {dog_data['dog_name']} → {reassignment['to_driver']}:{{'&'.join(map(str, reassignment['to_groups']))}}")
            
            # Remove successfully assigned dogs
            for dog in dogs_assigned_this_iteration:
                dogs_to_reassign.remove(dog)
            
            if not dogs_assigned_this_iteration:
                print("No more assignments possible in this iteration")
                break
        
        print(f"\n📊 Reassignment Summary:")
        print(f"   ✅ Successfully assigned: {len(successful_reassignments)}")
        print(f"   ❌ Could not assign: {len(dogs_to_reassign)}")
        
        if dogs_to_reassign:
            print(f"   Unassigned dogs:")
            for dog in dogs_to_reassign:
                print(f"      - {dog['dog_name']} ({dog['dog_id']}) - needed {dog['original_callout']}")
        
        return successful_reassignments
    
    def update_combined_column(self, reassignments: List[Dict], spreadsheet) -> bool:
        """Write new assignments to Combined column (H) in Google Sheets"""
        try:
            if not reassignments:
                print("ℹ️ No reassignments to write")
                return True
            
            print(f"📝 Writing {len(reassignments)} assignments to Combined column...")
            
            # Try different sheet names that might exist
            sheet_names_to_try = ["New districts Map 8", "Map", "Districts Map", "Sheet1"]
            sheet = None
            
            for sheet_name in sheet_names_to_try:
                try:
                    sheet = spreadsheet.worksheet(sheet_name)
                    print(f"✅ Found sheet: {sheet_name}")
                    break
                except:
                    continue
            
            if not sheet:
                print("❌ Could not find map sheet")
                return False
            
            # Get all data to find row numbers for dog IDs
            all_data = sheet.get_all_values()
            
            # Find Dog ID column (should be J = column 10)
            dog_id_col = 10  # Column J (1-based)
            
            # Prepare batch updates
            updates = []
            
            for r in reassignments:
                dog_id = r['dog_id']
                new_assignment = f"{r['to_driver']}:{'&'.join(map(str, r['to_groups']))}"
                
                # Find the row for this dog (Dog ID is in column J = index 9)
                row_number = None
                for row_idx, row in enumerate(all_data):
                    if len(row) > 9 and str(row[9]).strip() == str(dog_id).strip():
                        row_number = row_idx + 1  # Convert to 1-based
                        break
                
                if row_number:
                    # Combined column is H = column 8 (1-based)
                    updates.append({
                        'range': f'H{row_number}',
                        'values': [[new_assignment]]
                    })
                    print(f"   📝 Will update row {row_number}: {dog_id} → {new_assignment}")
                else:
                    print(f"   ⚠️ Could not find row for dog {dog_id}")
            
            if updates:
                # Batch update all changes
                sheet.batch_update(updates)
                print(f"✅ Successfully updated {len(updates)} assignments in Combined column")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating Combined column: {e}")
            return False
    
    def write_results_to_sheets(self, reassignments: List[Dict]) -> bool:
        """Write reassignment results back to Google Sheets"""
        try:
            if not reassignments:
                print("ℹ️ No callout assignments to write back")
                return True
            
            print("📤 Writing results to Google Sheets...")
            
            client = self.setup_google_sheets_client()
            if not client:
                return False
            
            spreadsheet = client.open_by_key(self.MAP_SPREADSHEET_ID)
            
            # Update the Combined column with new assignments
            success = self.update_combined_column(reassignments, spreadsheet)
            
            if success:
                print("✅ Successfully wrote callout assignments to Google Sheets")
                
                # Send notification if webhook available
                webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
                if webhook_url:
                    self.send_slack_notification(reassignments, webhook_url)
            
            return success
            
        except Exception as e:
            print(f"❌ Error writing results to sheets: {e}")
            return False
    
    def send_slack_notification(self, reassignments: List[Dict], webhook_url: str):
        """Send Slack notification about callout assignments"""
        try:
            if not reassignments:
                return
            
            message = f"🚨 *Dog Callout Assignments Complete* 🚨\n\n"
            message += f"Assigned drivers for {len(reassignments)} callouts:\n\n"
            
            for r in reassignments:
                groups_str = '&'.join(map(str, r['to_groups']))
                message += f"• {r['dog_name']} ({r['dog_id']}) → {r['to_driver']}:{groups_str}\n"
            
            message += f"\n✅ All assignments updated in Google Sheets"
            
            payload = {"text": message}
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code == 200:
                print("✅ Slack notification sent")
            else:
                print(f"⚠️ Slack notification failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Error sending Slack notification: {e}")


def main():
    """Main function for GitHub Actions"""
    print("🚀 Production Dog Reassignment System")
    print("=" * 50)
    
    # Check M1 trigger value if provided
    trigger_value = os.environ.get('TRIGGER_VALUE')
    if trigger_value:
        print(f"🎯 Triggered by M1 cell change: {trigger_value}")
    
    # Initialize system
    system = ProductionDogReassignmentSystem()
    
    # Load all data
    print("\n📊 Loading data from Google Sheets...")
    
    if not system.load_distance_matrix():
        print("❌ Failed to load distance matrix")
        return
    
    if not system.load_dog_assignments():
        print("❌ Failed to load dog assignments")
        return
    
    if not system.load_driver_capacities():
        print("❌ Failed to load driver capacities")
        return
    
    # Data validation
    print("\n🔍 Data validation:")
    matrix_dogs = set()
    for (row_dog, col_dog) in system.distance_matrix.keys():
        matrix_dogs.add(row_dog)
        matrix_dogs.add(col_dog)
    
    assignment_dogs = set(system.dog_assignments.keys())
    callout_dogs = set([c['dog_id'] for c in system.callout_dogs])
    all_dogs = assignment_dogs.union(callout_dogs)
    
    matching_dogs = matrix_dogs.intersection(all_dogs)
    
    print(f"   Matrix dogs: {len(matrix_dogs)}")
    print(f"   Assignment dogs: {len(assignment_dogs)}")
    print(f"   Callout dogs: {len(callout_dogs)}")
    print(f"   Matching dogs: {len(matching_dogs)}")
    
    if len(matching_dogs) == 0:
        print("❌ NO MATCHING DOG IDs! Check that Dog IDs match between sheets.")
        print("Error: Process completed with exit code 1.")
        return
    
    # Check for callouts
    if not system.callout_dogs:
        print("\n✅ No callouts detected - all dogs have drivers assigned!")
        print("🎯 System ready - will process callouts when M1 changes")
        return
    
    # Run your exact reassignment logic
    print("\n🔄 Processing callout assignments...")
    reassignments = system.reassign_dogs()
    
    # Write results back to Google Sheets
    if system.write_results_to_sheets(reassignments):
        print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments")
    else:
        print(f"\n❌ Failed to write results to Google Sheets")


if __name__ == "__main__":
    main()
