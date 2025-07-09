# production_reassignment.py
# Your EXACT working logic from Streamlit, adapted for GitHub Actions → Google Sheets
# All algorithms preserved exactly as you built them

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
    Only the I/O changed (GitHub + Google Sheets instead of Streamlit)
    """
    
    def __init__(self):
        # Your exact configuration
        self.MAX_REASSIGNMENT_DISTANCE = 5.0
        self.distance_matrix = {}
        self.dogs_going_today = {}
        self.driver_capacities = {}
        self.driver_callouts = {}  # Keep for compatibility but won't be used
        self.callout_dogs = []  # NEW: Store dogs that need driver assignment
        self.reassignments = []
        
        # Google Sheets URLs - YOUR ACTUAL URLS
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=2146002137"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        self.DRIVER_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=1359695250"
        
        # For writing results back - YOUR ACTUAL SPREADSHEET ID
        self.MAP_SPREADSHEET_ID = "1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0"
        
        # Initialize Google Sheets connection
        self.setup_google_sheets()

    def setup_google_sheets(self):
        """Setup Google Sheets API connection using GitHub secrets"""
        try:
            credentials_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            if not credentials_json:
                print("❌ GOOGLE_SERVICE_ACCOUNT_JSON environment variable not found")
                self.sheets_client = None
                return
            
            credentials_dict = json.loads(credentials_json)
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"
                ]
            )
            
            self.sheets_client = gspread.authorize(credentials)
            print("✅ Connected to Google Sheets API")
            
        except Exception as e:
            print(f"❌ Error setting up Google Sheets: {e}")
            self.sheets_client = None

    def load_distance_matrix(self):
        """YOUR EXACT distance matrix loading logic from the 'new' file"""
        try:
            print("📊 Loading distance matrix...")
            
            response = requests.get(self.DISTANCE_MATRIX_URL)
            df = pd.read_csv(pd.io.common.StringIO(response.text))
            
            print(f"🔍 Distance Matrix Shape: {df.shape}")
            
            # YOUR EXACT FORMAT: First column has row Dog IDs, other columns have column Dog IDs
            # Get column Dog IDs (skip first column which is ":" or similar)
            column_dog_ids = []
            for col in df.columns[1:]:  # Skip first column
                if pd.notna(col) and str(col).strip() != '':
                    clean_id = str(col).strip()
                    column_dog_ids.append(clean_id)
            
            print(f"🔍 Found {len(column_dog_ids)} column Dog IDs")
            
            # Process each row to build distance matrix
            rows_processed = 0
            for idx, row in df.iterrows():
                # Get row Dog ID from first column (whatever it's named)
                row_dog_id = row.iloc[0]  # First column by position
                
                if pd.isna(row_dog_id) or str(row_dog_id).strip() == '':
                    continue
                    
                row_dog_id = str(row_dog_id).strip()
                
                # Skip if this doesn't look like a Dog ID
                if not row_dog_id or 'x' not in row_dog_id.lower():
                    continue
                
                # Initialize this dog's distances
                self.distance_matrix[row_dog_id] = {}
                
                # Get distances for each column Dog ID
                for col_idx, col_dog_id in enumerate(column_dog_ids):
                    try:
                        # Get distance value from corresponding column (skip first column)
                        distance_val = row.iloc[col_idx + 1]
                        
                        if pd.isna(distance_val):
                            distance = 0.0
                        else:
                            distance = float(str(distance_val).strip())
                    except (ValueError, TypeError):
                        distance = 0.0
                    
                    self.distance_matrix[row_dog_id][col_dog_id] = distance
                
                rows_processed += 1
                
                # Progress for large files
                if rows_processed % 200 == 0:
                    print(f"  Processed {rows_processed} dogs...")
            
            print(f"✅ Loaded distance matrix for {len(self.distance_matrix)} dogs")
            
            # Show sample
            if len(self.distance_matrix) > 0:
                first_dog = list(self.distance_matrix.keys())[0]
                sample_distances = list(self.distance_matrix[first_dog].items())[:3]
                print(f"📏 Sample distances for {first_dog}: {sample_distances}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading distance matrix: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return False
    
    def load_dog_assignments(self):
        """Load dog assignments and detect callouts from Name column (G) being blank but gropu column (I) not blank"""
        try:
            response = requests.get(self.MAP_SHEET_URL)
            df = pd.read_csv(pd.io.common.StringIO(response.text))
            
            print(f"🔍 Map Sheet Shape: {df.shape}")
            print(f"🔍 Map Sheet Columns: {list(df.columns)}")
            print(f"🔍 Column mapping:")
            for i, col in enumerate(df.columns):
                print(f"   Column {chr(65+i)} (index {i}): '{col}'")
            
            assignments_found = 0
            callouts_found = 0
            callout_entries = []
            
            for _, row in df.iterrows():
                # Get Dog ID 
                dog_id = row.get('Dog ID')
                if pd.isna(dog_id) or str(dog_id).strip() == '':
                    continue
                dog_id = str(dog_id).strip()
                
                # Get Name column (Column G) and gropu column (Column I)
                name_col = row.get('Name', '')  # Column G
                gropu_col = row.get('gropu', '')  # Column I - the group assignment
                
                # Check for callout: Name is blank but gropu is not blank
                name_is_blank = pd.isna(name_col) or str(name_col).strip() == ''
                gropu_not_blank = not (pd.isna(gropu_col) or str(gropu_col).strip() == '')
                
                is_callout = name_is_blank and gropu_not_blank
                
                if is_callout:
                    # This is a callout - needs to be filled
                    callouts_found += 1
                    group_assignment = str(gropu_col).strip()
                    
                    callout_entries.append({
                        'dog_id': dog_id,
                        'group_assignment': group_assignment,
                        'dog_name': str(row.get('Dog Name', '')).strip(),
                        'address': str(row.get('Address', '')).strip()
                    })
                    
                    print(f"🚨 Callout detected: {dog_id} ({row.get('Dog Name', '')}) needs driver for groups {group_assignment}")
                    continue
                
                # Regular assignment - get Today column
                assignment = row.get('Today')
                if pd.isna(assignment) or str(assignment).strip() == '':
                    continue
                assignment = str(assignment).strip()
                
                # Skip if no valid assignment or if it's not a driver assignment
                if not assignment or ':' not in assignment:
                    continue
                
                # Get number of dogs
                num_dogs = 1
                if 'Number of dogs' in row and not pd.isna(row['Number of dogs']):
                    try:
                        num_dogs = int(float(row['Number of dogs']))
                    except (ValueError, TypeError):
                        num_dogs = 1
                
                # Store regular assignment
                self.dogs_going_today[dog_id] = {
                    'assignment': assignment,
                    'num_dogs': num_dogs,
                    'address': str(row.get('Address', '')).strip(),
                    'dog_name': str(row.get('Dog Name', '')).strip(),
                    'is_callout': False
                }
                assignments_found += 1
            
            # Store callouts separately for processing
            self.callout_dogs = callout_entries
            
            print(f"✅ Loaded {assignments_found} regular dog assignments")
            if callouts_found > 0:
                print(f"🚨 Found {callouts_found} callouts that need drivers assigned:")
                for callout in callout_entries[:5]:  # Show first 5
                    print(f"   - {callout['dog_name']} ({callout['dog_id']}) needs groups {callout['group_assignment']}")
                if len(callout_entries) > 5:
                    print(f"   ... and {len(callout_entries) - 5} more")
            else:
                print("✅ No callouts found - all dogs have drivers assigned")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dog assignments: {e}")
            return False
    
    def load_driver_capacities(self):
        """Load driver capacities - simplified since callouts are now detected from main sheet"""
        try:
            response = requests.get(self.DRIVER_SHEET_URL)
            df = pd.read_csv(pd.io.common.StringIO(response.text))
            
            print(f"🔍 Driver Sheet Shape: {df.shape}")
            print(f"🔍 Driver Sheet Columns: {list(df.columns)}")
            
            drivers_loaded = 0
            
            for _, row in df.iterrows():
                # Get driver name - exact column name from your file
                driver = row.get('Driver')
                if pd.isna(driver) or str(driver).strip() == '':
                    continue
                driver = str(driver).strip()
                
                # Get group capacities - exact column names from your file
                group1 = str(row.get('Group 1', '')).strip().upper()
                group2 = str(row.get('Group 2', '')).strip().upper()
                group3 = str(row.get('Group 3', '')).strip().upper()
                
                # Parse capacities (number = capacity, empty = default 9)
                def parse_capacity(val):
                    if val == '' or val == 'NAN':
                        return 9  # Default capacity
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return 9
                
                self.driver_capacities[driver] = {
                    'group1': parse_capacity(group1),
                    'group2': parse_capacity(group2), 
                    'group3': parse_capacity(group3)
                }
                
                drivers_loaded += 1
            
            print(f"✅ Loaded capacities for {drivers_loaded} drivers")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading driver capacities: {e}")
            return False
    
    def parse_group_assignment(self, assignment):
        """YOUR EXACT group assignment parsing logic from the 'new' file"""
        if ':' not in assignment:
            return None, []
        
        parts = assignment.split(':')
        driver = parts[0].strip()
        group_part = parts[1].strip()
        
        # Handle special cases like "XX" - ignore them
        if 'XX' in group_part.upper():
            return None, []
        
        # Parse groups (1, 2, 3, 1&2, 2&3, 1&2&3, etc.)
        groups = []
        if '&' in group_part:
            for g in group_part.split('&'):
                try:
                    group_num = int(g.strip())
                    if 1 <= group_num <= 3:
                        groups.append(group_num)
                except ValueError:
                    continue
        else:
            # Handle single group or extract numbers
            import re
            numbers = re.findall(r'\d+', group_part)
            for num_str in numbers:
                try:
                    group_num = int(num_str)
                    if 1 <= group_num <= 3:
                        groups.append(group_num)
                except ValueError:
                    continue
        
        return driver, sorted(list(set(groups)))  # Remove duplicates and sort
    
    def get_dogs_to_reassign(self):
        """Find dogs that need driver assignment (callouts detected from blank Name column)"""
        dogs_to_reassign = []
        
        for callout in self.callout_dogs:
            dog_id = callout['dog_id']
            group_assignment = callout['group_assignment']
            
            # Parse the group assignment from column I
            groups = self._parse_groups(group_assignment)
            
            if not groups:
                continue
            
            dogs_to_reassign.append({
                'dog_id': dog_id,
                'original_driver': None,  # No driver assigned yet
                'original_groups': groups,
                'affected_groups': groups,  # All groups need assignment
                'dog_info': {
                    'assignment': f"NEEDS_DRIVER:{group_assignment}",
                    'num_dogs': 1,  # Default to 1 for callouts
                    'dog_name': callout['dog_name'],
                    'address': callout['address']
                }
            })
        
        return dogs_to_reassign
    
    def _parse_groups(self, group_str: str) -> List[int]:
        """Parse group string into list of integers"""
        # Handle special cases like "2XX2" -> [2]
        if 'XX' in group_str and any(c.isdigit() for c in group_str):
            digits = [int(c) for c in group_str if c.isdigit()]
            return sorted(set(digits))
        
        # Regular case: extract all digits
        digits = [int(c) for c in group_str if c.isdigit()]
        return sorted(set(digits))
    
    def get_nearby_dogs(self, dog_id, max_distance=3.0):
        """YOUR EXACT nearby dogs logic from the 'new' file"""
        if dog_id not in self.distance_matrix:
            return []
        
        nearby = []
        for other_dog_id, distance in self.distance_matrix[dog_id].items():
            if distance > 0 and distance <= max_distance:
                nearby.append({
                    'dog_id': other_dog_id,
                    'distance': distance
                })
        
        return sorted(nearby, key=lambda x: x['distance'])
    
    def get_current_driver_loads(self):
        """YOUR EXACT driver load calculation from the 'new' file"""
        driver_loads = {}
        
        for dog_id, dog_info in self.dogs_going_today.items():
            assignment = dog_info['assignment']
            driver, groups = self.parse_group_assignment(assignment)
            
            if not driver or not groups:
                continue
            
            if driver not in driver_loads:
                driver_loads[driver] = {'group1': 0, 'group2': 0, 'group3': 0}
            
            num_dogs = dog_info['num_dogs']
            for group in groups:
                if group == 1:
                    driver_loads[driver]['group1'] += num_dogs
                elif group == 2:
                    driver_loads[driver]['group2'] += num_dogs
                elif group == 3:
                    driver_loads[driver]['group3'] += num_dogs
        
        return driver_loads
    
    def find_best_reassignment(self, dog_to_reassign, iteration=0):
        """YOUR EXACT best reassignment logic, adapted for callouts (no original driver)"""
        dog_id = dog_to_reassign['dog_id']
        original_groups = dog_to_reassign['original_groups']
        num_dogs = dog_to_reassign['dog_info']['num_dogs']
        dog_name = dog_to_reassign['dog_info']['dog_name']
        original_driver = dog_to_reassign.get('original_driver')  # May be None for callouts
        
        # Get nearby dogs
        nearby_dogs = self.get_nearby_dogs(dog_id, max_distance=3.0)
        current_loads = self.get_current_driver_loads()
        
        candidates = []
        
        # Adjust thresholds by iteration
        same_group_threshold = 0.5 + (iteration * 0.5)
        adjacent_group_threshold = 0.25 + (iteration * 0.25)
        
        for nearby in nearby_dogs:
            nearby_dog_id = nearby['dog_id']
            distance = nearby['distance']
            
            if nearby_dog_id not in self.dogs_going_today:
                continue
            
            nearby_assignment = self.dogs_going_today[nearby_dog_id]['assignment']
            nearby_driver, nearby_groups = self.parse_group_assignment(nearby_assignment)
            
            if not nearby_driver or not nearby_groups:
                continue
            
            # Skip if it's the same driver that called out (only if we had an original driver)
            if original_driver and nearby_driver == original_driver:
                continue
            
            # For callouts, we don't need to check if nearby driver is calling out
            # since we're just looking for available drivers
            
            # Check for group compatibility
            score = 0
            compatible_groups = []
            
            for group in original_groups:
                # Same group match (higher score, stricter distance)
                if group in nearby_groups and distance <= same_group_threshold:
                    score += 10
                    compatible_groups.append(group)
                # Adjacent group match (lower score, even stricter distance)
                elif self.is_adjacent_group(group, nearby_groups) and distance <= adjacent_group_threshold:
                    score += 5
                    compatible_groups.append(group)
            
            # Must match ALL original groups
            if len(compatible_groups) != len(original_groups):
                continue
            
            # Check capacity constraints
            driver_capacity = self.driver_capacities.get(nearby_driver, {'group1': 9, 'group2': 9, 'group3': 9})
            current_load = current_loads.get(nearby_driver, {'group1': 0, 'group2': 0, 'group3': 0})
            
            capacity_ok = True
            for group in original_groups:
                current = current_load.get(f'group{group}', 0)
                capacity = driver_capacity.get(f'group{group}', 9)
                if current + num_dogs > capacity:
                    capacity_ok = False
                    break
            
            if not capacity_ok:
                continue
            
            # Calculate final score
            load_penalty = sum(current_load.values())
            final_score = score - (distance * 2) - (load_penalty * 0.1)
            
            candidates.append({
                'driver': nearby_driver,
                'groups': original_groups,
                'distance': distance,
                'score': final_score,
                'current_load': dict(current_load)
            })
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[0] if candidates else None
    
    def is_adjacent_group(self, group, nearby_groups):
        """YOUR EXACT adjacent group logic from the 'new' file"""
        adjacent_map = {
            1: [2],
            2: [1, 3], 
            3: [2]
        }
        
        adjacent_to_group = adjacent_map.get(group, [])
        return any(adj_group in nearby_groups for adj_group in adjacent_to_group)
    
    def reassign_dogs(self, max_iterations=5):
        """Main assignment logic - assign drivers to callouts and handle any reassignments"""
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts need driver assignment.")
            return []
        
        print(f"🔄 Found {len(dogs_to_reassign)} callouts that need driver assignment")
        
        reassignments = []
        iteration = 0
        
        while dogs_to_reassign and iteration < max_iterations:
            print(f"Processing iteration {iteration + 1}/{max_iterations}...")
            
            successful_reassignments = 0
            
            for dog_data in dogs_to_reassign[:]:  # Copy to avoid modification during iteration
                best_match = self.find_best_reassignment(dog_data, iteration)
                
                if best_match:
                    dog_id = dog_data['dog_id']
                    new_driver = best_match['driver']
                    new_groups = best_match['groups']
                    
                    # Create new assignment string
                    if len(new_groups) == 1:
                        new_assignment = f"{new_driver}:{new_groups[0]}"
                    else:
                        new_assignment = f"{new_driver}:{'&'.join(map(str, sorted(new_groups)))}"
                    
                    # Add this assignment to dogs_going_today for future proximity checks
                    self.dogs_going_today[dog_id] = {
                        'assignment': new_assignment,
                        'num_dogs': dog_data['dog_info']['num_dogs'],
                        'address': dog_data['dog_info']['address'],
                        'dog_name': dog_data['dog_info']['dog_name'],
                        'is_callout': True
                    }
                    
                    original_driver = dog_data.get('original_driver', 'UNASSIGNED')
                    
                    reassignments.append({
                        'dog_id': dog_id,
                        'dog_name': dog_data['dog_info']['dog_name'],
                        'from_driver': original_driver,
                        'to_driver': new_driver,
                        'from_groups': dog_data['original_groups'],
                        'to_groups': new_groups,
                        'distance': best_match['distance'],
                        'reason': f"Callout assignment - driver needed for groups {dog_data['original_groups']}"
                    })
                    
                    dogs_to_reassign.remove(dog_data)
                    successful_reassignments += 1
                    
                    print(f"  ✅ Assigned {dog_data['dog_info']['dog_name']} → {new_assignment}")
            
            if successful_reassignments == 0:
                print(f"No more assignments possible after iteration {iteration + 1}")
                break
            
            iteration += 1
        
        print("Processing complete!")
        
        # Report results
        if reassignments:
            print(f"✅ Successfully assigned drivers to {len(reassignments)} callouts!")
        
        if dogs_to_reassign:
            print(f"⚠️ {len(dogs_to_reassign)} callouts could not be assigned:")
            for dog_data in dogs_to_reassign:
                dog_name = dog_data['dog_info']['dog_name']
                dog_id = dog_data['dog_id']
                groups = '&'.join(map(str, dog_data['original_groups']))
                print(f"  - {dog_name} (ID: {dog_id}) needs driver for groups {groups}")
        
        return reassignments

    def write_results_to_sheets(self, reassignments: List[Dict]) -> bool:
        """Write results back to Google Sheets"""
        if not self.sheets_client:
            print("❌ No Google Sheets connection")
            return False
        
        try:
            # Open the spreadsheet
            spreadsheet = self.sheets_client.open_by_key(self.MAP_SPREADSHEET_ID)
            
            # Create or update results sheet
            try:
                results_sheet = spreadsheet.worksheet("Reassignment Results")
                results_sheet.clear()
            except gspread.WorksheetNotFound:
                results_sheet = spreadsheet.add_worksheet(title="Reassignment Results", rows=1000, cols=12)
            
            if not reassignments:
                results_sheet.update('A1', [['No callouts found - all dogs have drivers assigned']])
                print("✅ Updated Google Sheets: No callouts found")
                return True
            
            # Prepare data with your exact format
            headers = [
                'Timestamp',
                'Dog ID', 
                'Dog Name',
                'From Driver',
                'To Driver', 
                'From Groups',
                'To Groups',
                'New Assignment',
                'Distance (miles)',
                'Reason'
            ]
            
            rows = [headers]
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for r in reassignments:
                rows.append([
                    timestamp,
                    r['dog_id'],
                    r['dog_name'],
                    r['from_driver'],
                    r['to_driver'],
                    '&'.join(map(str, r['from_groups'])),
                    '&'.join(map(str, r['to_groups'])),
                    f"{r['to_driver']}:{'&'.join(map(str, r['to_groups']))}",
                    f"{r['distance']:.2f}",
                    r['reason']
                ])
            
            # Write to sheet
            results_sheet.update('A1', rows)
            
            # Format header
            results_sheet.format('A1:J1', {
                'backgroundColor': {'red': 0.8, 'green': 0.9, 'blue': 1.0},
                'textFormat': {'bold': True}
            })
            
            print(f"✅ Wrote {len(reassignments)} results to Google Sheets")
            
            # Update the main Map sheet with new assignments
            self.update_map_sheet_today_column(reassignments, spreadsheet)
            
            return True
            
        except Exception as e:
            print(f"❌ Error writing to Google Sheets: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return False

    def update_map_sheet_today_column(self, reassignments: List[Dict], spreadsheet) -> bool:
        """Update the Name column (Column G) in the Map sheet with assigned drivers"""
        try:
            # Try different sheet names that might exist
            sheet_names_to_try = ["New districts Map 8", "Map", "Districts Map"]
            map_sheet = None
            
            for sheet_name in sheet_names_to_try:
                try:
                    map_sheet = spreadsheet.worksheet(sheet_name)
                    print(f"✅ Found map sheet: {sheet_name}")
                    break
                except gspread.WorksheetNotFound:
                    continue
            
            if not map_sheet:
                print("⚠️ Could not find map sheet to update")
                return False
            
            # Get all data to find the right rows
            all_data = map_sheet.get_all_records()
            
            updates = []
            for r in reassignments:
                dog_id = r['dog_id']
                new_driver = r['to_driver']  # Just the driver name for Name column
                
                # Find the row for this dog
                for i, row in enumerate(all_data):
                    if str(row.get('Dog ID', '')).strip() == str(dog_id).strip():
                        row_number = i + 2  # +2 for header and 0-based index
                        
                        # Find Name column (should be column G = 7th column)
                        try:
                            name_col = list(row.keys()).index('Name') + 1
                            col_letter = chr(64 + name_col)  # Convert to letter
                            
                            updates.append({
                                'range': f'{col_letter}{row_number}',
                                'values': [[new_driver]]
                            })
                            
                        except ValueError:
                            print(f"⚠️ 'Name' column not found for Dog ID {dog_id}")
                        break
            
            if updates:
                map_sheet.batch_update(updates)
                print(f"✅ Updated {len(updates)} driver assignments in Name column!")
                
                # Print what was updated for verification
                print("\n📝 Updates made to Name column:")
                for r in reassignments:
                    print(f"   {r['dog_id']} ({r['dog_name']}): {r['to_driver']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating Map sheet: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return False

    def send_notification(self, reassignments: List[Dict]):
        """Send notification if there were callout assignments"""
        if not reassignments:
            return
        
        # Slack notification (optional)
        slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
        if slack_webhook:
            try:
                message = f"🐕 Driver Assignment Alert: {len(reassignments)} callouts assigned drivers"
                details = "\n".join([f"• {r['dog_name']}: assigned to {r['to_driver']}" for r in reassignments[:5]])
                if len(reassignments) > 5:
                    details += f"\n... and {len(reassignments) - 5} more"
                
                payload = {
                    'text': message,
                    'blocks': [
                        {
                            'type': 'section',
                            'text': {'type': 'mrkdwn', 'text': f"*{message}*\n{details}"}
                        }
                    ]
                }
                requests.post(slack_webhook, json=payload)
                print("✅ Slack notification sent")
            except Exception as e:
                print(f"⚠️ Could not send Slack notification: {e}")

def main():
    """Main function for GitHub Actions"""
    print("🚀 Production Dog Reassignment System")
    print("=" * 50)
    
    # Check M1 trigger value if provided
    trigger_value = os.environ.get('TRIGGER_VALUE')
    if trigger_value:
        print(f"🎯 Triggered by M1 change: '{trigger_value}'")
        
        # Optional: Add conditional logic based on M1 value
        # if trigger_value.lower() == 'skip':
        #     print("⏭️ M1 says 'skip' - exiting without processing")
        #     exit(0)
    
    # Initialize system with your exact logic
    system = ProductionDogReassignmentSystem()
    
    # Load data with your exact methods
    print("📊 Loading data from Google Sheets...")
    
    if not system.load_distance_matrix():
        exit(1)
    
    if not system.load_dog_assignments():
        exit(1)
    
    if not system.load_driver_capacities():
        exit(1)
    
    # Data validation (your exact validation logic)
    print("\n🔍 Data validation:")
    matrix_ids = set(system.distance_matrix.keys())
    assignment_ids = set(system.dogs_going_today.keys())
    matching_ids = matrix_ids.intersection(assignment_ids)
    
    print(f"   Matrix dogs: {len(matrix_ids)}")
    print(f"   Assignment dogs: {len(assignment_ids)}")
    print(f"   Matching dogs: {len(matching_ids)}")
    
    if len(matching_ids) == 0:
        print("❌ NO MATCHING DOG IDs! Check that Dog IDs match between sheets.")
        exit(1)
    
    print(f"✅ Found {len(matching_ids)} matching Dog IDs")
    sample_matches = list(matching_ids)[:10]
    print(f"Sample matching IDs: {sample_matches}")
    
    # Check if there are actually callouts
    dogs_to_reassign = system.get_dogs_to_reassign()
    
    if not dogs_to_reassign:
        print("✅ No callouts today - all dogs have drivers assigned")
        # Still write to sheets to show the system ran
        if system.sheets_client:
            system.write_results_to_sheets([])
        exit(0)
    
    # Run your exact reassignment logic
    print("\n🔄 Processing callout assignments...")
    reassignments = system.reassign_dogs()
    
    # Write results back to Google Sheets
    if system.write_results_to_sheets(reassignments):
        system.send_notification(reassignments)
        print(f"\n🎉 Successfully assigned drivers to {len(reassignments)} callouts!")
        
        # Print summary
        if reassignments:
            print("\n📋 Summary of driver assignments:")
            for r in reassignments:
                print(f"   {r['dog_name']} ({r['dog_id']}): assigned to {r['to_driver']}:{r['to_groups']} ({r['distance']:.2f} mi)")
    else:
        print("❌ Failed to write results to Google Sheets")
        exit(1)

if __name__ == "__main__":
    main()
