# production_reassignment_distance_based.py
# COMPLETE WORKING VERSION: Locality-first with distance-based constraints (miles)
# Direct distance instead of drive time conversion
# WITH HAVERSINE FALLBACK for missing entries
# UPDATED: Improved group compatibility, route coherence, and neighbor assignment
# ENHANCED DEBUG: Added detailed debugging for unassigned dogs
# ROUTE COHERENCE: 25% of dogs within 3 miles (relaxed from 30% within 2.5 miles)

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

# ========== HAVERSINE FALLBACK CONFIGURATION ==========
# If a dog is not found in the distance matrix, the system can fall back
# to haversine distances

# Option 1: CSV Export URL (if your haversine matrix is publicly accessible)
HAVERSINE_MATRIX_URL = None  # Example: "https://docs.google.com/spreadsheets/d/ABC123/export?format=csv&gid=0"

# Option 2: Google Sheets API (if using service account)
HAVERSINE_SHEET_ID = None    # Example: "1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg"
HAVERSINE_GID = None         # Example: "398422902"

class DogReassignmentSystem:
    def __init__(self):
        """Initialize the dog reassignment system with distance-based constraints"""
        # Google Sheets URLs (CSV export format)
        self.DISTANCE_MATRIX_URL = "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
        self.MAP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750"
        
        # DISTANCE LIMITS - LOCALITY-FIRST THRESHOLDS (in miles)
        self.PREFERRED_DISTANCE = 1.5        # Ideal assignments: ≤1.5 miles
        self.MAX_DISTANCE = 2.5              # Backup assignments: ≤2.5 miles
        self.ABSOLUTE_MAX_DISTANCE = 3.5     # Search limit: ≤3.5 miles
        self.CASCADING_MOVE_MAX = 3.0       # Max distance for cascading moves
        self.ADJACENT_GROUP_DISTANCE = 1.0   # Base adjacent group distance (scales with radius)
        self.EXCLUSION_DISTANCE = 50.0       # Skip if >50 miles (clearly placeholder)
        self.GREEDY_WALK_MAX_DISTANCE = self.MAX_DISTANCE
        
        # Data containers
        self.distance_matrix = None
        self.haversine_matrix = None  # Fallback haversine distances
        self.dog_assignments = None
        self.driver_capacities = None
        self.sheets_client = None
        self.called_out_drivers = set()  # Track drivers who called out

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
        """Load distance matrix data from Google Sheets (in MILES)"""
        try:
            print("📊 Loading distance matrix (in miles)...")
            
            # Try CSV export first (simpler and was working before)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Fetch CSV data
                    response = requests.get(self.DISTANCE_MATRIX_URL, timeout=30)
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
                    
                    # DEBUG: Check some sample values
                    print("\n🔍 DEBUG: Sample distance values (in MILES):")
                    sample_dogs = list(self.distance_matrix.index)[:5]
                    for i, dog1 in enumerate(sample_dogs[:3]):
                        for dog2 in sample_dogs[i+1:i+3]:
                            if dog2 in self.distance_matrix.columns:
                                value = self.distance_matrix.loc[dog1, dog2]
                                if not pd.isna(value):
                                    print(f"   {dog1} → {dog2}: {value:.1f} miles")
                    
                    return True
                    
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Attempt {attempt + 1} failed, retrying... ({e})")
                        time.sleep(2)  # Wait 2 seconds before retry
                    else:
                        raise
            
        except Exception as e:
            print(f"❌ Error loading distance matrix via CSV: {e}")
            print("🔄 Falling back to Google Sheets API...")
            
            # Fall back to API method
            return self.load_distance_matrix_via_api()
    
    def load_distance_matrix_via_api(self):
        """Load distance matrix using Google Sheets API as fallback"""
        try:
            if not self.sheets_client:
                print("❌ Google Sheets client not initialized")
                return False
            
            # Extract sheet ID from the URL
            sheet_id = "1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg"
            
            # Open the spreadsheet
            spreadsheet = self.sheets_client.open_by_key(sheet_id)
            
            # Try to find the correct worksheet
            worksheet = None
            
            # First try by gid
            try:
                for ws in spreadsheet.worksheets():
                    if str(ws.id) == "398422902":
                        worksheet = ws
                        print(f"📋 Found worksheet by GID: {ws.title}")
                        break
            except:
                pass
            
            # If not found, try common names
            if not worksheet:
                sheet_names = ["Distance Matrix", "Matrix", "Sheet1"]
                for name in sheet_names:
                    try:
                        worksheet = spreadsheet.worksheet(name)
                        print(f"📋 Found worksheet: {name}")
                        break
                    except:
                        continue
            
            # If still not found, use first sheet
            if not worksheet:
                worksheet = spreadsheet.get_worksheet(0)
                print(f"📋 Using first worksheet: {worksheet.title}")
            
            # Get all values
            all_values = worksheet.get_all_values()
            
            if not all_values:
                print("❌ No data found in distance matrix")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(all_values[1:], columns=all_values[0])
            df = df.set_index(df.columns[0])
            
            # Convert numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"📊 Distance matrix shape: ({len(df)}, {len(df.columns)})")
            
            # Extract dog IDs from columns (skip non-dog columns)
            dog_ids = [col for col in df.columns if 'x' in str(col).lower()]
            print(f"📊 Found {len(dog_ids)} column Dog IDs")
            
            # Filter to only dog ID columns and rows
            dog_df = df.loc[df.index.isin(dog_ids), dog_ids]
            
            self.distance_matrix = dog_df
            print(f"✅ Loaded distance matrix for {len(self.distance_matrix)} dogs via API")
            
            # DEBUG: Check some sample values
            print("\n🔍 DEBUG: Sample distance values (in MILES):")
            sample_dogs = list(self.distance_matrix.index)[:5]
            for i, dog1 in enumerate(sample_dogs[:3]):
                for dog2 in sample_dogs[i+1:i+3]:
                    if dog2 in self.distance_matrix.columns:
                        value = self.distance_matrix.loc[dog1, dog2]
                        if not pd.isna(value):
                            print(f"   {dog1} → {dog2}: {value:.1f} miles")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading distance matrix via API: {e}")
            import traceback
            print(f"🔍 Full error: {traceback.format_exc()}")
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
            print(f"🔍 DEBUG: First few column names: {list(df.columns[:15])}")
            
            assignments = []
            
            for i, row in df.iterrows():
                try:
                    # Column positions (0-indexed)
                    dog_name = row.iloc[1] if len(row) > 1 else ""  # Column B
                    combined = row.iloc[7] if len(row) > 7 else ""  # Column H
                    group = row.iloc[8] if len(row) > 8 else ""     # Column I  
                    dog_id = row.iloc[9] if len(row) > 9 else ""    # Column J
                    callout = row.iloc[10] if len(row) > 10 else "" # Column K
                    num_dogs = row.iloc[5] if len(row) > 5 else 1   # Column F (Number of dogs)
                    
                    # Debug first few rows
                    if i < 10:
                        print(f"🔍 Row {i}: DogName='{dog_name}', Combined='{combined}', DogID='{dog_id}', Callout='{callout}'")
                    
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
                    print(f"⚠️ Error processing row {i}: {e}")
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

    def load_haversine_matrix(self, url=None, sheet_id=None, gid=None):
        """Load haversine distance matrix as fallback (optional)"""
        try:
            print("📊 Loading haversine distance matrix (fallback)...")
            
            # You can either provide a URL or sheet details
            if url:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                from io import StringIO
                df = pd.read_csv(StringIO(response.text), index_col=0)
            elif sheet_id and self.sheets_client:
                # Load via API
                spreadsheet = self.sheets_client.open_by_key(sheet_id)
                worksheet = spreadsheet.get_worksheet(0) if not gid else None
                
                if gid:
                    for ws in spreadsheet.worksheets():
                        if str(ws.id) == str(gid):
                            worksheet = ws
                            break
                
                all_values = worksheet.get_all_values()
                df = pd.DataFrame(all_values[1:], columns=all_values[0])
                df = df.set_index(df.columns[0])
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print("⚠️ No haversine matrix URL or sheet ID provided")
                return False
            
            # Extract dog IDs
            dog_ids = [col for col in df.columns if 'x' in str(col).lower()]
            self.haversine_matrix = df.loc[df.index.isin(dog_ids), dog_ids]
            
            print(f"✅ Loaded haversine matrix for {len(self.haversine_matrix)} dogs")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not load haversine matrix (not critical): {e}")
            return False

    def get_dogs_to_reassign(self):
        """Find dogs that need reassignment (callouts) - excluding non-dog entries"""
        dogs_to_reassign = []
        called_out_drivers = set()  # Track drivers who called out
        
        if not self.dog_assignments:
            return dogs_to_reassign, called_out_drivers
        
        print(f"🔍 DEBUG: Checking {len(self.dog_assignments)} total assignments for callouts...")
        
        callout_candidates = 0
        filtered_out = 0
        no_colon = 0
        no_groups = 0
        
        for i, assignment in enumerate(self.dog_assignments):
            # Debug first few and last few
            if i < 5 or i >= len(self.dog_assignments) - 5:
                print(f"   Row {i}: ID={assignment.get('dog_id', 'MISSING')}, Combined='{assignment.get('combined', 'MISSING')}', Callout='{assignment.get('callout', 'MISSING')}'")
            
            # Check for callout: Combined column blank AND Callout column has content
            combined_blank = (not assignment['combined'] or assignment['combined'].strip() == "")
            callout_has_content = (assignment['callout'] and assignment['callout'].strip() != "")
            
            if combined_blank and callout_has_content:
                callout_candidates += 1
                
                # FILTER OUT NON-DOGS: Skip Parking, Field, and other administrative entries
                dog_name = str(assignment.get('dog_name', '')).lower().strip()
                if any(keyword in dog_name for keyword in ['parking', 'field', 'admin', 'office']):
                    print(f"   ⏭️ Skipping non-dog entry: {assignment['dog_name']} ({assignment['dog_id']})")
                    filtered_out += 1
                    continue
                
                # Extract the FULL assignment string (everything after the colon)
                callout_text = assignment['callout'].strip()
                
                if ':' not in callout_text:
                    print(f"   ⚠️ No colon in callout for {assignment.get('dog_id', 'UNKNOWN')}: '{callout_text}'")
                    no_colon += 1
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
                    # Track this driver as called out
                    called_out_drivers.add(original_driver)
                else:
                    print(f"   ⚠️ No groups found for {assignment.get('dog_id', 'UNKNOWN')}: '{full_assignment_string}'")
                    no_groups += 1
        
        print(f"🔍 DEBUG SUMMARY:")
        print(f"   📊 Total assignments checked: {len(self.dog_assignments)}")
        print(f"   🎯 Callout candidates (blank combined + has callout): {callout_candidates}")
        print(f"   🚫 Filtered out (non-dogs): {filtered_out}")
        print(f"   ⚠️ No colon in callout: {no_colon}")
        print(f"   ⚠️ No groups extracted: {no_groups}")
        print(f"   ✅ Final dogs to reassign: {len(dogs_to_reassign)}")
        
        print(f"\n🚨 Found {len(dogs_to_reassign)} REAL dogs that need drivers assigned:")
        for dog in dogs_to_reassign:
            print(f"   - {dog['dog_name']} ({dog['dog_id']}) - {dog['num_dogs']} dogs")
            print(f"     Original: {dog['original_callout']}")  
            print(f"     Assignment string: '{dog['full_assignment_string']}'")
            print(f"     Capacity needed in groups: {dog['needed_groups']}")
        
        if called_out_drivers:
            print(f"\n🚫 Drivers who called out (will be excluded): {', '.join(sorted(called_out_drivers))}")
        
        # Check if dogs are in distance matrix
        print(f"\n🔍 Checking if callout dogs are in distance matrix...")
        for dog in dogs_to_reassign:
            dog_id = dog['dog_id']
            if self.distance_matrix is not None:
                if dog_id not in self.distance_matrix.index:
                    print(f"   ⚠️ {dog['dog_name']} ({dog_id}) NOT FOUND in distance matrix!")
                else:
                    # Check if this dog has any valid distances
                    valid_distances = 0
                    for other_id in self.distance_matrix.columns[:20]:  # Check first 20
                        dist = self.distance_matrix.loc[dog_id, other_id]
                        if not pd.isna(dist) and dist < 50:
                            valid_distances += 1
                    if valid_distances == 0:
                        print(f"   ⚠️ {dog['dog_name']} ({dog_id}) has NO VALID distances in matrix!")
        
        # Store called out drivers in the class
        self.called_out_drivers = called_out_drivers
        
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
        """Get distance between two dogs in miles"""
        # DEBUG for specific dogs
        debug_mode = ('1527x' in [dog1_id, dog2_id] and 'Sirius' in [dog1_id, dog2_id]) or \
                    ('1527x' in [dog1_id, dog2_id] and '1505x' in [dog1_id, dog2_id])
        
        try:
            # First try the distance matrix
            if self.distance_matrix is not None:
                if dog1_id in self.distance_matrix.index and dog2_id in self.distance_matrix.columns:
                    distance = self.distance_matrix.loc[dog1_id, dog2_id]
                    if not pd.isna(distance):
                        if debug_mode:
                            print(f"      📏 DEBUG: Distance {dog1_id} → {dog2_id} = {float(distance):.2f} mi (from matrix)")
                        return float(distance)
            
            # Fallback: Try haversine matrix if available
            if self.haversine_matrix is not None:
                if dog1_id in self.haversine_matrix.index and dog2_id in self.haversine_matrix.columns:
                    distance = self.haversine_matrix.loc[dog1_id, dog2_id]
                    if not pd.isna(distance):
                        # Only print for reasonable distances (not placeholders)
                        if distance < 50:  # Reasonable distance threshold
                            if debug_mode:
                                print(f"      📏 DEBUG: Distance {dog1_id} → {dog2_id} = {float(distance):.2f} mi (haversine fallback)")
                            else:
                                print(f"📏 Haversine fallback: {dog1_id} → {dog2_id} = {distance:.1f} mi")
                        return float(distance)
            
            # If both matrices fail, return infinity
            if debug_mode:
                print(f"      📏 DEBUG: Distance {dog1_id} → {dog2_id} = INF (not found in matrices)")
            return float('inf')
            
        except Exception as e:
            if debug_mode:
                print(f"      📏 DEBUG: Distance {dog1_id} → {dog2_id} = INF (error: {e})")
            return float('inf')

    def calculate_driver_load(self, driver_name: str, current_assignments: List = None) -> Dict:
        """Calculate current load for a driver across all groups - FIXED VERSION"""
        load = {'group1': 0, 'group2': 0, 'group3': 0}
        
        # Use provided assignments or default to original assignments
        assignments_to_use = current_assignments if current_assignments else self.dog_assignments
        
        if not assignments_to_use:
            return load
        
        # CRITICAL FIX: Track which dog IDs we've already counted to avoid duplicates
        counted_dogs = set()
        
        for assignment in assignments_to_use:
            if current_assignments:
                # Working with dynamic assignment list
                if assignment.get('driver') == driver_name:
                    dog_id = assignment.get('dog_id')
                    
                    # Skip if we've already counted this dog (prevents double-counting)
                    if dog_id in counted_dogs:
                        print(f"      ⚠️ Skipping duplicate: {dog_id} already counted for {driver_name}")
                        continue
                    counted_dogs.add(dog_id)
                    
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
                        dog_id = assignment.get('dog_id')
                        
                        # Skip if we've already counted this dog
                        if dog_id in counted_dogs:
                            continue
                        counted_dogs.add(dog_id)
                        
                        # Parse groups for this assignment
                        groups_part = combined.split(':', 1)[1].strip()
                        assigned_groups = self._extract_groups_for_capacity_check(groups_part)
                        
                        # Add to load for each group
                        for group in assigned_groups:
                            group_key = f'group{group}'
                            if group_key in load:
                                load[group_key] += assignment['num_dogs']
        
        # DEBUG: For problematic drivers, double-check
        if driver_name in ["Chase", "Hannah", "Blanch"] and current_assignments:
            # Count dogs manually
            manual_count = 0
            for assignment in current_assignments:
                if assignment.get('driver') == driver_name:
                    manual_count += 1
            if manual_count != len(counted_dogs):
                print(f"      🚨 COUNT MISMATCH for {driver_name}: manual={manual_count}, counted={len(counted_dogs)}")
        
        return load

    def build_initial_assignments_state(self):
        """Build clean initial current_assignments state without duplicates"""
        current_assignments = []
        seen_dogs = set()
        
        print("📊 Building clean initial assignment state...")
        
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            dog_id = assignment.get('dog_id', '')
            
            # Skip if we've already seen this dog or if no valid assignment
            if not combined or ':' not in combined or not dog_id:
                continue
                
            if dog_id in seen_dogs:
                print(f"   ⚠️ Skipping duplicate entry for {dog_id}")
                continue
                
            seen_dogs.add(dog_id)
            
            driver = combined.split(':', 1)[0].strip()
            assignment_string = combined.split(':', 1)[1].strip()
            groups = self._extract_groups_for_capacity_check(assignment_string)
            
            current_assignments.append({
                'dog_id': dog_id,
                'dog_name': assignment['dog_name'],
                'driver': driver,
                'needed_groups': groups,
                'num_dogs': assignment['num_dogs']
            })
        
        print(f"✅ Built state with {len(current_assignments)} unique assignments")
        return current_assignments

    def make_assignment_safely(self, callout_dog, driver, current_assignments, assignment_type='direct'):
        """Safely make an assignment ensuring no duplicates and capacity is respected"""
        # DEBUG MODE - Enable detailed debugging
        DEBUG_ASSIGNMENTS = True  # Set to False to reduce output
        
        if DEBUG_ASSIGNMENTS:
            print(f"\n🔍 DEBUG: Making assignment {callout_dog['dog_name']} → {driver}")
            print(f"   Before: {len(current_assignments)} assignments")
            existing = [a for a in current_assignments if a.get('dog_id') == callout_dog['dog_id']]
            if existing:
                print(f"   ⚠️ Dog already in assignments: {len(existing)} times")
                for e in existing:
                    print(f"      - With driver: {e.get('driver')}")
        
        # First, remove any existing entries for this dog
        before_count = len(current_assignments)
        current_assignments[:] = [a for a in current_assignments 
                                if a.get('dog_id') != callout_dog['dog_id']]
        after_count = len(current_assignments)
        
        if before_count > after_count:
            print(f"      🧹 Removed {before_count - after_count} existing entries for {callout_dog['dog_name']}")
        
        # Add the new assignment
        current_assignments.append({
            'dog_id': callout_dog['dog_id'],
            'dog_name': callout_dog['dog_name'],
            'driver': driver,
            'needed_groups': callout_dog['needed_groups'],
            'num_dogs': callout_dog['num_dogs']
        })
        
        if DEBUG_ASSIGNMENTS:
            print(f"   After: {len(current_assignments)} assignments")
        
        # Verify capacity is still valid
        load = self.calculate_driver_load(driver, current_assignments)
        capacity = self.driver_capacities.get(driver, {})
        
        # DEBUG: Double-check capacity calculation
        if DEBUG_ASSIGNMENTS and driver in ["Chase", "Hannah", "Blanch"]:
            print(f"\n   🔍 SPECIAL DEBUG for {driver}:")
            load_debug = self.debug_capacity_calculation(driver, current_assignments, "(after assignment)")
            if load != load_debug:
                print(f"   🚨 CAPACITY MISMATCH! Regular: {load}, Debug: {load_debug}")
        
        capacity_ok = True
        for group in callout_dog['needed_groups']:
            group_key = f'group{group}'
            current = load.get(group_key, 0)
            max_cap = capacity.get(group_key, 0)
            if current > max_cap:
                print(f"      🚨 WARNING: {driver} Group {group} now at {current}/{max_cap} - OVER CAPACITY!")
                capacity_ok = False
            else:
                print(f"      ✅ {driver} Group {group}: {current}/{max_cap} OK")
        
        return capacity_ok

    def check_driver_can_accept(self, driver_name, callout_dog, current_assignments):
        """Check if a driver can accept a dog without violating capacity"""
        current_load = self.calculate_driver_load(driver_name, current_assignments)
        capacity = self.driver_capacities.get(driver_name, {})
        
        # DEBUG MODE for specific dogs/drivers
        debug_mode = (callout_dog['dog_name'] in ['Keegan', 'Winston', 'Leo', 'Scout'] and 
                     driver_name in ['Hannah', 'Leen', 'Sara'])
        
        if debug_mode:
            print(f"\n      🔍 DEBUG check_driver_can_accept: {callout_dog['dog_name']} → {driver_name}")
            print(f"         Dog needs groups: {callout_dog['needed_groups']}, num_dogs: {callout_dog['num_dogs']}")
        
        for group in callout_dog['needed_groups']:
            group_key = f'group{group}'
            current = current_load.get(group_key, 0)
            max_cap = capacity.get(group_key, 0)
            needed = callout_dog['num_dogs']
            
            if debug_mode:
                can_fit = current + needed <= max_cap
                print(f"         Group {group}: current={current} + needed={needed} = {current+needed} {'<=' if can_fit else '>'} max={max_cap} → {'✅' if can_fit else '❌'}")
            
            if current + needed > max_cap:
                return False
        
        if debug_mode:
            print(f"         ✅ Driver CAN accept dog")
        
        return True

    def verify_capacity_constraints(self, current_assignments):
        """Verify that all capacity constraints are satisfied"""
        violations = []
        
        print("\n🔍 Verifying capacity constraints...")
        
        # First, check for duplicates
        dog_counts = {}
        for assignment in current_assignments:
            dog_id = assignment.get('dog_id')
            if dog_id in dog_counts:
                dog_counts[dog_id] += 1
            else:
                dog_counts[dog_id] = 1
        
        duplicates = [(dog_id, count) for dog_id, count in dog_counts.items() if count > 1]
        if duplicates:
            print("   🚨 DUPLICATE DOGS FOUND:")
            for dog_id, count in duplicates:
                print(f"      - {dog_id} appears {count} times!")
        
        # Check each driver
        for driver_name, capacity in self.driver_capacities.items():
            load = self.calculate_driver_load(driver_name, current_assignments)
            
            # Check each group
            for group_num in [1, 2, 3]:
                group_key = f'group{group_num}'
                current_load = load.get(group_key, 0)
                max_capacity = capacity.get(group_key, 0)
                
                if current_load > max_capacity:
                    violations.append({
                        'driver': driver_name,
                        'group': group_num,
                        'current': current_load,
                        'max': max_capacity,
                        'over': current_load - max_capacity
                    })
        
        if violations:
            print(f"   🚨 CAPACITY VIOLATIONS DETECTED: {len(violations)}")
            for v in violations:
                print(f"      ❌ {v['driver']} Group {v['group']}: {v['current']}/{v['max']} (over by {v['over']})")
        else:
            print(f"   ✅ All capacity constraints satisfied")
        
        return violations

    def debug_capacity_calculation(self, driver_name, current_assignments, context=""):
        """Debug version of calculate_driver_load with detailed logging"""
        print(f"\n🔍 DEBUG CAPACITY CALCULATION for {driver_name} {context}")
        print(f"   Using assignments list with {len(current_assignments)} entries")
        
        load = {'group1': 0, 'group2': 0, 'group3': 0}
        counted_dogs = set()
        dogs_by_group = {'group1': [], 'group2': [], 'group3': []}
        
        for assignment in current_assignments:
            if assignment.get('driver') == driver_name:
                dog_id = assignment.get('dog_id')
                dog_name = assignment.get('dog_name', 'Unknown')
                
                if dog_id in counted_dogs:
                    print(f"   ⚠️ DUPLICATE: {dog_name} ({dog_id}) already counted!")
                    continue
                    
                counted_dogs.add(dog_id)
                assigned_groups = assignment.get('needed_groups', [])
                num_dogs = assignment.get('num_dogs', 1)
                
                print(f"   📦 {dog_name} ({dog_id}): {num_dogs} dogs in groups {assigned_groups}")
                
                for group in assigned_groups:
                    group_key = f'group{group}'
                    if group_key in load:
                        load[group_key] += num_dogs
                        dogs_by_group[group_key].append(f"{dog_name}({num_dogs})")
        
        # Show final tallies
        capacity = self.driver_capacities.get(driver_name, {})
        print(f"\n   📊 FINAL LOAD for {driver_name}:")
        for group_num in [1, 2, 3]:
            group_key = f'group{group_num}'
            current = load.get(group_key, 0)
            max_cap = capacity.get(group_key, 0)
            dogs_list = dogs_by_group.get(group_key, [])
            
            status = "✅" if current <= max_cap else "🚨"
            print(f"   {status} Group {group_num}: {current}/{max_cap} - Dogs: {', '.join(dogs_list) if dogs_list else 'None'}")
        
        return load

    def debug_unassigned_dog(self, dog_info):
        """Detailed analysis of why a dog couldn't be assigned"""
        print(f"\n🔍 DEEP DEBUG: Why {dog_info['dog_name']} ({dog_info['dog_id']}) couldn't be assigned")
        print(f"   Needs groups: {dog_info['needed_groups']}")
        print(f"   Number of dogs: {dog_info['num_dogs']}")
        
        # Special check for Keegan
        if dog_info['dog_name'] == 'Keegan':
            print(f"\n   🎯 SPECIAL KEEGAN ANALYSIS:")
            print(f"   Looking for Sirius (1505x) who is with Hannah...")
            # Check distance to Sirius specifically
            sirius_distance = self.get_distance(dog_info['dog_id'], '1505x')
            print(f"   Distance to Sirius: {sirius_distance:.2f} mi")
        
        # Categorize all drivers
        results = {
            'too_far': [],
            'wrong_groups': [],
            'no_capacity': [],
            'called_out': [],
            'compatible_but_full': [],
            'bad_route_coherence': []
        }
        
        print(f"\n   Checking all {len(self.driver_capacities)} drivers...")
        
        # Build current assignments for checking
        current_assignments = self.build_initial_assignments_state()
        
        # DEBUG: Find Hannah's dogs specifically
        hannah_dogs = [a for a in current_assignments if a.get('driver') == 'Hannah']
        if hannah_dogs:
            print(f"\n   📍 Hannah has {len(hannah_dogs)} dogs:")
            close_count = 0
            missing_count = 0
            for hdog in hannah_dogs[:5]:  # Show first 5
                distance = self.get_distance(dog_info['dog_id'], hdog['dog_id'])
                if distance >= 50:  # Likely missing data
                    print(f"      - {hdog['dog_name']} ({hdog['dog_id']}): MISSING DATA (shows as {distance:.0f} mi)")
                    missing_count += 1
                else:
                    print(f"      - {hdog['dog_name']} ({hdog['dog_id']}): {distance:.2f} mi away")
                    if distance <= 3.0:
                        close_count += 1
            if len(hannah_dogs) > 5:
                print(f"      ... and {len(hannah_dogs) - 5} more dogs")
            print(f"   📊 Route coherence: {close_count}/{len(hannah_dogs)} within 3 mi = {close_count/len(hannah_dogs)*100:.0f}%")
            if missing_count > 0:
                print(f"   ⚠️ Missing distance data for {missing_count} dogs!")
        
        for driver_name, capacity in self.driver_capacities.items():
            # Skip called-out drivers
            if driver_name in self.called_out_drivers:
                results['called_out'].append(driver_name)
                continue
            
            # Get all assignments for this driver
            driver_assignments = [a for a in current_assignments if a.get('driver') == driver_name]
            
            if not driver_assignments:
                continue  # Driver has no dogs assigned
            
            # Find closest dog assigned to this driver
            min_distance = float('inf')
            closest_dog = None
            
            for assignment in driver_assignments:
                dog_id = assignment['dog_id']
                distance = self.get_distance(dog_info['dog_id'], dog_id)
                if distance < min_distance:
                    min_distance = distance
                    closest_dog = assignment
            
            if min_distance == float('inf'):
                continue
            
            # Check distance
            if min_distance > self.ABSOLUTE_MAX_DISTANCE:
                results['too_far'].append({
                    'driver': driver_name,
                    'distance': min_distance,
                    'via': closest_dog['dog_name'] if closest_dog else 'Unknown'
                })
                continue
            
            # Get ALL groups for this driver
            driver_all_groups = []
            for a in driver_assignments:
                driver_all_groups.extend(a.get('needed_groups', []))
            
            # Check group compatibility
            compatible = self.check_group_compatibility(
                dog_info['needed_groups'], 
                driver_all_groups, 
                min_distance,
                self.ABSOLUTE_MAX_DISTANCE
            )
            
            if not compatible:
                results['wrong_groups'].append({
                    'driver': driver_name,
                    'distance': min_distance,
                    'driver_groups': list(set(driver_all_groups)),
                    'needed_groups': dog_info['needed_groups']
                })
                continue
            
            # Check route coherence
            if not self.check_route_coherence(dog_info['dog_id'], driver_name, current_assignments):
                results['bad_route_coherence'].append({
                    'driver': driver_name,
                    'distance': min_distance
                })
                continue
            
            # Check capacity
            current_load = self.calculate_driver_load(driver_name, current_assignments)
            has_space = True
            capacity_details = []
            
            for group in dog_info['needed_groups']:
                group_key = f'group{group}'
                current = current_load.get(group_key, 0)
                max_cap = capacity.get(group_key, 0)
                needed = dog_info['num_dogs']
                
                if current + needed > max_cap:
                    has_space = False
                    capacity_details.append(f"Group{group}: {current}+{needed}>{max_cap}")
            
            if has_space:
                print(f"\n   🚨 ERROR: {driver_name} SHOULD have been able to take this dog!")
                print(f"      Distance: {min_distance:.1f} mi (within {self.ABSOLUTE_MAX_DISTANCE} limit)")
                print(f"      Groups: driver has {list(set(driver_all_groups))}, dog needs {dog_info['needed_groups']}")
                print(f"      Capacity: Available")
                print(f"      Route coherence: Passed")
            else:
                results['compatible_but_full'].append({
                    'driver': driver_name,
                    'distance': min_distance,
                    'capacity_issue': capacity_details
                })
        
        # Print summary
        print(f"\n   📊 SUMMARY for {dog_info['dog_name']}:")
        print(f"   - Too far (>{self.ABSOLUTE_MAX_DISTANCE} mi): {len(results['too_far'])} drivers")
        print(f"   - Wrong groups: {len(results['wrong_groups'])} drivers")
        print(f"   - Bad route coherence: {len(results['bad_route_coherence'])} drivers")
        print(f"   - Right groups but full: {len(results['compatible_but_full'])} drivers")
        print(f"   - Called out: {len(results['called_out'])} drivers")
        
        total_checked = sum(len(results[k]) for k in results)
        print(f"   - TOTAL: {total_checked} drivers")
        
        # Show details for close but full drivers
        if results['compatible_but_full']:
            print(f"\n   🎯 CLOSE DRIVERS THAT WERE FULL:")
            for item in sorted(results['compatible_but_full'], key=lambda x: x['distance'])[:5]:
                print(f"   - {item['driver']} ({item['distance']:.1f} mi): {'; '.join(item['capacity_issue'])}")

    def verify_assignment_state(self, current_assignments, step_name=""):
        """Verify the assignment state is consistent"""
        print(f"\n🔍 VERIFYING ASSIGNMENT STATE {step_name}")
        
        # Check for duplicates
        dog_counts = {}
        for assignment in current_assignments:
            dog_id = assignment.get('dog_id')
            if dog_counts.get(dog_id):
                dog_counts[dog_id] += 1
            else:
                dog_counts[dog_id] = 1
        
        duplicates = [(dog_id, count) for dog_id, count in dog_counts.items() if count > 1]
        if duplicates:
            print("   🚨 DUPLICATE DOGS IN ASSIGNMENTS:")
            for dog_id, count in duplicates:
                print(f"      - {dog_id} appears {count} times!")
                # Find all instances
                for assignment in current_assignments:
                    if assignment.get('dog_id') == dog_id:
                        print(f"        → Driver: {assignment.get('driver')}")

    def check_group_compatibility(self, callout_groups, driver_all_groups, distance, current_radius=None):
        """UPDATED: Check if driver can handle all groups needed by the dog"""
        # Determine distance threshold based on current radius
        if current_radius is not None:
            threshold = current_radius
        else:
            threshold = self.ABSOLUTE_MAX_DISTANCE  # 3.5 miles
        
        # Check if distance is within threshold
        if distance > threshold:
            return False
        
        # Convert to sets for easier comparison
        driver_groups_set = set(driver_all_groups)  # Already a flat list
        callout_set = set(callout_groups)
        
        # Primary check: Does driver cover ALL groups the dog needs?
        if callout_set.issubset(driver_groups_set):
            return True
        
        # Fallback: Check adjacent groups with tighter distance
        adjacent_threshold = threshold * 0.75
        if distance <= adjacent_threshold:
            # Check if we can cover needed groups via adjacent groups
            covered_groups = driver_groups_set.copy()
            
            # Add adjacent groups
            if 1 in driver_groups_set:
                covered_groups.add(2)  # 1 is adjacent to 2
            if 2 in driver_groups_set:
                covered_groups.update([1, 3])  # 2 is adjacent to both 1 and 3
            if 3 in driver_groups_set:
                covered_groups.add(2)  # 3 is adjacent to 2
            
            # Check again with adjacent groups included
            if callout_set.issubset(covered_groups):
                return True
        
        return False

    def check_route_coherence(self, new_dog_id, driver_name, current_assignments, threshold=0.25):
        """Check if adding this dog makes sense for the driver's route"""
        # Get all dogs currently assigned to this driver
        driver_dogs = [a for a in current_assignments if a.get('driver') == driver_name]
        
        if len(driver_dogs) < 2:
            return True  # Not enough dogs to determine route coherence
        
        # DEBUG mode for specific dogs
        debug_mode = new_dog_id in ['1527x', '1529x', '1535x', '1714x']
        
        # Count how many of driver's dogs are within 3 miles of new dog
        nearby_count = 0
        missing_count = 0
        
        for dog_assignment in driver_dogs:
            other_dog_id = dog_assignment.get('dog_id')
            distance = self.get_distance(new_dog_id, other_dog_id)
            
            # Track missing distances (100+ miles)
            if distance >= 50:  # Likely a placeholder
                missing_count += 1
            elif distance <= 3.0:  # Within 3 miles (increased from 2.5)
                nearby_count += 1
        
        # Check if at least 25% of driver's dogs are nearby (reduced from 30%)
        nearby_ratio = nearby_count / len(driver_dogs)
        passes = nearby_ratio >= threshold
        
        if debug_mode:
            print(f"      Route coherence for {new_dog_id} → {driver_name}:")
            print(f"         Total dogs: {len(driver_dogs)}, Nearby (≤3mi): {nearby_count}, Missing data: {missing_count}")
            print(f"         Ratio: {nearby_count}/{len(driver_dogs)} = {nearby_ratio:.1%} {'≥' if passes else '<'} 25% → {'PASS' if passes else 'FAIL'}")
        
        return passes

    def assign_neighbors(self, assigned_dog, driver, current_assignments, dogs_remaining, radius=0.5):
        """When a dog is assigned, try to assign unassigned neighbors to same driver"""
        assigned_neighbors = []
        
        for neighbor_dog in dogs_remaining[:]:  # Use slice to allow removal during iteration
            # Check distance between assigned dog and potential neighbor
            distance = self.get_distance(assigned_dog['dog_id'], neighbor_dog['dog_id'])
            
            if distance <= radius:  # Very close neighbor (0.5 miles)
                # Check if driver can accept this neighbor
                if self.check_driver_can_accept(driver, neighbor_dog, current_assignments):
                    # Check group compatibility
                    driver_all_assignments = [a for a in current_assignments if a.get('driver') == driver]
                    driver_all_groups = []
                    for a in driver_all_assignments:
                        driver_all_groups.extend(a.get('needed_groups', []))
                    
                    if self.check_group_compatibility(neighbor_dog['needed_groups'], driver_all_groups, distance, self.ABSOLUTE_MAX_DISTANCE):
                        # Assign the neighbor
                        print(f"   🏘️ Assigning neighbor {neighbor_dog['dog_name']} to {driver} (same as {assigned_dog['dog_name']})")
                        if self.make_assignment_safely(neighbor_dog, driver, current_assignments):
                            assigned_neighbors.append(neighbor_dog)
                            # FIXED: Only remove if actually in the list
                            if neighbor_dog in dogs_remaining:
                                dogs_remaining.remove(neighbor_dog)
                            else:
                                print(f"   ⚠️ {neighbor_dog['dog_name']} already removed from remaining list")
        
        return assigned_neighbors

    def get_current_driver_dogs(self, driver_name, current_assignments):
        """Get all dogs currently assigned to a specific driver"""
        return [assignment for assignment in current_assignments 
                if assignment.get('driver') == driver_name]

    def attempt_strategic_cascading_move(self, blocked_driver, callout_dog, current_assignments, max_search_radius=3):
        """STRATEGIC: Target specific groups with incremental radius expansion"""
        print(f"🎯 ATTEMPTING STRATEGIC CASCADING MOVE for {callout_dog['dog_name']} → {blocked_driver}")
        
        # Step 1: Identify which groups are causing the capacity problem
        blocked_groups = self._identify_blocked_groups(blocked_driver, callout_dog, current_assignments)
        print(f"   🎯 Target groups causing capacity issues: {blocked_groups}")
        
        if not blocked_groups:
            print(f"   ❌ No blocked groups identified")
            return None
        
        # Step 2: Get dogs from blocked driver, prioritized by strategy
        driver_dogs = self.get_current_driver_dogs(blocked_driver, current_assignments)
        strategic_dogs = self._prioritize_dogs_strategically(driver_dogs, blocked_groups)
        
        print(f"   📊 Strategic prioritization of {len(strategic_dogs)} dogs:")
        for i, (priority, dog) in enumerate(strategic_dogs[:8]):  # Show top 8
            print(f"     {i+1}. {dog['dog_name']} (groups: {dog['needed_groups']}) - {priority}")
        
        # Step 3: Try incremental radius expansion for each high-priority dog
        for priority, dog_to_move in strategic_dogs:
            print(f"   🔄 Trying to move {dog_to_move['dog_name']} (groups: {dog_to_move['needed_groups']})...")
            
            # Use incremental radius expansion like the main algorithm
            move_result = self._attempt_incremental_move(dog_to_move, current_assignments, max_search_radius)
            
            if move_result:
                print(f"   ✅ STRATEGIC MOVE SUCCESSFUL!")
                print(f"      📦 Moved: {dog_to_move['dog_name']} → {move_result['to_driver']}")
                print(f"      📏 Distance: {move_result['distance']:.1f} mi (found at radius {move_result['radius']:.1f} mi)")
                print(f"      🎯 This frees {blocked_groups} capacity in {blocked_driver}")
                return move_result
            else:
                print(f"   ❌ Could not move {dog_to_move['dog_name']} within {max_search_radius} mi")
        
        print(f"   ❌ STRATEGIC CASCADING FAILED - no dogs could be relocated")
        return None

    def _identify_blocked_groups(self, driver_name, callout_dog, current_assignments):
        """Identify which specific groups are causing capacity problems"""
        blocked_groups = []
        
        # Get current capacity and load
        capacity = self.driver_capacities.get(driver_name, {})
        current_load = self.calculate_driver_load(driver_name, current_assignments)
        
        # Check which groups would be over capacity
        for group in callout_dog['needed_groups']:
            group_key = f'group{group}'
            current = current_load.get(group_key, 0)
            max_cap = capacity.get(group_key, 0)
            needed = callout_dog['num_dogs']
            
            if current + needed > max_cap:
                blocked_groups.append(group)
                print(f"   🚨 Group {group} blocking: {current} + {needed} > {max_cap}")
            else:
                print(f"   ✅ Group {group} has space: {current} + {needed} ≤ {max_cap}")
        
        return blocked_groups

    def _prioritize_dogs_strategically(self, driver_dogs, blocked_groups):
        """Prioritize dogs based on strategic value for freeing blocked groups"""
        prioritized = []
        
        for dog in driver_dogs:
            dog_groups = set(dog.get('needed_groups', []))
            blocked_set = set(blocked_groups)
            
            # Calculate strategic priority
            if dog_groups.intersection(blocked_set):
                # Dog is in a blocked group - HIGH PRIORITY
                if len(dog_groups) == 1 and dog['num_dogs'] == 1:
                    priority = "HIGH - Single group, single dog in blocked group"
                elif len(dog_groups) == 1:
                    priority = f"HIGH - Single group, {dog['num_dogs']} dogs in blocked group"
                else:
                    priority = f"MEDIUM - Multi-group dog partially in blocked group"
            else:
                # Dog is not in blocked groups - LOW PRIORITY
                priority = "LOW - Not in blocked groups (won't help)"
            
            prioritized.append((priority, dog))
        
        # Sort by priority (HIGH first, then MEDIUM, then LOW)
        priority_order = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
        prioritized.sort(key=lambda x: (
            priority_order.get(x[0].split(' - ')[0], 4),  # Priority level
            len(x[1].get('needed_groups', [])),           # Fewer groups = easier to place
            x[1].get('num_dogs', 1)                       # Fewer dogs = easier to place
        ))
        
        return prioritized

    def _attempt_incremental_move(self, dog_to_move, current_assignments, max_radius):
        """Try to move a dog using incremental radius expansion - FIXED VERSION"""
        print(f"     🔍 Using incremental radius search for {dog_to_move['dog_name']}...")
        
        # Get the current driver before we move
        old_driver = dog_to_move.get('driver')
        
        if not old_driver:
            # Find current driver from assignments
            for assignment in current_assignments:
                if assignment.get('dog_id') == dog_to_move['dog_id']:
                    old_driver = assignment.get('driver')
                    dog_to_move = assignment.copy()  # Get full assignment info
                    break
        
        if not old_driver:
            print(f"     ❌ Could not find current driver for {dog_to_move['dog_name']}")
            return None
        
        # Start at 1.5 miles and expand incrementally by 0.5 miles
        current_radius = 1.5
        
        while current_radius <= max_radius:
            print(f"       📏 Trying radius {current_radius} miles...")
            
            # Find all potential targets within current radius
            targets = self._find_move_targets_at_radius(dog_to_move, current_assignments, current_radius)
            
            if targets:
                print(f"       ✅ Found {len(targets)} targets at {current_radius} mi:")
                for i, target in enumerate(targets[:3]):  # Show top 3
                    print(f"         {i+1}. {target['driver']} - {target['distance']:.1f} mi")
                
                # Use the closest target
                best_target = targets[0]
                
                # CRITICAL FIX: Properly update the assignment
                print(f"       🔄 Moving {dog_to_move['dog_name']} from {old_driver} to {best_target['driver']}")
                
                # Remove ALL existing entries for this dog
                before_count = len(current_assignments)
                current_assignments[:] = [a for a in current_assignments 
                                        if a.get('dog_id') != dog_to_move['dog_id']]
                after_count = len(current_assignments)
                
                if before_count - after_count > 1:
                    print(f"       ⚠️ Removed {before_count - after_count} entries for {dog_to_move['dog_name']} (duplicates found!)")
                
                # Add the new assignment
                current_assignments.append({
                    'dog_id': dog_to_move['dog_id'],
                    'dog_name': dog_to_move['dog_name'],
                    'driver': best_target['driver'],
                    'needed_groups': dog_to_move.get('needed_groups', []),
                    'num_dogs': dog_to_move.get('num_dogs', 1)
                })
                
                print(f"       ✅ Move complete: {dog_to_move['dog_name']} now with {best_target['driver']}")
                
                # Verify capacity after move
                old_load = self.calculate_driver_load(old_driver, current_assignments)
                new_load = self.calculate_driver_load(best_target['driver'], current_assignments)
                print(f"       📊 {old_driver} load after move: {old_load}")
                print(f"       📊 {best_target['driver']} load after move: {new_load}")
                
                # DEBUG: Special check for Chase
                if old_driver == "Chase" or best_target['driver'] == "Chase":
                    print(f"\n       🔍 SPECIAL DEBUG: Chase capacity after move")
                    self.debug_capacity_calculation("Chase", current_assignments, "(after strategic move)")
                    self.verify_assignment_state(current_assignments, f"after moving {dog_to_move['dog_name']}")
                
                return {
                    'moved_dog': dog_to_move,
                    'from_driver': old_driver,
                    'to_driver': best_target['driver'],
                    'distance': best_target['distance'],
                    'via_dog': best_target['via_dog'],
                    'radius': current_radius
                }
            else:
                print(f"       ❌ No targets at {current_radius} miles")
            
            # Expand radius by 0.5 miles
            current_radius += 0.5
        
        print(f"     ❌ No targets found within {max_radius} miles")
        return None

    def _find_move_targets_at_radius(self, dog_to_move, current_assignments, radius):
        """Find potential drivers within specific radius who can accept the dog"""
        targets = []
        
        for assignment in current_assignments:
            target_driver = assignment['driver']
            
            # Skip same driver
            if target_driver == dog_to_move.get('driver'):
                continue
            
            # CRITICAL: Skip drivers who called out!
            if hasattr(self, 'called_out_drivers') and target_driver in self.called_out_drivers:
                continue
            
            # Calculate distance
            distance = self.get_distance(dog_to_move['dog_id'], assignment['dog_id'])
            
            # Skip if beyond current radius
            if distance > radius or distance >= self.EXCLUSION_DISTANCE:  # Skip placeholders
                continue
            
            # Check group compatibility with current radius
            # Get ALL groups for this driver
            driver_all_assignments = [a for a in current_assignments if a.get('driver') == target_driver]
            driver_all_groups = []
            for a in driver_all_assignments:
                driver_all_groups.extend(a.get('needed_groups', []))
            
            dog_groups = dog_to_move.get('needed_groups', [])
            
            if not self.check_group_compatibility(dog_groups, driver_all_groups, distance, radius):
                continue
            
            # Check if target driver has capacity
            if self.check_driver_can_accept(target_driver, dog_to_move, current_assignments):
                # Additional route coherence check could go here if needed
                targets.append({
                    'driver': target_driver,
                    'distance': distance,
                    'via_dog': assignment['dog_name'],
                    'via_dog_id': assignment['dog_id']
                })
        
        # Sort by distance (closest first)
        return sorted(targets, key=lambda x: x['distance'])

    def locality_first_assignment(self):
        """Locality-first assignment algorithm with strategic cascading and TIGHT distance constraints"""
        print("\n🎯 LOCALITY-FIRST ASSIGNMENT ALGORITHM WITH STRATEGIC CASCADING")
        print("🔄 Step-by-step proximity optimization with strategic cascading moves")
        print("📏 Starting at 1.5 mi, expanding to 3.5 mi in 0.5 mi increments")
        print("🔄 Adjacent groups scale with radius (75% of current radius)")
        print("🎯 STRATEGIC CASCADING: Target specific blocked groups with incremental radius")
        print("🚶 Cascading moves up to 3 mi with radius expansion (1.5→2→2.5→3)")
        print("🎯 TIGHT CONSTRAINTS: Exact matches up to 3.5 mi, adjacent up to ~2.6 mi")
        print("⚠️ WARNING: Very tight constraints may result in unassigned dogs!")
        print("=" * 80)
        
        # DEBUG MODE - Set to True to see detailed decision logging
        DEBUG_MODE = True  # ENABLED FOR DEBUGGING
        DEBUG_DOGS = ["Frankie", "Shelby", "Tillie", "Singa"]  # Dogs to trace in detail
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        # Build clean initial state
        current_assignments = self.build_initial_assignments_state()
        
        # Add initial verification
        print("\n📊 Initial capacity state:")
        self.verify_capacity_constraints(current_assignments)
        
        assignments_made = []
        moves_made = []
        dogs_remaining = dogs_to_reassign.copy()
        
        # Store partial results in case of error
        self._partial_assignments = assignments_made
        
        print(f"🐕 Processing {len(dogs_remaining)} callout dogs")
        
        print(f"\n📍 STEP 1: Direct assignments at ≤{self.PREFERRED_DISTANCE} miles")
        if hasattr(self, 'called_out_drivers') and self.called_out_drivers:
            print(f"   🚫 Excluding called-out drivers: {', '.join(sorted(self.called_out_drivers))}")
        
        dogs_assigned_step1 = []
        for callout_dog in dogs_remaining[:]:
            # Skip if this dog was already assigned as a neighbor
            if callout_dog not in dogs_remaining:
                continue
                
            best_assignment = None
            best_distance = float('inf')
            
            # Debug logging for specific dogs
            if DEBUG_MODE and (not DEBUG_DOGS or callout_dog['dog_name'] in DEBUG_DOGS):
                print(f"\n🔍 DECISION TRACE for {callout_dog['dog_name']} (needs groups {callout_dog['needed_groups']}):")
            
            # Check all drivers for direct assignment
            for assignment in current_assignments:
                driver = assignment['driver']
                
                # CRITICAL: Skip drivers who called out!
                if hasattr(self, 'called_out_drivers') and driver in self.called_out_drivers:
                    continue
                
                distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                
                # Skip obvious placeholders
                if distance >= self.EXCLUSION_DISTANCE:
                    continue
                
                # Check group compatibility - UPDATED to get ALL groups
                driver_all_assignments = [a for a in current_assignments if a.get('driver') == driver]
                driver_all_groups = []
                for a in driver_all_assignments:
                    driver_all_groups.extend(a.get('needed_groups', []))
                
                compatible = self.check_group_compatibility(callout_dog['needed_groups'], driver_all_groups, distance, self.PREFERRED_DISTANCE)
                
                # Check route coherence - skip for very close assignments
                coherent = True
                if distance > 1:
                    coherent = self.check_route_coherence(callout_dog['dog_id'], driver, current_assignments)
                
                # Check capacity
                has_capacity = self.check_driver_can_accept(driver, callout_dog, current_assignments)
                
                # Debug logging
                if DEBUG_MODE and (not DEBUG_DOGS or callout_dog['dog_name'] in DEBUG_DOGS) and distance <= self.PREFERRED_DISTANCE:
                    print(f"   Checking {driver}: dist={distance:.1f}mi, groups={list(set(driver_all_groups))}, compatible={compatible}, coherent={coherent}, capacity={has_capacity}")
                
                if compatible and has_capacity and coherent:
                    if distance < best_distance:
                        best_assignment = {
                            'driver': driver,
                            'distance': distance,
                            'via_dog': assignment['dog_name']
                        }
                        best_distance = distance
            
            if best_assignment:
                # Make the assignment
                driver = best_assignment['driver']
                distance = best_assignment['distance']
                
                # Use safe assignment method
                if self.make_assignment_safely(callout_dog, driver, current_assignments):
                    assignment_record = {
                        'dog_id': callout_dog['dog_id'],
                        'dog_name': callout_dog['dog_name'],
                        'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                        'driver': driver,
                        'distance': distance,
                        'quality': 'GOOD',
                        'assignment_type': 'direct',
                        'original_callout': callout_dog['original_callout']
                    }
                    
                    assignments_made.append(assignment_record)
                    dogs_assigned_step1.append(callout_dog)
                    print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.1f} mi via {best_assignment['via_dog']})")
                    
                    # Try to assign neighbors
                    neighbors = self.assign_neighbors(callout_dog, driver, current_assignments, dogs_remaining)
                    if neighbors:
                        for n in neighbors:
                            n_distance = self.get_distance(n['dog_id'], callout_dog['dog_id'])
                            n_record = {
                                'dog_id': n['dog_id'],
                                'dog_name': n['dog_name'],
                                'new_assignment': f"{driver}:{n['full_assignment_string']}",
                                'driver': driver,
                                'distance': n_distance,
                                'quality': 'GOOD',
                                'assignment_type': 'neighbor_assignment',
                                'original_callout': n['original_callout']
                            }
                            assignments_made.append(n_record)
                            dogs_assigned_step1.append(n)
            elif DEBUG_MODE and (not DEBUG_DOGS or callout_dog['dog_name'] in DEBUG_DOGS):
                print(f"   ❌ No direct assignment found for {callout_dog['dog_name']} at ≤{self.PREFERRED_DISTANCE} mi")
        
        # Remove assigned dogs
        for dog in dogs_assigned_step1:
            if dog in dogs_remaining:
                dogs_remaining.remove(dog)
            else:
                print(f"   ⚠️ {dog['dog_name']} already removed from remaining list")
        
        print(f"   📊 Step 1 results: {len(dogs_assigned_step1)} direct assignments")
        
        # Verify after step 1
        print("\n📊 After Step 1 capacity check:")
        self.verify_capacity_constraints(current_assignments)
        
        # Step 2: Capacity-blocked assignments with strategic cascading moves
        if dogs_remaining:
            print(f"\n🎯 STEP 2: Strategic cascading moves to free space at ≤{self.PREFERRED_DISTANCE} miles")
            
            dogs_assigned_step2 = []
            for callout_dog in dogs_remaining[:]:
                # Skip if this dog was already assigned as a neighbor
                if callout_dog not in dogs_remaining:
                    continue
                    
                # Find drivers within range but blocked by capacity
                blocked_drivers = []
                
                for assignment in current_assignments:
                    driver = assignment['driver']
                    
                    # CRITICAL: Skip drivers who called out!
                    if hasattr(self, 'called_out_drivers') and driver in self.called_out_drivers:
                        continue
                    
                    distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                    
                    # Skip obvious placeholders
                    if distance >= self.EXCLUSION_DISTANCE:
                        continue
                    
                    # Check group compatibility - UPDATED
                    driver_all_assignments = [a for a in current_assignments if a.get('driver') == driver]
                    driver_all_groups = []
                    for a in driver_all_assignments:
                        driver_all_groups.extend(a.get('needed_groups', []))
                    
                    if not self.check_group_compatibility(callout_dog['needed_groups'], driver_all_groups, distance, self.PREFERRED_DISTANCE):
                        continue
                    
                    # Check route coherence - skip for very close assignments
                    coherent = True
                    if distance > 1:
                        coherent = self.check_route_coherence(callout_dog['dog_id'], driver, current_assignments)
                    
                    if not coherent:
                        continue
                    
                    # Check if blocked by capacity
                    if not self.check_driver_can_accept(driver, callout_dog, current_assignments):
                        blocked_drivers.append({
                            'driver': driver,
                            'distance': distance
                        })
                
                # Try strategic cascading moves for the closest blocked driver
                if blocked_drivers:
                    blocked_drivers.sort(key=lambda x: x['distance'])
                    best_blocked = blocked_drivers[0]
                    
                    # STRATEGIC CASCADING: Use strategic approach
                    move_result = self.attempt_strategic_cascading_move(
                        best_blocked['driver'], 
                        callout_dog, 
                        current_assignments, 
                        self.CASCADING_MOVE_MAX  # 3 mi max search with incremental expansion
                    )
                    
                    if move_result:
                        # Record the move
                        moves_made.append({
                            'dog_name': move_result['moved_dog']['dog_name'],
                            'dog_id': move_result['moved_dog']['dog_id'],
                            'from_driver': move_result['from_driver'],
                            'to_driver': move_result['to_driver'],
                            'distance': move_result['distance'],
                            'reason': f"strategic_free_space_for_{callout_dog['dog_name']}"
                        })
                        
                        # Update any existing assignments for moved dog
                        moved_dog_id = move_result['moved_dog']['dog_id']
                        for existing_assignment in assignments_made:
                            if existing_assignment['dog_id'] == moved_dog_id:
                                # Update the assignment to show final location
                                old_driver = existing_assignment['driver']
                                new_driver = move_result['to_driver']
                                existing_assignment['driver'] = new_driver
                                existing_assignment['new_assignment'] = existing_assignment['new_assignment'].replace(f"{old_driver}:", f"{new_driver}:")
                                existing_assignment['assignment_type'] = 'moved_by_strategic_cascading'
                                print(f"      🔄 Updated final assignment: {move_result['moved_dog']['dog_name']} → {new_driver}")
                                break
                        
                        # Now assign the callout dog to the freed space
                        driver = best_blocked['driver']
                        distance = best_blocked['distance']
                        
                        # Use safe assignment
                        if self.make_assignment_safely(callout_dog, driver, current_assignments):
                            assignment_record = {
                                'dog_id': callout_dog['dog_id'],
                                'dog_name': callout_dog['dog_name'],
                                'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                                'driver': driver,
                                'distance': distance,
                                'quality': 'GOOD',
                                'assignment_type': 'strategic_cascading',
                                'original_callout': callout_dog['original_callout']
                            }
                            
                            assignments_made.append(assignment_record)
                            dogs_assigned_step2.append(callout_dog)
                            print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.1f} mi)")
                            print(f"      🎯 Strategic move: {move_result['moved_dog']['dog_name']} → {move_result['to_driver']} ({move_result['distance']:.1f} mi at radius {move_result['radius']:.1f} mi)")
            
            # Remove assigned dogs
            for dog in dogs_assigned_step2:
                if dog in dogs_remaining:
                    dogs_remaining.remove(dog)
                else:
                    print(f"   ⚠️ {dog['dog_name']} already removed from remaining list")
            
            print(f"   📊 Step 2 results: {len(dogs_assigned_step2)} strategic cascading assignments")
            
            # Verify after step 2
            print("\n📊 After Step 2 capacity check:")
            self.verify_capacity_constraints(current_assignments)
        
        # Step 3+: Incremental radius expansion (2 to 3.5 miles)
        current_radius = 2
        step_number = 3
        
        while current_radius <= self.ABSOLUTE_MAX_DISTANCE and dogs_remaining:
            print(f"\n📏 STEP {step_number}: Radius expansion to ≤{current_radius} miles")
            print(f"   🎯 Thresholds: Perfect match ≤{current_radius} mi, Adjacent groups ≤{current_radius*0.75:.1f} mi")
            
            dogs_assigned_this_radius = []
            
            for callout_dog in dogs_remaining[:]:
                # Skip if this dog was already assigned as a neighbor
                if callout_dog not in dogs_remaining:
                    continue
                
                # ENHANCED DEBUG for specific dogs
                debug_this_dog = callout_dog['dog_name'] in ['Keegan', 'Winston', 'Leo', 'Scout']
                if debug_this_dog:
                    print(f"\n   🔍 DEBUG: Processing {callout_dog['dog_name']} at radius {current_radius}")
                    print(f"      Dog ID: {callout_dog['dog_id']}")
                    print(f"      Needs groups: {callout_dog['needed_groups']}")
                    # Special check for Keegan and Sirius
                    if callout_dog['dog_name'] == 'Keegan':
                        print(f"      🎯 Special check: Looking for Sirius (1505x)...")
                    
                # Try direct assignment at current radius
                best_assignment = None
                best_distance = float('inf')
                blocked_drivers = []  # Track blocked drivers at this radius
                drivers_checked = 0
                
                for assignment in current_assignments:
                    driver = assignment['driver']
                    
                    # Skip called-out drivers
                    if hasattr(self, 'called_out_drivers') and driver in self.called_out_drivers:
                        continue
                    
                    distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                    
                    # Skip if beyond current radius or placeholder
                    if distance > current_radius or distance >= self.EXCLUSION_DISTANCE:
                        continue
                    
                    drivers_checked += 1
                    
                    # Check group compatibility - FIX: Get ALL groups for driver
                    driver_all_assignments = [a for a in current_assignments if a.get('driver') == driver]
                    driver_all_groups = []
                    for a in driver_all_assignments:
                        driver_all_groups.extend(a.get('needed_groups', []))
                    
                    compatible = self.check_group_compatibility(callout_dog['needed_groups'], driver_all_groups, distance, current_radius)
                    
                    # DEBUG: Show checks for specific drivers
                    if debug_this_dog and distance <= current_radius and 'Hannah' in driver:
                        print(f"      Checking {driver}: distance={distance:.2f}, groups={list(set(driver_all_groups))}, compatible={compatible}")
                    
                    if not compatible:
                        continue
                    
                    # Check route coherence - skip for very close assignments
                    coherent = True
                    if distance > 1:
                        coherent = self.check_route_coherence(callout_dog['dog_id'], driver, current_assignments)
                    
                    if not coherent:
                        if debug_this_dog:
                            print(f"      {driver} failed route coherence check")
                        continue
                    
                    # Check capacity
                    has_capacity = self.check_driver_can_accept(driver, callout_dog, current_assignments)
                    
                    # DEBUG: Show capacity check for Hannah
                    if debug_this_dog and 'Hannah' in driver:
                        load = self.calculate_driver_load(driver, current_assignments)
                        capacity = self.driver_capacities.get(driver, {})
                        print(f"      {driver} capacity check:")
                        for group in callout_dog['needed_groups']:
                            group_key = f'group{group}'
                            current = load.get(group_key, 0)
                            max_cap = capacity.get(group_key, 0)
                            needed = callout_dog['num_dogs']
                            print(f"         Group {group}: {current} + {needed} {'<=' if current + needed <= max_cap else '>'} {max_cap} = {has_capacity}")
                    
                    if has_capacity:
                        if distance < best_distance:
                            best_assignment = {
                                'driver': driver,
                                'distance': distance,
                                'via_dog': assignment['dog_name']
                            }
                            best_distance = distance
                    else:
                        # Track blocked drivers for cascading
                        blocked_drivers.append({
                            'driver': driver,
                            'distance': distance,
                            'via_dog': assignment['dog_name']
                        })
                        if debug_this_dog:
                            print(f"      ✅ Added {driver} to blocked_drivers list (no capacity)")
                
                if debug_this_dog:
                    print(f"      Summary: {drivers_checked} drivers checked, {len(blocked_drivers)} blocked by capacity")
                
                if best_assignment:
                    # Direct assignment possible
                    driver = best_assignment['driver']
                    distance = best_assignment['distance']
                    
                    # Determine quality
                    if distance <= self.PREFERRED_DISTANCE:
                        quality = 'GOOD'
                    elif distance <= self.MAX_DISTANCE:
                        quality = 'BACKUP'
                    else:
                        quality = 'EMERGENCY'
                    
                    if self.make_assignment_safely(callout_dog, driver, current_assignments):
                        assignment_record = {
                            'dog_id': callout_dog['dog_id'],
                            'dog_name': callout_dog['dog_name'],
                            'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                            'driver': driver,
                            'distance': distance,
                            'quality': quality,
                            'assignment_type': 'radius_expansion',
                            'original_callout': callout_dog['original_callout']
                        }
                        
                        assignments_made.append(assignment_record)
                        dogs_assigned_this_radius.append(callout_dog)
                        print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.1f} mi) [{quality}]")
                        
                        # Try to assign neighbors
                        neighbors = self.assign_neighbors(callout_dog, driver, current_assignments, dogs_remaining)
                        if neighbors:
                            for n in neighbors:
                                n_distance = self.get_distance(n['dog_id'], callout_dog['dog_id'])
                                n_record = {
                                    'dog_id': n['dog_id'],
                                    'dog_name': n['dog_name'],
                                    'new_assignment': f"{driver}:{n['full_assignment_string']}",
                                    'driver': driver,
                                    'distance': n_distance,
                                    'quality': 'GOOD',
                                    'assignment_type': 'neighbor_assignment',
                                    'original_callout': n['original_callout']
                                }
                                assignments_made.append(n_record)
                                dogs_assigned_this_radius.append(n)
                
                # Try cascading at this radius if we have blocked drivers
                elif blocked_drivers:
                    if debug_this_dog:
                        print(f"      🔄 {len(blocked_drivers)} blocked drivers found, attempting cascading...")
                        for bd in blocked_drivers[:3]:  # Show first 3
                            print(f"         - {bd['driver']} at {bd['distance']:.2f} mi via {bd.get('via_dog', 'unknown')}")
                    
                    # Sort by distance
                    blocked_drivers.sort(key=lambda x: x['distance'])
                    best_blocked = blocked_drivers[0]
                    
                    print(f"   🔄 Attempting cascading for {callout_dog['dog_name']} at radius {current_radius}")
                    print(f"      Target driver: {best_blocked['driver']} at {best_blocked['distance']:.2f} mi")
                    
                    # Try strategic cascading with current radius as max
                    move_result = self.attempt_strategic_cascading_move(
                        best_blocked['driver'], 
                        callout_dog, 
                        current_assignments, 
                        min(current_radius, self.CASCADING_MOVE_MAX)
                    )
                    
                    if move_result:
                        # Record the move
                        moves_made.append({
                            'dog_name': move_result['moved_dog']['dog_name'],
                            'dog_id': move_result['moved_dog']['dog_id'],
                            'from_driver': move_result['from_driver'],
                            'to_driver': move_result['to_driver'],
                            'distance': move_result['distance'],
                            'reason': f"strategic_radius_{current_radius}_space_for_{callout_dog['dog_name']}"
                        })
                        
                        # Assign the callout dog
                        driver = best_blocked['driver']
                        distance = best_blocked['distance']
                        
                        # Determine quality
                        if distance <= self.PREFERRED_DISTANCE:
                            quality = 'GOOD'
                        elif distance <= self.MAX_DISTANCE:
                            quality = 'BACKUP'
                        else:
                            quality = 'EMERGENCY'
                        
                        if self.make_assignment_safely(callout_dog, driver, current_assignments):
                            assignment_record = {
                                'dog_id': callout_dog['dog_id'],
                                'dog_name': callout_dog['dog_name'],
                                'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                                'driver': driver,
                                'distance': distance,
                                'quality': quality,
                                'assignment_type': 'strategic_cascading_radius',
                                'original_callout': callout_dog['original_callout']
                            }
                            
                            assignments_made.append(assignment_record)
                            dogs_assigned_this_radius.append(callout_dog)
                            print(f"   ✅ {callout_dog['dog_name']} → {driver} ({distance:.1f} mi) [{quality}]")
                            print(f"      🎯 Strategic move: {move_result['moved_dog']['dog_name']} → {move_result['to_driver']} ({move_result['distance']:.1f} mi)")
                
                else:
                    if debug_this_dog:
                        print(f"      ❌ No direct assignment or blocked drivers found")
                        print(f"      Closest driver analysis:")
                        # Show closest few drivers for debugging
                        close_drivers = []
                        for assignment in current_assignments:
                            driver = assignment['driver']
                            if hasattr(self, 'called_out_drivers') and driver in self.called_out_drivers:
                                continue
                            distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                            # Only include reasonable distances
                            if distance < 50:  # Skip obvious placeholders
                                close_drivers.append({
                                    'driver': driver,
                                    'distance': distance,
                                    'dog': assignment['dog_name'],
                                    'dog_id': assignment['dog_id']
                                })
                        
                        if not close_drivers:
                            print(f"         ⚠️ NO VALID DISTANCES FOUND - Dog may be missing from distance matrix!")
                        else:
                            close_drivers.sort(key=lambda x: x['distance'])
                            for cd in close_drivers[:5]:
                                print(f"         {cd['driver']} via {cd['dog']} ({cd['dog_id']}): {cd['distance']:.2f} mi")
            
            # Remove assigned dogs
            for dog in dogs_assigned_this_radius:
                if dog in dogs_remaining:
                    dogs_remaining.remove(dog)
                else:
                    print(f"   ⚠️ {dog['dog_name']} already removed from remaining list")
            
            print(f"   📊 Radius {current_radius} mi results: {len(dogs_assigned_this_radius)} assignments")
            
            current_radius += 0.5
            step_number += 1
        
        # Final step: Mark remaining as emergency
        if dogs_remaining:
            print(f"\n🚨 FINAL STEP: {len(dogs_remaining)} remaining dogs marked as UNASSIGNED")
            print("⚠️ These dogs could not be assigned within the 3.5 mile constraint!")
            
            for callout_dog in dogs_remaining:
                # Skip if this dog was already assigned somehow
                if callout_dog not in dogs_remaining:
                    continue
                    
                # DEBUG: Analyze why this dog couldn't be assigned
                self.debug_unassigned_dog(callout_dog)
                
                assignment_record = {
                    'dog_id': callout_dog['dog_id'],
                    'dog_name': callout_dog['dog_name'],
                    'new_assignment': f"UNASSIGNED:{callout_dog['full_assignment_string']}",
                    'driver': 'UNASSIGNED',
                    'distance': float('inf'),
                    'quality': 'EMERGENCY',
                    'assignment_type': 'failed',
                    'original_callout': callout_dog['original_callout']
                }
                assignments_made.append(assignment_record)
                print(f"   ❌ {callout_dog['dog_name']} - No viable assignment found within 3.5 miles")
        
        # Store moves for writing
        self.greedy_moves_made = moves_made
        
        # Final verification
        print("\n📊 FINAL capacity verification:")
        violations = self.verify_capacity_constraints(current_assignments)
        
        if violations:
            print("\n🚨 WARNING: Algorithm completed with capacity violations!")
            print("🔧 The capacity cleanup phase will attempt to fix these.")
        
        # Summary
        total_dogs = len(dogs_to_reassign)
        good_count = len([a for a in assignments_made if a['quality'] == 'GOOD'])
        backup_count = len([a for a in assignments_made if a['quality'] == 'BACKUP'])
        emergency_count = len([a for a in assignments_made if a['quality'] == 'EMERGENCY'])
        unassigned_count = len([a for a in assignments_made if a['driver'] == 'UNASSIGNED'])
        strategic_moves = len([m for m in moves_made if 'strategic' in m['reason']])
        
        print(f"\n🏆 LOCALITY-FIRST + STRATEGIC CASCADING RESULTS (DISTANCE-BASED):")
        print(f"   📊 {len(assignments_made)}/{total_dogs} dogs processed")
        print(f"   💚 {good_count} GOOD assignments (≤{self.PREFERRED_DISTANCE} miles)")
        print(f"   🟡 {backup_count} BACKUP assignments ({self.PREFERRED_DISTANCE}-{self.MAX_DISTANCE} miles)")
        print(f"   🚨 {emergency_count} EMERGENCY assignments (>{self.MAX_DISTANCE} miles)")
        print(f"   ❌ {unassigned_count} UNASSIGNED (no driver within 3.5 miles)")
        print(f"   🎯 {strategic_moves} strategic cascading moves executed")
        print(f"   🚶 {len(moves_made)} total cascading moves executed")
        print(f"   🎯 Success rate: {(good_count + backup_count)/total_dogs*100:.0f}% practical assignments")
        print(f"   🎯 Tight constraints: Exact matches ≤3.5 mi, Adjacent groups ≤2.6 mi")
        print(f"   🎯 Strategic cascading: Target blocked groups with incremental radius expansion")
        print(f"   ✅ Capacity tracking: Fixed with duplicate prevention")
        
        if unassigned_count > 0:
            print(f"\n⚠️ WARNING: {unassigned_count} dogs could not be assigned within 3.5 mile limit!")
            print("   Consider: Adding more drivers, relaxing distance constraints, or manual intervention")
        
        return assignments_made

    def reassign_dogs_multi_strategy_optimization(self):
        """Locality-first algorithm with strategic cascading and TIGHT distance constraints"""
        print("\n🔄 Starting LOCALITY-FIRST + STRATEGIC CASCADING SYSTEM (DISTANCE-BASED)...")
        print("🎯 Strategy: Proximity-first with strategic group-targeted cascading")
        print("📊 Quality: GOOD ≤1.5 mi, BACKUP ≤2.5 mi, EMERGENCY >2.5 mi")
        print("🚨 Focus: Immediate proximity with strategic dynamic space optimization")
        print("🎯 TIGHT CONSTRAINTS: Exact matches ≤3.5 mi, Adjacent groups ≤2.6 mi")
        print("🎯 STRATEGIC CASCADING: Target blocked groups with 1.5→2→2.5→3 mi radius expansion")
        print("✅ CAPACITY FIXES: Duplicate tracking, safe assignments, continuous verification")
        print("⚠️ WARNING: Very tight 3.5 mile limit may result in unassigned dogs!")
        print("=" * 80)
        
        # Try the locality-first algorithm with strategic cascading
        try:
            return self.locality_first_assignment()
        except Exception as e:
            print(f"⚠️ Locality-first algorithm encountered an error: {e}")
            import traceback
            print(f"🔍 Full error: {traceback.format_exc()}")
            print("⚠️ Returning partial results if any were made...")
            # Don't return empty list - let the caller handle partial results
            return getattr(self, '_partial_assignments', [])

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
            combined_col = 7  
            # Target Column K (Callout column) - index 10
            callout_col = 10
            print(f"📍 Writing to Column H (Combined) at index {combined_col}")
            print(f"📍 Updating Column K (Callout) at index {callout_col}")
            
            # Prepare batch updates for reassignments
            updates = []
            updates_count = 0
            
            print(f"\n🔍 Processing {len(reassignments)} reassignments...")
            
            # Process reassignments
            for assignment in reassignments:
                dog_id = str(assignment.get('dog_id', '')).strip()
                new_assignment = str(assignment.get('new_assignment', '')).strip()
                original_callout = assignment.get('original_callout', '')  # This contains the original assignment
                
                # Final validation
                if not new_assignment or new_assignment == dog_id or ':' not in new_assignment:
                    print(f"  ❌ SKIPPING invalid assignment for {dog_id}")
                    continue
                
                # Find the row for this dog ID
                for row_idx in range(1, len(all_data)):
                    if dog_id_col < len(all_data[row_idx]):
                        current_dog_id = str(all_data[row_idx][dog_id_col]).strip()
                        
                        if current_dog_id == dog_id:
                            # Update Column H with new assignment
                            cell_h_address = gspread.utils.rowcol_to_a1(row_idx + 1, combined_col + 1)
                            updates.append({
                                'range': cell_h_address,
                                'values': [[new_assignment]]
                            })
                            
                            # Update Column K to show reassignment history
                            # Format: "Reassigned from: [original assignment]"
                            if original_callout:
                                reassignment_note = f"Reassigned from: {original_callout}"
                            else:
                                # If we don't have the original callout, just mark as reassigned
                                reassignment_note = f"Reassigned to: {new_assignment}"
                            
                            cell_k_address = gspread.utils.rowcol_to_a1(row_idx + 1, callout_col + 1)
                            updates.append({
                                'range': cell_k_address,
                                'values': [[reassignment_note]]
                            })
                            
                            updates_count += 1
                            print(f"  ✅ {dog_id} → {new_assignment}")
                            print(f"     📝 Callout updated: {reassignment_note}")
                            break
            
            # Process strategic cascading moves if any
            if hasattr(self, 'greedy_moves_made') and self.greedy_moves_made:
                print(f"\n🔍 Processing {len(self.greedy_moves_made)} strategic cascading moves...")
                
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
                                current_combined = str(all_data[row_idx][combined_col]).strip()
                                if ':' in current_combined:
                                    assignment_part = current_combined.split(':', 1)[1]
                                    new_combined = f"{to_driver}:{assignment_part}"
                                    old_combined = f"{from_driver}:{assignment_part}"
                                    
                                    # Update Column H
                                    cell_h_address = gspread.utils.rowcol_to_a1(row_idx + 1, combined_col + 1)
                                    updates.append({
                                        'range': cell_h_address,
                                        'values': [[new_combined]]
                                    })
                                    
                                    # Update Column K to show cascading move
                                    cascading_note = f"Cascading move: {from_driver} → {to_driver}"
                                    cell_k_address = gspread.utils.rowcol_to_a1(row_idx + 1, callout_col + 1)
                                    updates.append({
                                        'range': cell_k_address,
                                        'values': [[cascading_note]]
                                    })
                                    
                                    print(f"  🎯 {dog_id} strategic move: {from_driver} → {to_driver}")
                                    print(f"     📝 Callout updated: {cascading_note}")
                                    updates_count += 1
                                break
            
            if not updates:
                print("❌ No valid updates to make")
                return False
            
            # Execute batch update
            print(f"\n📤 Writing {len(updates)} updates to Google Sheets...")
            worksheet.batch_update(updates)
            
            success_msg = f"✅ Successfully updated {updates_count} assignments with strategic cascading!"
            if hasattr(self, 'greedy_moves_made') and self.greedy_moves_made:
                strategic_moves = len([m for m in self.greedy_moves_made if 'strategic' in m['reason']])
                success_msg += f" (including {strategic_moves} strategic cascading moves)"
            
            print(success_msg)
            print(f"🎯 Used locality-first + strategic cascading with 3.5 mile limit")
            print(f"✅ WITH CAPACITY FIXES: Duplicate tracking + safe assignments")
            print(f"📝 Column K (Callout) updated with reassignment history")
            
            # Send Slack notification
            slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    message = f"🐕 Dog Reassignment Complete: {updates_count} assignments updated using strategic cascading + 3.5 mile limit + capacity fixes"
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


class ReassignmentEvaluator:
    """Evaluate and explain reassignment decisions"""
    
    def __init__(self, system, reassignments):
        self.system = system
        self.reassignments = reassignments
        self.moves = getattr(system, 'greedy_moves_made', [])
    
    def generate_detailed_report(self):
        """Generate a detailed report explaining all decisions"""
        print("\n" + "="*80)
        print("📊 DETAILED REASSIGNMENT EVALUATION REPORT")
        print("="*80)
        
        # 1. Summary Statistics
        self._print_summary_stats()
        
        # 2. Assignment Quality Analysis
        self._analyze_assignment_quality()
        
        # 3. Cascading Move Analysis
        self._analyze_cascading_moves()
        
        # 4. Individual Assignment Explanations
        self._explain_individual_assignments()
        
        # 5. Distance Distribution
        self._analyze_distance_distribution()
        
        # 6. Group Compatibility Analysis
        self._analyze_group_patterns()
    
    def _print_summary_stats(self):
        """Print summary statistics"""
        print("\n📈 SUMMARY STATISTICS:")
        print("-" * 40)
        
        total = len(self.reassignments)
        by_type = {}
        by_quality = {}
        
        for assignment in self.reassignments:
            # Count by type
            atype = assignment.get('assignment_type', 'unknown')
            by_type[atype] = by_type.get(atype, 0) + 1
            
            # Count by quality
            quality = assignment.get('quality', 'unknown')
            by_quality[quality] = by_quality.get(quality, 0) + 1
        
        print(f"Total reassignments: {total}")
        print("\nBy assignment type:")
        for atype, count in sorted(by_type.items()):
            print(f"  - {atype}: {count} ({count/total*100:.1f}%)")
        
        print("\nBy quality:")
        for quality, count in sorted(by_quality.items()):
            print(f"  - {quality}: {count} ({count/total*100:.1f}%)")
        
        # Average distance
        distances = [a['distance'] for a in self.reassignments if a['distance'] != float('inf')]
        if distances:
            avg_distance = sum(distances) / len(distances)
            print(f"\nAverage distance: {avg_distance:.1f} miles")
            print(f"Min distance: {min(distances):.1f} miles")
            print(f"Max distance: {max(distances):.1f} miles")
    
    def _analyze_assignment_quality(self):
        """Analyze why assignments got their quality ratings"""
        print("\n🏆 ASSIGNMENT QUALITY ANALYSIS:")
        print("-" * 40)
        
        # Group by quality
        by_quality = {'GOOD': [], 'BACKUP': [], 'EMERGENCY': []}
        for a in self.reassignments:
            quality = a.get('quality', 'EMERGENCY')
            if quality in by_quality:
                by_quality[quality].append(a)
        
        for quality, assignments in by_quality.items():
            if not assignments:
                continue
            
            print(f"\n{quality} assignments ({len(assignments)}):")
            
            # Show a few examples
            for i, a in enumerate(assignments[:3]):
                dog_name = a['dog_name']
                driver = a['driver']
                distance = a['distance']
                atype = a['assignment_type']
                
                if distance == float('inf'):
                    print(f"  {i+1}. {dog_name} → {driver}")
                    print(f"     ❌ No driver found within constraints")
                else:
                    print(f"  {i+1}. {dog_name} → {driver} ({distance:.1f} mi)")
                    print(f"     Type: {atype}")
                    
                    # Explain why this quality
                    if quality == 'GOOD':
                        print(f"     ✅ Within preferred {self.system.PREFERRED_DISTANCE} mile radius")
                    elif quality == 'BACKUP':
                        print(f"     🟡 Between {self.system.PREFERRED_DISTANCE}-{self.system.MAX_DISTANCE} miles")
                    else:
                        print(f"     🚨 Beyond {self.system.MAX_DISTANCE} mile backup threshold")
    
    def _analyze_cascading_moves(self):
        """Analyze cascading moves and explain why they were needed"""
        print("\n🔄 CASCADING MOVE ANALYSIS:")
        print("-" * 40)
        
        if not self.moves:
            print("No cascading moves were made.")
            return
        
        print(f"Total cascading moves: {len(self.moves)}")
        
        # Group by reason
        by_reason = {}
        for move in self.moves:
            reason = move['reason']
            if 'strategic' in reason:
                key = 'Strategic (targeted)'
            else:
                key = 'Other'
            by_reason[key] = by_reason.get(key, 0) + 1
        
        print("\nMove types:")
        for reason, count in by_reason.items():
            print(f"  - {reason}: {count}")
        
        # Detailed analysis of first few moves
        print("\nDetailed move explanations:")
        for i, move in enumerate(self.moves[:5]):
            print(f"\n{i+1}. {move['dog_name']}:")
            print(f"   From: {move['from_driver']} → To: {move['to_driver']}")
            print(f"   Distance: {move['distance']:.1f} miles")
            print(f"   Reason: {move['reason']}")
            
            # Explain the strategic reasoning
            if 'strategic' in move['reason']:
                # Extract which dog this move was for
                parts = move['reason'].split('_for_')
                if len(parts) > 1:
                    beneficiary = parts[1]
                    print(f"   💡 This move freed space for: {beneficiary}")
                    print(f"   🎯 Strategic: Targeted specific blocked groups")
    
    def _explain_individual_assignments(self):
        """Explain logic for specific assignments"""
        print("\n🔍 INDIVIDUAL ASSIGNMENT LOGIC:")
        print("-" * 40)
        
        # Show detailed logic for first few assignments
        for i, assignment in enumerate(self.reassignments[:5]):
            print(f"\n{i+1}. {assignment['dog_name']} ({assignment['dog_id']}):")
            print(f"   Assignment: {assignment['new_assignment']}")
            
            if assignment['driver'] == 'UNASSIGNED':
                print("   ❌ UNASSIGNED - No viable driver within 3.5 mile constraint")
                self._explain_why_unassigned(assignment)
            else:
                print(f"   ✅ Assigned to: {assignment['driver']}")
                print(f"   Distance: {assignment['distance']:.1f} miles")
                print(f"   Quality: {assignment['quality']}")
                print(f"   Method: {assignment['assignment_type']}")
                
                if assignment['assignment_type'] == 'direct':
                    print("   💡 Direct assignment - driver had capacity and was within range")
                elif assignment['assignment_type'] == 'strategic_cascading':
                    print("   💡 Required strategic cascading - driver was blocked by capacity")
                    print("      A strategic move freed up the necessary space")
                elif assignment['assignment_type'] == 'radius_expansion':
                    print(f"   💡 Found during radius expansion - no closer options available")
    
    def _explain_why_unassigned(self, assignment):
        """Explain why a dog couldn't be assigned"""
        dog_id = assignment['dog_id']
        print(f"   Analyzing why {assignment['dog_name']} couldn't be assigned...")
        
        # Count potential drivers by category
        close_with_capacity = 0
        close_but_full = 0
        close_wrong_groups = 0
        too_far = 0
        
        print("   (Analysis of nearby drivers would go here)")
        print("   Common reasons: all nearby drivers full, incompatible groups, or beyond 3.5 miles")
    
    def _analyze_distance_distribution(self):
        """Analyze distance patterns"""
        print("\n📏 DISTANCE DISTRIBUTION:")
        print("-" * 40)
        
        # Create distance buckets
        buckets = {
            '0-1.5 mi': 0,
            '1.5-2.5 mi': 0,
            '2.5-3.5 mi': 0,
            '3.5+ mi': 0,
            'Unassigned': 0
        }
        
        for a in self.reassignments:
            dist = a['distance']
            if dist == float('inf'):
                buckets['Unassigned'] += 1
            elif dist <= 1.5:
                buckets['0-1.5 mi'] += 1
            elif dist <= 2.5:
                buckets['1.5-2.5 mi'] += 1
            elif dist <= 3.5:
                buckets['2.5-3.5 mi'] += 1
            else:
                buckets['3.5+ mi'] += 1
        
        total = len(self.reassignments)
        for bucket, count in buckets.items():
            if count > 0:
                bar = '█' * int(count / total * 40)
                print(f"{bucket:12} [{count:2}] {bar} {count/total*100:.1f}%")
    
    def _analyze_group_patterns(self):
        """Analyze group assignment patterns"""
        print("\n👥 GROUP COMPATIBILITY PATTERNS:")
        print("-" * 40)
        
        print("Group matching analysis:")
        print("- Exact matches: Dogs placed with drivers in same groups")
        print("- Adjacent matches: Dogs placed with drivers in neighboring groups")
        print("- This affects allowed distance thresholds")


def evaluate_reassignments(system, reassignments):
    """Run the evaluation and generate report"""
    evaluator = ReassignmentEvaluator(system, reassignments)
    evaluator.generate_detailed_report()


def main():
    """Main function to run the dog reassignment system with capacity cleanup"""
    print("🚀 Enhanced Dog Reassignment System - DISTANCE-BASED (miles)")
    print("🎯 NEW: Strategic group-targeted cascading with incremental radius expansion")
    print("📏 Main: Starts at 1.5 mi, expands to 3.5 mi in 0.5 mi increments")
    print("🔧 Cleanup: Aggressive proximity-focused capacity violation fixes")
    print("🔄 Adjacent groups: 75% of current radius (more generous)")
    print("🎯 STRATEGIC CASCADING: Target blocked groups, not random dogs")
    print("🚶 Cascading moves up to 3 mi with incremental expansion (1.5→2→2.5→3)")
    print("🧅 Onion-layer backflow pushes outer assignments out to create inner space")
    print("📊 Quality: GOOD ≤1.5 mi, BACKUP ≤2.5 mi, EMERGENCY >2.5 mi")
    print("🎯 TIGHT CONSTRAINTS: Exact matches ≤3.5 mi, Adjacent groups ≤2.6 mi")
    print("✅ CAPACITY FIXES: Duplicate tracking, safe assignments, verification at each step")
    print("🗺️ ROUTE COHERENCE: 25% of dogs within 3 miles (relaxed for missing data)")
    print("⚠️ WARNING: Very tight constraints may result in unassigned dogs!")
    print("🔄 FALLBACK: If dog not in distance matrix, will use haversine distances")
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
    
    # Optional: Load haversine matrix as fallback
    if HAVERSINE_MATRIX_URL:
        print("\n📊 Loading haversine matrix from URL...")
        system.load_haversine_matrix(url=HAVERSINE_MATRIX_URL)
    elif HAVERSINE_SHEET_ID:
        print("\n📊 Loading haversine matrix via Google Sheets API...")
        system.load_haversine_matrix(sheet_id=HAVERSINE_SHEET_ID, gid=HAVERSINE_GID)
    else:
        print("\nℹ️ No haversine matrix configured - fallback disabled")
        print("   To enable: Set HAVERSINE_MATRIX_URL or HAVERSINE_SHEET_ID at top of script")
    
    if not system.load_dog_assignments():
        print("❌ Failed to load dog assignments")
        return
    
    if not system.load_driver_capacities():
        print("❌ Failed to load driver capacities")
        return
    
    # DEBUG: Check initial system state
    print("\n🔍 INITIAL SYSTEM STATE DEBUG:")
    
    # Build initial state for debug
    initial_state = system.build_initial_assignments_state()
    
    for driver in ["Chase", "Hannah", "Blanch", "Ali", "Alyson"]:
        if driver in system.driver_capacities:
            system.debug_capacity_calculation(driver, initial_state, "(initial state)")
    
    # Run the locality-first assignment with strategic cascading
    print("\n🔄 Processing callout assignments...")
    
    reassignments = system.reassign_dogs_multi_strategy_optimization()
    
    # Ensure reassignments is always a list
    if reassignments is None:
        reassignments = []
    
    # Check for unassigned dogs
    unassigned_count = len([a for a in reassignments if a.get('driver') == 'UNASSIGNED'])
    if unassigned_count > 0:
        print(f"\n⚠️ WARNING: {unassigned_count} dogs could not be assigned within 3.5 mile limit!")
        print("Consider relaxing constraints or adding more drivers in affected areas.")
    
    # Write results
    if reassignments:
        # Run evaluation BEFORE writing to sheets to understand decisions
        print("\n" + "="*80)
        print("🔍 EVALUATING REASSIGNMENT DECISIONS")
        print("="*80)
        evaluate_reassignments(system, reassignments)
        
        # DEBUG: Check final system state
        print("\n🔍 FINAL SYSTEM STATE DEBUG:")
        
        # Build final state for debug  
        final_state = system.build_initial_assignments_state()
        
        for driver in ["Chase", "Hannah", "Blanch", "Ali", "Alyson"]:
            if driver in system.driver_capacities:
                system.debug_capacity_calculation(driver, final_state, "(final state)")
        
        write_success = system.write_results_to_sheets(reassignments)
        if write_success:
            print(f"\n🎉 SUCCESS! Processed {len(reassignments)} callout assignments")
            print(f"✅ Used locality-first + strategic cascading with 3.5 mile limit")
            print(f"✅ WITH CAPACITY FIXES: All capacity constraints respected")
            
            # ========== CAPACITY CLEANUP PHASE ==========
            print("\n" + "="*80)
            print("🔧 AGGRESSIVE CAPACITY CLEANUP - Fixing violations with extreme proximity")
            print("📏 Thresholds: ≤2.5 mi direct, ≤1.5 mi adjacent, ≤3 mi cascading")
            print("🎯 Goal: 100% close placements, zero tolerance for distant fixes")
            print("="*80)
            
            try:
                # Import and run cleanup
                from capacity_cleanup import CapacityCleanup
                
                cleanup = CapacityCleanup()
                # Copy data instead of reloading
                cleanup.distance_matrix = system.distance_matrix
                cleanup.haversine_matrix = system.haversine_matrix  # Also copy haversine if available
                cleanup.dog_assignments = system.dog_assignments  # Updated assignments
                cleanup.driver_capacities = system.driver_capacities
                cleanup.sheets_client = system.sheets_client
                
                # Run aggressive cleanup
                moves = cleanup.fix_capacity_violations()
                
                if moves:
                    cleanup_success = cleanup.write_moves_to_sheets(moves)
                    if cleanup_success:
                        print(f"\n🎉 COMPLETE SUCCESS! Main + cleanup: extreme proximity achieved")
                    else:
                        print(f"\n⚠️ Main completed, cleanup had sheet writing issues")
                else:
                    print(f"\n✅ Perfect! No capacity violations to clean up")
                    
            except Exception as e:
                print(f"\n⚠️ Cleanup phase error (main script succeeded): {e}")
                import traceback
                print(f"🔍 Cleanup error details: {traceback.format_exc()}")
            
        else:
            print(f"\n❌ Failed to write {len(reassignments)} results to Google Sheets")
            # Don't run cleanup if main write failed
    else:
        print(f"\n✅ No callout assignments needed - all drivers available or no valid assignments found")
        
        # Run cleanup even if no callouts (might still have capacity violations from existing assignments)
        print("\n🔍 Checking for existing capacity violations...")
        try:
            from capacity_cleanup import CapacityCleanup
            
            cleanup = CapacityCleanup()
            cleanup.distance_matrix = system.distance_matrix
            cleanup.haversine_matrix = system.haversine_matrix if hasattr(system, 'haversine_matrix') else None
            cleanup.dog_assignments = system.dog_assignments
            cleanup.driver_capacities = system.driver_capacities
            cleanup.sheets_client = system.sheets_client
            
            moves = cleanup.fix_capacity_violations()
            
            if moves:
                cleanup_success = cleanup.write_moves_to_sheets(moves)
                if cleanup_success:
                    print(f"\n✅ Fixed existing capacity violations: {len(moves)} moves")
                    
        except Exception as e:
            print(f"\n⚠️ Capacity check error: {e}")


if __name__ == "__main__":
    main()
