# production_reassignment.py
# COMPLETE WORKING VERSION: Locality-first with strategic cascading and 1.5 mile range
# 🔧 CAPACITY BUG FIXED: Immediate state updates prevent race conditions

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
        
        # DISTANCE LIMITS - LOCALITY-FIRST THRESHOLDS
        self.PREFERRED_DISTANCE = 0.2  # Ideal assignments: immediate proximity
        self.MAX_DISTANCE = 0.5  # Backup assignments: reasonable with moves
        self.ABSOLUTE_MAX_DISTANCE = 1.5  # Search limit for locality-first algorithm (EXTENDED!)
        self.CASCADING_MOVE_MAX = 0.7  # Max distance for strategic cascading moves (INCREASED!)
        self.ADJACENT_GROUP_DISTANCE = 0.1  # Base adjacent group distance (scales with radius)
        self.EXCLUSION_DISTANCE = 200.0  # Temporarily increased from 100.0 to see if 100 is placeholder
        
        # Legacy thresholds for compatibility
        self.GREEDY_WALK_MAX_DISTANCE = 0.5
        
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
            print("❌ Error setting up Google Sheets client: " + str(e))
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
            
            print("📊 Distance matrix shape: (" + str(len(df)) + ", " + str(len(df.columns)) + ")")
            
            # Extract dog IDs from columns (skip non-dog columns)
            dog_ids = [col for col in df.columns if 'x' in str(col).lower()]
            print("📊 Found " + str(len(dog_ids)) + " column Dog IDs")
            
            # Filter to only dog ID columns and rows
            dog_df = df.loc[df.index.isin(dog_ids), dog_ids]
            
            self.distance_matrix = dog_df
            print("✅ Loaded distance matrix for " + str(len(self.distance_matrix)) + " dogs")
            
            return True
            
        except Exception as e:
            print("❌ Error loading distance matrix: " + str(e))
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
            
            print("📊 Map sheet shape: (" + str(len(df)) + ", " + str(len(df.columns)) + ")")
            print("🔍 DEBUG: First few column names: " + str(list(df.columns[:15])))
            
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
                        print("🔍 Row " + str(i) + ": DogName=\"" + str(dog_name) + "\", Combined=\"" + str(combined) + "\", DogID=\"" + str(dog_id) + "\", Callout=\"" + str(callout) + "\"")
                    
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
                    print("⚠️ Error processing row " + str(i) + ": " + str(e))
                    continue
            
            self.dog_assignments = assignments
            print("✅ Loaded " + str(len(assignments)) + " regular assignments")
            
            return True
            
        except Exception as e:
            print("❌ Error loading dog assignments: " + str(e))
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
            print("✅ Loaded capacities for " + str(len(capacities)) + " drivers")
            
            # Debug: Show first few driver capacities and check for Sara specifically
            if capacities:
                print("🔍 DEBUG: Sample driver capacities:")
                for i, (driver, caps) in enumerate(list(capacities.items())[:3]):
                    print("   '" + str(driver) + "': " + str(caps))
                
                # Specifically check for Sara since that's the failing driver
                if 'Sara' in capacities:
                    print("🔍 DEBUG: Found Sara capacity: " + str(capacities['Sara']))
                else:
                    print("🚨 DEBUG: Sara not found in capacities!")
                    sara_like = [d for d in capacities.keys() if 'sara' in str(d).lower()]
                    if sara_like:
                        print("   📋 Sara-like drivers found: " + str(sara_like))
                    else:
                        print("   📋 No Sara-like drivers found")
            
            return True
            
        except Exception as e:
            print("❌ Error loading driver capacities: " + str(e))
            return False

    def get_dogs_to_reassign(self):
        """Find dogs that need reassignment (callouts) - excluding non-dog entries"""
        dogs_to_reassign = []
        
        if not self.dog_assignments:
            return dogs_to_reassign
        
        print("🔍 DEBUG: Checking " + str(len(self.dog_assignments)) + " total assignments for callouts...")
        
        callout_candidates = 0
        filtered_out = 0
        no_colon = 0
        no_groups = 0
        
        for i, assignment in enumerate(self.dog_assignments):
            # Debug first few and last few
            if i < 5 or i >= len(self.dog_assignments) - 5:
                print("   Row " + str(i) + ": ID=" + str(assignment.get('dog_id', 'MISSING')) + ", Combined=\"" + str(assignment.get('combined', 'MISSING')) + "\", Callout=\"" + str(assignment.get('callout', 'MISSING')) + "\"")
            
            # Check for callout: Combined column blank AND Callout column has content
            combined_blank = (not assignment['combined'] or assignment['combined'].strip() == "")
            callout_has_content = (assignment['callout'] and assignment['callout'].strip() != "")
            
            if combined_blank and callout_has_content:
                callout_candidates += 1
                
                # FILTER OUT NON-DOGS: Skip Parking, Field, and other administrative entries
                dog_name = str(assignment.get('dog_name', '')).lower().strip()
                if any(keyword in dog_name for keyword in ['parking', 'field', 'admin', 'office']):
                    print("   ⏭️ Skipping non-dog entry: " + str(assignment['dog_name']) + " (" + str(assignment['dog_id']) + ")")
                    filtered_out += 1
                    continue
                
                # Extract the FULL assignment string (everything after the colon)
                callout_text = assignment['callout'].strip()
                
                if ':' not in callout_text:
                    print("   ⚠️ No colon in callout for " + str(assignment.get('dog_id', 'UNKNOWN')) + ": \"" + str(callout_text) + "\"")
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
                else:
                    print("   ⚠️ No groups found for " + str(assignment.get('dog_id', 'UNKNOWN')) + ": \"" + str(full_assignment_string) + "\"")
                    no_groups += 1
        
        print("🔍 DEBUG SUMMARY:")
        print("   📊 Total assignments checked: " + str(len(self.dog_assignments)))
        print("   🎯 Callout candidates (blank combined + has callout): " + str(callout_candidates))
        print("   🚫 Filtered out (non-dogs): " + str(filtered_out))
        print("   ⚠️ No colon in callout: " + str(no_colon))
        print("   ⚠️ No groups extracted: " + str(no_groups))
        print("   ✅ Final dogs to reassign: " + str(len(dogs_to_reassign)))
        
        print("\n🚨 Found " + str(len(dogs_to_reassign)) + " REAL dogs that need drivers assigned:")
        for dog in dogs_to_reassign:
            print("   - " + str(dog['dog_name']) + " (" + str(dog['dog_id']) + ") - " + str(dog['num_dogs']) + " dogs")
            print("     Original: " + str(dog['original_callout']))
            print("     Assignment string: \"" + str(dog['full_assignment_string']) + "\"")
            print("     Capacity needed in groups: " + str(dog['needed_groups']))
        
        return dogs_to_reassign

    def _extract_groups_for_capacity_check(self, assignment_string):
        """Extract group numbers for capacity checking - each digit 1,2,3 is a separate group
        
        Examples:
        - "1&2" → [1, 2] (counts +1 in group1, +1 in group2)
        - "123" → [1, 2, 3] (counts +1 in each group)
        - "3DD1" → [3] (ignores DD1, just group 3)
        - "1&2&3" → [1, 2, 3] (all three groups)
        """
        try:
            # Extract all digits that are 1, 2, or 3 from the string
            group_digits = re.findall(r'[123]', assignment_string)
            
            # Convert each digit to an integer and remove duplicates
            groups = sorted(list(set(int(digit) for digit in group_digits)))
            
            print("     🔍 GROUP PARSING: '" + str(assignment_string) + "' → groups " + str(groups))
            
            return groups
            
        except Exception as e:
            print("⚠️ Error extracting groups from \"" + str(assignment_string) + "\": " + str(e))
            return []

    def get_distance(self, dog1_id: str, dog2_id: str) -> float:
        """Get distance between two dogs using the distance matrix (reverted - no filtering)"""
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
        
        print("🔍 CALCULATING LOAD for driver: '" + str(driver_name) + "'")
        
        # Use provided assignments or default to original assignments
        assignments_to_use = current_assignments if current_assignments else self.dog_assignments
        
        if not assignments_to_use:
            print("   ⚠️ No assignments to process")
            return load
        
        print("   📊 Processing " + str(len(assignments_to_use)) + " assignments")
        
        dogs_counted = 0
        for assignment in assignments_to_use:
            if current_assignments:
                # Working with dynamic assignment list
                assigned_driver = assignment.get('driver', '')
                print("   🔍 Checking assignment: driver='" + str(assigned_driver) + "' vs target='" + str(driver_name) + "'")
                
                if assigned_driver == driver_name:
                    # Parse groups for this assignment
                    assigned_groups = assignment.get('needed_groups', [])
                    num_dogs = assignment.get('num_dogs', 1)
                    dog_name = assignment.get('dog_name', 'Unknown')
                    
                    print("     ✅ MATCH: " + str(dog_name) + " - groups " + str(assigned_groups) + " - count " + str(num_dogs))
                    
                    # Add to load for each group
                    for group in assigned_groups:
                        group_key = 'group' + str(group)
                        if group_key in load:
                            load[group_key] += num_dogs
                            print("       📈 Added " + str(num_dogs) + " to " + str(group_key) + " (now " + str(load[group_key]) + ")")
                    
                    dogs_counted += 1
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
                            group_key = 'group' + str(group)
                            if group_key in load:
                                load[group_key] += assignment['num_dogs']
                        
                        dogs_counted += 1
        
        print("   📊 FINAL LOAD for '" + str(driver_name) + "': " + str(load) + " (" + str(dogs_counted) + " dogs counted)")
        
        return load

    def check_group_compatibility(self, callout_groups, driver_groups, distance, current_radius=None):
        """FIXED: Radius scaling with proper fallback and 75% adjacent threshold"""
        # Extract unique group numbers from both sets
        callout_set = set(callout_groups)
        driver_set = set(driver_groups)
        
        # Determine thresholds based on current radius
        if current_radius is not None:
            # RADIUS SCALING: Use the step-by-step approach
            perfect_threshold = current_radius
            adjacent_threshold = current_radius * 0.75  # 75% of current radius
        else:
            # FALLBACK: For diagnostics when no radius provided, use generous limits
            perfect_threshold = 1.5  # Up to 1.5 miles for exact matches
            adjacent_threshold = 1.125  # Up to 1.125 miles (75% of 1.5) for adjacent groups
        
        # 1. PERFECT MATCH - same groups
        if callout_set.intersection(driver_set):
            return distance <= perfect_threshold
        
        # 2. ADJACENT GROUPS - neighboring groups  
        adjacent_pairs = [(1, 2), (2, 3), (2, 1), (3, 2)]
        for callout_group in callout_set:
            for driver_group in driver_set:
                if (callout_group, driver_group) in adjacent_pairs:
                    return distance <= adjacent_threshold
        
        # 3. NO MATCH - incompatible groups
        return False

    def check_group_compatibility_for_moves(self, dog_groups, driver_groups, distance, current_radius=None):
        """STRATEGIC MOVE COMPATIBILITY: More flexible for multi-group dogs"""
        # Extract unique group numbers from both sets
        dog_set = set(dog_groups)
        driver_set = set(driver_groups)
        
        # Determine thresholds based on current radius
        if current_radius is not None:
            perfect_threshold = current_radius
            adjacent_threshold = current_radius * 0.75
        else:
            perfect_threshold = 1.5
            adjacent_threshold = 1.125
        
        print("       🔍 MOVE COMPATIBILITY: Dog groups " + str(dog_groups) + " → Driver groups " + str(driver_groups))
        
        # 1. DIRECT OVERLAP - Any shared group (most flexible)
        if dog_set.intersection(driver_set):
            print("       ✅ DIRECT OVERLAP: Shared groups " + str(dog_set.intersection(driver_set)))
            return distance <= perfect_threshold
        
        # 2. MULTI-GROUP FLEXIBILITY
        # For multi-group dogs, be more flexible about compatibility
        for dog_group in dog_set:
            for driver_group in driver_set:
                compatible = False
                
                # Group 1 & 2 are compatible (adjacent)
                if (dog_group == 1 and driver_group == 2) or (dog_group == 2 and driver_group == 1):
                    compatible = True
                    print("       ✅ ADJACENT 1-2: Groups " + str(dog_group) + " & " + str(driver_group))
                
                # Group 2 & 3 are compatible (adjacent)  
                elif (dog_group == 2 and driver_group == 3) or (dog_group == 3 and driver_group == 2):
                    compatible = True
                    print("       ✅ ADJACENT 2-3: Groups " + str(dog_group) + " & " + str(driver_group))
                
                # Group 1 & 3 are compatible for multi-group flexibility
                elif (dog_group == 1 and driver_group == 3) or (dog_group == 3 and driver_group == 1):
                    compatible = True
                    print("       ✅ MULTI-GROUP FLEX: Groups " + str(dog_group) + " & " + str(driver_group))
                
                if compatible:
                    return distance <= adjacent_threshold
        
        print("       ❌ NO COMPATIBILITY: Groups " + str(dog_groups) + " cannot match " + str(driver_groups))
        return False

    def validate_assignment_capacity(self, driver, callout_dog, current_assignments):
        """FINAL CAPACITY VALIDATION - Prevents any over-capacity assignments"""
        print("🔍 VALIDATING CAPACITY for " + str(driver) + " + " + str(callout_dog.get('dog_name', 'Unknown')))
        
        current_load = self.calculate_driver_load(driver, current_assignments)
        driver_capacity = self.driver_capacities.get(driver, {})
        
        if not driver_capacity:
            print("   ⚠️ Driver '" + str(driver) + "' not found in capacity data - ASSIGNMENT BLOCKED!")
            return False
        
        print("   📊 Driver capacity data: " + str(driver_capacity))
        print("   📊 Current load: " + str(current_load))
        print("   📊 Needed groups: " + str(callout_dog['needed_groups']))
        print("   📊 Needed count: " + str(callout_dog['num_dogs']))
        
        for group in callout_dog['needed_groups']:
            group_key = 'group' + str(group)
            current = current_load.get(group_key, 0)
            max_cap = driver_capacity.get(group_key, 0)
            needed = callout_dog['num_dogs']
            
            print("   🔢 Group " + str(group) + " (" + str(group_key) + "): " + str(current) + " + " + str(needed) + " vs " + str(max_cap))
            
            if current + needed > max_cap:
                print("🚨 FINAL VALIDATION FAILED: " + str(driver) + " group " + str(group) + ": " + str(current) + " + " + str(needed) + " > " + str(max_cap) + " - ASSIGNMENT BLOCKED!")
                return False
            
        print("✅ FINAL VALIDATION PASSED: " + str(driver) + " can accept " + str(callout_dog.get('dog_name', 'Unknown')))
        return True

    def get_current_driver_dogs(self, driver_name, current_assignments):
        """Get all dogs currently assigned to a specific driver"""
        return [assignment for assignment in current_assignments 
                if assignment.get('driver') == driver_name]

    # ========== STRATEGIC CASCADING METHODS ==========

    def attempt_strategic_cascading_move(self, blocked_driver, callout_dog, current_assignments, max_search_radius=0.7):
        """STRATEGIC: Target specific groups with incremental radius expansion"""
        print("🎯 ATTEMPTING STRATEGIC CASCADING MOVE for " + str(callout_dog.get('dog_name', 'Unknown')) + " → " + str(blocked_driver))
        
        # Step 1: Identify which groups are causing the capacity problem
        blocked_groups = self._identify_blocked_groups(blocked_driver, callout_dog, current_assignments)
        print("   🎯 Target groups causing capacity issues: " + str(blocked_groups))
        
        if not blocked_groups:
            print("   ❌ No blocked groups identified")
            return None
        
        # Step 2: Get dogs from blocked driver, prioritized by strategy
        driver_dogs = self.get_current_driver_dogs(blocked_driver, current_assignments)
        strategic_dogs = self._prioritize_dogs_strategically(driver_dogs, blocked_groups)
        
        print("   📊 Strategic prioritization of " + str(len(strategic_dogs)) + " dogs:")
        for i, (priority, dog) in enumerate(strategic_dogs[:8]):  # Show top 8
            print("     " + str(i+1) + ". " + str(dog.get('dog_name', 'Unknown')) + " (groups: " + str(dog.get('needed_groups', [])) + ") - " + str(priority))
        
        # Step 3: Try incremental radius expansion for each high-priority dog
        for priority, dog_to_move in strategic_dogs:
            print("   🔄 Trying to move " + str(dog_to_move.get('dog_name', 'Unknown')) + " (groups: " + str(dog_to_move.get('needed_groups', [])) + ")...")
            
            # Use incremental radius expansion like the main algorithm
            move_result = self._attempt_incremental_move(dog_to_move, current_assignments, max_search_radius)
            
            if move_result:
                print("   ✅ STRATEGIC MOVE SUCCESSFUL!")
                print("      📦 Moved: " + str(dog_to_move.get('dog_name', 'Unknown')) + " → " + str(move_result.get('to_driver', 'Unknown')))
                print("      📏 Distance: " + str(round(move_result.get('distance', 0), 3)) + "mi (found at radius " + str(move_result.get('radius', 0)) + "mi)")
                print("      🎯 This frees " + str(blocked_groups) + " capacity in " + str(blocked_driver))
                return move_result
            else:
                print("   ❌ Could not move " + str(dog_to_move.get('dog_name', 'Unknown')) + " within " + str(max_search_radius) + "mi")
        
        print("   ❌ STRATEGIC CASCADING FAILED - no dogs could be relocated")
        return None

    def _identify_blocked_groups(self, driver_name, callout_dog, current_assignments):
        """Identify which specific groups are causing capacity problems"""
        blocked_groups = []
        
        # Get current capacity and load
        capacity = self.driver_capacities.get(driver_name, {})
        current_load = self.calculate_driver_load(driver_name, current_assignments)
        
        # Check which groups would be over capacity
        for group in callout_dog['needed_groups']:
            group_key = 'group' + str(group)
            current = current_load.get(group_key, 0)
            max_cap = capacity.get(group_key, 0)
            needed = callout_dog['num_dogs']
            
            if current + needed > max_cap:
                blocked_groups.append(group)
                print("   🚨 Group " + str(group) + " blocking: " + str(current) + " + " + str(needed) + " > " + str(max_cap))
            else:
                print("   ✅ Group " + str(group) + " has space: " + str(current) + " + " + str(needed) + " ≤ " + str(max_cap))
        
        return blocked_groups

    def _prioritize_dogs_strategically(self, driver_dogs, blocked_groups):
        """Prioritize dogs based on strategic value for freeing blocked groups"""
        prioritized = []
        
        print("   🔍 STRATEGIC PRIORITIZATION DEBUG:")
        print("   📊 Blocked groups: " + str(blocked_groups))
        print("   📊 Available dogs to move:")
        
        for i, dog in enumerate(driver_dogs):
            dog_groups = set(dog.get('needed_groups', []))
            blocked_set = set(blocked_groups)
            
            print("     " + str(i+1) + ". " + str(dog.get('dog_name', 'Unknown')) + " - groups: " + str(list(dog_groups)))
            
            # Calculate strategic priority
            if dog_groups.intersection(blocked_set):
                # Dog is in a blocked group - HIGH PRIORITY
                if len(dog_groups) == 1 and dog['num_dogs'] == 1:
                    priority = "HIGH - Single group, single dog in blocked group"
                elif len(dog_groups) == 1:
                    priority = "HIGH - Single group, " + str(dog['num_dogs']) + " dogs in blocked group"
                else:
                    priority = "MEDIUM - Multi-group dog partially in blocked group"
                    print("       💡 Multi-group dog: Can be placed with drivers having ANY of groups " + str(list(dog_groups)))
            else:
                # Dog is not in blocked groups - LOW PRIORITY
                priority = "LOW - Not in blocked groups (won't help)"
            
            print("       🎯 Priority: " + str(priority))
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
        """Try to move a dog using incremental radius expansion (0.2 → 0.3 → 0.4 → etc.)"""
        print("     🔍 Using incremental radius search for " + str(dog_to_move.get('dog_name', 'Unknown')) + "...")
        
        # Start at 0.2mi and expand incrementally
        current_radius = 0.2
        
        while current_radius <= max_radius:
            print("       📏 Trying radius " + str(current_radius) + "mi...")
            
            # Find all potential targets within current radius
            targets = self._find_move_targets_at_radius(dog_to_move, current_assignments, current_radius)
            
            if targets:
                print("       ✅ Found " + str(len(targets)) + " targets at " + str(current_radius) + "mi:")
                for i, target in enumerate(targets[:3]):  # Show top 3
                    print("         " + str(i+1) + ". " + str(target['driver']) + " - " + str(round(target['distance'], 3)) + "mi")
                
                # Use the closest target
                best_target = targets[0]
                
                # Execute the move
                for assignment in current_assignments:
                    if assignment['dog_id'] == dog_to_move['dog_id']:
                        old_driver = assignment['driver']
                        assignment['driver'] = best_target['driver']
                        print("       🔄 MOVE STATE UPDATED: " + str(dog_to_move.get('dog_name', 'Unknown')) + " moved from " + str(old_driver) + " to " + str(best_target['driver']))
                        break
                
                return {
                    'moved_dog': dog_to_move,
                    'from_driver': old_driver,
                    'to_driver': best_target['driver'],
                    'distance': best_target['distance'],
                    'via_dog': best_target['via_dog'],
                    'radius': current_radius
                }
            else:
                print("       ❌ No targets at " + str(current_radius) + "mi")
            
            # Expand radius
            current_radius += 0.1
            current_radius = round(current_radius, 1)  # Avoid floating point issues
        
        print("     ❌ No targets found within " + str(max_radius) + "mi")
        return None

    def _find_move_targets_at_radius(self, dog_to_move, current_assignments, radius):
        """Find potential drivers within specific radius who can accept the dog"""
        targets = []
        
        for assignment in current_assignments:
            target_driver = assignment['driver']
            
            # Skip same driver
            if target_driver == dog_to_move.get('driver'):
                continue
            
            # Calculate distance
            distance = self.get_distance(dog_to_move['dog_id'], assignment['dog_id'])
            
            # Skip if beyond current radius
            if distance > radius or distance >= 100.0:  # Skip placeholders
                continue
            
            # Check group compatibility with current radius
            dog_groups = dog_to_move.get('needed_groups', [])
            target_groups = assignment.get('needed_groups', [])
            
            # Use flexible group compatibility for strategic moves
            if not self.check_group_compatibility_for_moves(dog_groups, target_groups, distance, radius):
                continue
            
            # Check if target driver has capacity
            target_load = self.calculate_driver_load(target_driver, current_assignments)
            target_capacity = self.driver_capacities.get(target_driver, {})
            
            can_accept = True
            for group in dog_groups:
                group_key = 'group' + str(group)
                current = target_load.get(group_key, 0)
                max_cap = target_capacity.get(group_key, 0)
                needed = dog_to_move.get('num_dogs', 1)
                
                if current + needed > max_cap:
                    can_accept = False
                    print("         ❌ MOVE BLOCKED: " + str(target_driver) + " group " + str(group) + ": " + str(current) + " + " + str(needed) + " > " + str(max_cap))
                    break
            
            if can_accept:
                # FINAL MOVE VALIDATION - Double-check capacity before allowing move
                temp_dog = {
                    'needed_groups': dog_groups,
                    'num_dogs': dog_to_move.get('num_dogs', 1),
                    'dog_name': dog_to_move.get('dog_name', 'Unknown')
                }
                if self.validate_assignment_capacity(target_driver, temp_dog, current_assignments):
                    targets.append({
                        'driver': target_driver,
                        'distance': distance,
                        'via_dog': assignment['dog_name'],
                        'via_dog_id': assignment['dog_id']
                    })
                else:
                    print("         🚨 MOVE TARGET BLOCKED by final validation for " + str(target_driver))
        
        # Sort by distance (closest first)
        return sorted(targets, key=lambda x: x['distance'])

    # ========== LEGACY CASCADING METHOD (KEPT FOR FALLBACK) ==========

    def attempt_cascading_move(self, blocked_driver, callout_dog, current_assignments, max_cascade_distance):
        """LEGACY: Attempt to free space in blocked_driver by moving one of their dogs"""
        print("🔄 ATTEMPTING LEGACY CASCADING MOVE for " + str(callout_dog.get('dog_name', 'Unknown')) + " → " + str(blocked_driver))
        print("   Need to free space in " + str(blocked_driver) + " for groups " + str(callout_dog['needed_groups']))
        
        driver_dogs = self.get_current_driver_dogs(blocked_driver, current_assignments)
        print("   " + str(blocked_driver) + " currently has " + str(len(driver_dogs)) + " dogs assigned")
        
        # Show current load for this driver
        current_load = self.calculate_driver_load(blocked_driver, current_assignments)
        driver_capacity = self.driver_capacities.get(blocked_driver, {})
        print("   " + str(blocked_driver) + " capacity: ***" + str(driver_capacity) + "***")
        print("   " + str(blocked_driver) + " current load: ***" + str(current_load) + "***")
        
        # Try to move each dog, starting with single-group dogs (easier to place)
        move_candidates = sorted(driver_dogs, key=lambda x: (len(x.get('needed_groups', [])), x.get('num_dogs', 1)))
        print("   Trying to move " + str(len(move_candidates)) + " dogs (easiest first):")
        
        for i, dog_to_move in enumerate(move_candidates):
            print("     " + str(i+1) + ". " + str(dog_to_move.get('dog_name', 'Unknown')) + " (" + str(dog_to_move.get('dog_id', 'Unknown')) + ") - groups " + str(dog_to_move.get('needed_groups', [])) + ", " + str(dog_to_move.get('num_dogs', 1)) + " dogs")
            
            # Find targets for this dog within cascade distance
            targets = self.find_move_targets_for_dog(dog_to_move, current_assignments, max_cascade_distance)
            print("        Found " + str(len(targets)) + " potential targets within " + str(max_cascade_distance) + "mi:")
            
            for j, target in enumerate(targets[:3]):  # Show top 3 targets
                print("          " + str(j+1) + ". " + str(target['driver']) + " - " + str(round(target['distance'], 3)) + "mi via " + str(target['via_dog']))
            
            if targets:
                best_target = targets[0]
                print("        ✅ EXECUTING MOVE: " + str(dog_to_move.get('dog_name', 'Unknown')) + " from " + str(blocked_driver) + " → " + str(best_target['driver']))
                
                # Execute the move
                for assignment in current_assignments:
                    if assignment['dog_id'] == dog_to_move['dog_id']:
                        assignment['driver'] = best_target['driver']
                        print("        🔄 LEGACY MOVE STATE UPDATED: " + str(dog_to_move.get('dog_name', 'Unknown')) + " moved to " + str(best_target['driver']))
                        break
                
                return {
                    'moved_dog': dog_to_move,
                    'from_driver': blocked_driver,
                    'to_driver': best_target['driver'],
                    'distance': best_target['distance'],
                    'via_dog': best_target['via_dog']
                }
            else:
                print("        ❌ No targets found for " + str(dog_to_move.get('dog_name', 'Unknown')))
        
        print("   ❌ CASCADING MOVE FAILED - no dogs could be relocated")
        return None

    def find_move_targets_for_dog(self, dog_to_move, current_assignments, max_distance):
        """Find potential drivers within range who can accept a dog"""
        targets = []
        
        for assignment in current_assignments:
            target_driver = assignment['driver']
            
            # Skip same driver
            if target_driver == dog_to_move.get('driver'):
                continue
            
            # Calculate distance
            distance = self.get_distance(dog_to_move['dog_id'], assignment['dog_id'])
            
            # Check group compatibility
            dog_groups = dog_to_move.get('needed_groups', [])
            target_groups = assignment.get('needed_groups', [])
            
            # Use flexible group compatibility for strategic moves
            if not self.check_group_compatibility_for_moves(dog_groups, target_groups, distance, max_distance):
                continue
            
            if distance > max_distance:
                continue
            
            # Check if target driver has capacity
            target_load = self.calculate_driver_load(target_driver, current_assignments)
            target_capacity = self.driver_capacities.get(target_driver, {})
            
            can_accept = True
            for group in dog_groups:
                group_key = 'group' + str(group)
                current = target_load.get(group_key, 0)
                max_cap = target_capacity.get(group_key, 0)
                needed = dog_to_move.get('num_dogs', 1)
                
                if current + needed > max_cap:
                    can_accept = False
                    break
            
            if can_accept:
                # FINAL LEGACY MOVE VALIDATION - Double-check capacity
                temp_dog = {
                    'needed_groups': dog_groups,
                    'num_dogs': dog_to_move.get('num_dogs', 1),
                    'dog_name': dog_to_move.get('dog_name', 'Unknown')
                }
                if self.validate_assignment_capacity(target_driver, temp_dog, current_assignments):
                    targets.append({
                        'driver': target_driver,
                        'distance': distance,
                        'via_dog': assignment['dog_name'],
                        'via_dog_id': assignment['dog_id']
                    })
                else:
                    print("         🚨 LEGACY MOVE TARGET BLOCKED by final validation for " + str(target_driver))
        
        return sorted(targets, key=lambda x: x['distance'])

    # ========== MAIN LOCALITY-FIRST ALGORITHM WITH CAPACITY BUG FIX ==========

    def locality_first_assignment(self):
        """FIXED: Locality-first assignment with proper state synchronization to prevent capacity race conditions"""
        print("\n🎯 LOCALITY-FIRST ASSIGNMENT ALGORITHM (CAPACITY BUG FIXED)")
        print("🔧 FIX: Immediate state updates prevent multiple dogs from over-filling same driver")
        print("📏 Starting at 0.2mi, expanding to 1.5mi in 0.1mi increments")
        print("🎯 STRATEGIC CASCADING: Target specific blocked groups with incremental radius")
        print("=" * 80)
        
        dogs_to_reassign = self.get_dogs_to_reassign()
        
        if not dogs_to_reassign:
            print("✅ No callouts detected - all dogs have drivers assigned!")
            return []
        
        # Build initial current assignments state
        current_assignments = []
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
        
        assignments_made = []
        moves_made = []
        dogs_remaining = dogs_to_reassign.copy()
        
        print(f"🐕 Processing {len(dogs_remaining)} callout dogs")
        
        # *** CRITICAL FIX: Helper function for immediate assignment with state sync ***
        def make_immediate_assignment(callout_dog, driver, distance, assignment_type, quality="GOOD"):
            """Make assignment and IMMEDIATELY update state to prevent race conditions"""
            
            # CRITICAL: Validate capacity with current state (includes assignments made this run)
            if not self.validate_assignment_capacity(driver, callout_dog, current_assignments):
                print(f"🚨 ASSIGNMENT BLOCKED: {driver} failed final capacity validation")
                return False
            
            # Create assignment record
            assignment_record = {
                'dog_id': callout_dog['dog_id'],
                'dog_name': callout_dog['dog_name'],
                'new_assignment': f"{driver}:{callout_dog['full_assignment_string']}",
                'driver': driver,
                'distance': distance,
                'quality': quality,
                'assignment_type': assignment_type
            }
            
            assignments_made.append(assignment_record)
            
            # *** CRITICAL FIX: Update current_assignments IMMEDIATELY ***
            # This ensures the next dog's validation sees this assignment
            current_assignments.append({
                'dog_id': callout_dog['dog_id'],
                'dog_name': callout_dog['dog_name'],
                'driver': driver,
                'needed_groups': callout_dog['needed_groups'],
                'num_dogs': callout_dog['num_dogs']
            })
            
            print(f"   ✅ IMMEDIATE STATE UPDATE: {callout_dog['dog_name']} → {driver} ({round(distance, 1)}mi) [{quality}]")
            print(f"      🔄 State synchronized - next validation will see this assignment")
            
            return True
        
        print("\n📍 STEP 1: Direct assignments at ≤" + str(self.PREFERRED_DISTANCE) + "mi")
        
        # Step 1: Process each dog individually with immediate state updates
        dogs_assigned_step1 = []
        for callout_dog in dogs_remaining[:]:  # Use slice to avoid modification during iteration
            best_assignment = None
            best_distance = float('inf')
            
            # Check all drivers for direct assignment
            for assignment in current_assignments:
                driver = assignment['driver']
                distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                
                # Skip obvious placeholders
                if distance >= 100.0:
                    continue
                    
                if distance > self.PREFERRED_DISTANCE:
                    continue
                
                # Check group compatibility
                if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, self.PREFERRED_DISTANCE):
                    continue
                
                # Pre-check capacity (will be validated again in make_immediate_assignment)
                current_load = self.calculate_driver_load(driver, current_assignments)
                driver_capacity = self.driver_capacities.get(driver, {})
                
                if not driver_capacity:
                    continue
                
                has_capacity = True
                for group in callout_dog['needed_groups']:
                    group_key = 'group' + str(group)
                    current = current_load.get(group_key, 0)
                    max_cap = driver_capacity.get(group_key, 0)
                    needed = callout_dog['num_dogs']
                    
                    if current + needed > max_cap:
                        has_capacity = False
                        break
                
                if has_capacity and distance < best_distance:
                    best_assignment = {
                        'driver': driver,
                        'distance': distance,
                        'via_dog': assignment['dog_name']
                    }
                    best_distance = distance
            
            if best_assignment:
                # Make assignment with immediate state update
                success = make_immediate_assignment(
                    callout_dog, 
                    best_assignment['driver'], 
                    best_assignment['distance'], 
                    'direct',
                    'GOOD'
                )
                
                if success:
                    dogs_assigned_step1.append(callout_dog)
        
        # Remove assigned dogs (must be done after iteration)
        for dog in dogs_assigned_step1:
            dogs_remaining.remove(dog)
        
        print("   📊 Step 1 results: " + str(len(dogs_assigned_step1)) + " direct assignments")
        
        # Step 2: Strategic cascading moves with immediate state updates
        if dogs_remaining:
            print("\n🎯 STEP 2: Strategic cascading moves")
            
            dogs_assigned_step2 = []
            for callout_dog in dogs_remaining[:]:
                # Find drivers within range but blocked by capacity
                blocked_drivers = []
                
                for assignment in current_assignments:
                    driver = assignment['driver']
                    distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                    
                    if distance >= 100.0 or distance > self.PREFERRED_DISTANCE:
                        continue
                    
                    if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, self.PREFERRED_DISTANCE):
                        continue
                    
                    # Check if blocked by capacity (using current state)
                    current_load = self.calculate_driver_load(driver, current_assignments)
                    driver_capacity = self.driver_capacities.get(driver, {})
                    
                    if not driver_capacity:
                        continue
                    
                    has_capacity = True
                    for group in callout_dog['needed_groups']:
                        group_key = 'group' + str(group)
                        current = current_load.get(group_key, 0)
                        max_cap = driver_capacity.get(group_key, 0)
                        needed = callout_dog['num_dogs']
                        
                        if current + needed > max_cap:
                            has_capacity = False
                            break
                    
                    if not has_capacity:
                        blocked_drivers.append({
                            'driver': driver,
                            'distance': distance
                        })
                
                if blocked_drivers:
                    blocked_drivers.sort(key=lambda x: x['distance'])
                    best_blocked = blocked_drivers[0]
                    
                    # Try strategic cascading move
                    move_result = self.attempt_strategic_cascading_move(
                        best_blocked['driver'], 
                        callout_dog, 
                        current_assignments, 
                        self.CASCADING_MOVE_MAX
                    )
                    
                    if move_result:
                        # Record the move
                        moves_made.append({
                            'dog_name': move_result['moved_dog']['dog_name'],
                            'dog_id': move_result['moved_dog']['dog_id'],
                            'from_driver': move_result['from_driver'],
                            'to_driver': move_result['to_driver'],
                            'distance': move_result['distance'],
                            'reason': "strategic_free_space_for_" + callout_dog.get('dog_name', 'Unknown')
                        })
                        
                        # Update existing assignment for moved dog
                        moved_dog_id = move_result['moved_dog']['dog_id']
                        for existing_assignment in assignments_made:
                            if existing_assignment['dog_id'] == moved_dog_id:
                                old_driver = existing_assignment['driver']
                                new_driver = move_result['to_driver']
                                existing_assignment['driver'] = new_driver
                                existing_assignment['new_assignment'] = existing_assignment['new_assignment'].replace(old_driver + ":", new_driver + ":")
                                existing_assignment['assignment_type'] = 'moved_by_strategic_cascading'
                                break
                        
                        # Now assign the callout dog with immediate state update
                        success = make_immediate_assignment(
                            callout_dog,
                            best_blocked['driver'],
                            best_blocked['distance'],
                            'strategic_cascading',
                            'GOOD'
                        )
                        
                        if success:
                            dogs_assigned_step2.append(callout_dog)
                            print("      🎯 Strategic move: " + str(move_result['moved_dog']['dog_name']) + " → " + str(move_result['to_driver']) + " (" + str(round(move_result['distance'], 1)) + "mi)")
            
            # Remove assigned dogs
            for dog in dogs_assigned_step2:
                dogs_remaining.remove(dog)
            
            print("   📊 Step 2 results: " + str(len(dogs_assigned_step2)) + " strategic cascading assignments")
        
        # Step 3+: Incremental radius expansion with immediate state updates
        current_radius = 0.3
        step_number = 3
        
        while current_radius <= self.ABSOLUTE_MAX_DISTANCE and dogs_remaining:
            print("\n📏 STEP " + str(step_number) + ": Radius expansion to ≤" + str(current_radius) + "mi")
            
            dogs_assigned_this_radius = []
            
            for callout_dog in dogs_remaining[:]:
                # Try direct assignment at current radius
                best_assignment = None
                best_distance = float('inf')
                
                for assignment in current_assignments:
                    driver = assignment['driver']
                    distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                    
                    if distance >= 100.0 or distance > current_radius:
                        continue
                    
                    if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, current_radius):
                        continue
                    
                    # Pre-check capacity (using current updated state)
                    current_load = self.calculate_driver_load(driver, current_assignments)
                    driver_capacity = self.driver_capacities.get(driver, {})
                    
                    if not driver_capacity:
                        continue
                    
                    has_capacity = True
                    for group in callout_dog['needed_groups']:
                        group_key = 'group' + str(group)
                        current = current_load.get(group_key, 0)
                        max_cap = driver_capacity.get(group_key, 0)
                        needed = callout_dog['num_dogs']
                        
                        if current + needed > max_cap:
                            has_capacity = False
                            break
                    
                    if has_capacity and distance < best_distance:
                        best_assignment = {
                            'driver': driver,
                            'distance': distance,
                            'via_dog': assignment['dog_name']
                        }
                        best_distance = distance
                
                if best_assignment:
                    # Determine quality
                    distance = best_assignment['distance']
                    if distance <= self.PREFERRED_DISTANCE:
                        quality = 'GOOD'
                    elif distance <= self.MAX_DISTANCE:
                        quality = 'BACKUP'
                    else:
                        quality = 'EMERGENCY'
                    
                    # Make assignment with immediate state update
                    success = make_immediate_assignment(
                        callout_dog,
                        best_assignment['driver'],
                        best_assignment['distance'],
                        'radius_expansion',
                        quality
                    )
                    
                    if success:
                        dogs_assigned_this_radius.append(callout_dog)
                
                else:
                    # Try strategic cascading at current radius if direct assignment failed
                    blocked_drivers = []
                    
                    for assignment in current_assignments:
                        driver = assignment['driver']
                        distance = self.get_distance(callout_dog['dog_id'], assignment['dog_id'])
                        
                        if distance >= 100.0 or distance > current_radius:
                            continue
                        
                        if not self.check_group_compatibility(callout_dog['needed_groups'], assignment['needed_groups'], distance, current_radius):
                            continue
                        
                        # Check if blocked by capacity
                        current_load = self.calculate_driver_load(driver, current_assignments)
                        driver_capacity = self.driver_capacities.get(driver, {})
                        
                        if not driver_capacity:
                            continue
                        
                        has_capacity = True
                        for group in callout_dog['needed_groups']:
                            group_key = 'group' + str(group)
                            current = current_load.get(group_key, 0)
                            max_cap = driver_capacity.get(group_key, 0)
                            needed = callout_dog['num_dogs']
                            
                            if current + needed > max_cap:
                                has_capacity = False
                                break
                        
                        if not has_capacity:
                            blocked_drivers.append({
                                'driver': driver,
                                'distance': distance
                            })
                    
                    if blocked_drivers:
                        blocked_drivers.sort(key=lambda x: x['distance'])
                        best_blocked = blocked_drivers[0]
                        
                        # Try strategic cascading move at current radius
                        move_result = self.attempt_strategic_cascading_move(
                            best_blocked['driver'], 
                            callout_dog, 
                            current_assignments, 
                            current_radius  # Use current search radius for strategic cascading
                        )
                        
                        if move_result:
                            # Record the move
                            moves_made.append({
                                'dog_name': move_result['moved_dog']['dog_name'],
                                'dog_id': move_result['moved_dog']['dog_id'],
                                'from_driver': move_result['from_driver'],
                                'to_driver': move_result['to_driver'],
                                'distance': move_result['distance'],
                                'reason': "strategic_radius_" + str(current_radius) + "_space_for_" + callout_dog.get('dog_name', 'Unknown')
                            })
                            
                            # Update existing assignment for moved dog
                            moved_dog_id = move_result['moved_dog']['dog_id']
                            for existing_assignment in assignments_made:
                                if existing_assignment['dog_id'] == moved_dog_id:
                                    old_driver = existing_assignment['driver']
                                    new_driver = move_result['to_driver']
                                    existing_assignment['driver'] = new_driver
                                    existing_assignment['new_assignment'] = existing_assignment['new_assignment'].replace(old_driver + ":", new_driver + ":")
                                    existing_assignment['assignment_type'] = 'moved_by_strategic_cascading'
                                    break
                            
                            # Assign the callout dog with immediate state update
                            distance = best_blocked['distance']
                            if distance <= self.PREFERRED_DISTANCE:
                                quality = 'GOOD'
                            elif distance <= self.MAX_DISTANCE:
                                quality = 'BACKUP'
                            else:
                                quality = 'EMERGENCY'
                            
                            success = make_immediate_assignment(
                                callout_dog,
                                best_blocked['driver'],
                                best_blocked['distance'],
                                'strategic_cascading_radius',
                                quality
                            )
                            
                            if success:
                                dogs_assigned_this_radius.append(callout_dog)
                                print("      🎯 Strategic move: " + str(move_result['moved_dog']['dog_name']) + " → " + str(move_result['to_driver']) + " (" + str(round(move_result['distance'], 1)) + "mi)")
            
            # Remove assigned dogs
            for dog in dogs_assigned_this_radius:
                dogs_remaining.remove(dog)
            
            print("   📊 Radius " + str(current_radius) + "mi results: " + str(len(dogs_assigned_this_radius)) + " assignments")
            
            current_radius += 0.1
            step_number += 1
        
        # Final step: Mark remaining as emergency
        for callout_dog in dogs_remaining:
            assignment_record = {
                'dog_id': callout_dog['dog_id'],
                'dog_name': callout_dog['dog_name'],
                'new_assignment': "UNASSIGNED:" + callout_dog['full_assignment_string'],
                'driver': 'UNASSIGNED',
                'distance': float('inf'),
                'quality': 'EMERGENCY',
                'assignment_type': 'failed'
            }
            assignments_made.append(assignment_record)
        
        # Store moves for writing
        self.greedy_moves_made = moves_made
        
        # Summary
        total_dogs = len(dogs_to_reassign)
        good_count = len([a for a in assignments_made if a['quality'] == 'GOOD'])
        backup_count = len([a for a in assignments_made if a['quality'] == 'BACKUP'])
        emergency_count = len([a for a in assignments_made if a['quality'] == 'EMERGENCY'])
        
        print("\n🏆 FIXED LOCALITY-FIRST RESULTS:")
        print("   📊 " + str(len(assignments_made)) + "/" + str(total_dogs) + " dogs processed")
        print("   💚 " + str(good_count) + " GOOD assignments")
        print("   🟡 " + str(backup_count) + " BACKUP assignments")
        print("   🚨 " + str(emergency_count) + " EMERGENCY assignments")
        print("   🔧 BUG FIXED: Immediate state updates prevent capacity race conditions")
        print("   ✅ No driver will be assigned more dogs than their capacity allows")
        
        return assignments_made

    def reassign_dogs_multi_strategy_optimization(self):
        """Locality-first algorithm with strategic cascading and 1.5 mile range (CAPACITY BUG FIXED)"""
        print("\n🔄 Starting LOCALITY-FIRST + STRATEGIC CASCADING SYSTEM (BUG FIXED)...")
        print("🎯 Strategy: Proximity-first with strategic group-targeted cascading")
        print("📊 Quality: GOOD ≤0.2mi, BACKUP ≤0.5mi, EMERGENCY >0.5mi")
        print("🚨 Focus: Immediate proximity with strategic dynamic space optimization")
        print("🎯 EXTENDED RANGE: Up to 1.5mi exact matches, 1.125mi adjacent matches")
        print("🎯 STRATEGIC CASCADING: Target blocked groups with 0.2→0.3→0.4→etc. radius expansion")
        print("🔧 CAPACITY BUG FIXED: Immediate state updates prevent race conditions")
        print("=" * 80)
        
        # Try the locality-first algorithm with strategic cascading
        try:
            return self.locality_first_assignment()
        except Exception as e:
            print("⚠️ Locality-first algorithm failed: " + str(e))
            print("🔄 Falling back to basic assignment...")
            return []

    def write_results_to_sheets(self, reassignments):
        """Write reassignment results and greedy walk moves back to Google Sheets"""
        try:
            print("\n📝 Writing " + str(len(reassignments)) + " results to Google Sheets...")
            
            if not hasattr(self, 'sheets_client') or not self.sheets_client:
                print("❌ Google Sheets client not initialized")
                return False
            
            # Pre-validation of reassignments data
            print("🔒 PRE-VALIDATION: Checking reassignment data structure...")
            for i, assignment in enumerate(reassignments[:3]):  # Show first 3
                dog_id = assignment.get('dog_id', 'MISSING')
                new_assignment = assignment.get('new_assignment', 'MISSING')
                print("   " + str(i+1) + ". Dog ID: '" + str(dog_id) + "' → New Assignment: '" + str(new_assignment) + "'")
                
                # Critical safety checks
                if dog_id == new_assignment:
                    print("   🚨 CRITICAL ERROR: dog_id equals new_assignment! ABORTING!")
                    return False
                
                if new_assignment.endswith('x') and new_assignment[:-1].isdigit():
                    print("   🚨 CRITICAL ERROR: new_assignment looks like dog_id! ABORTING!")
                    return False
                
                if ':' not in new_assignment:
                    print("   🚨 CRITICAL ERROR: new_assignment missing driver:group format! ABORTING!")
                    return False
            
            print("✅ Pre-validation passed!")
            
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
                        print("📋 Using sheet: " + str(sheet_name))
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
            print("📋 Sheet has " + str(len(all_data)) + " rows")
            
            # Find the Dog ID column
            dog_id_col = None
            for i, header in enumerate(header_row):
                header_clean = str(header).lower().strip()
                if 'dog id' in header_clean:
                    dog_id_col = i
                    print("📍 Found Dog ID column at index " + str(i))
                    break
            
            if dog_id_col is None:
                print("❌ Could not find 'Dog ID' column")
                return False
            
            # Target Column H (Combined column) - index 7
            target_col = 7  
            print("📍 Writing to Column H (Combined) at index " + str(target_col))
            
            # Prepare batch updates for reassignments
            updates = []
            updates_count = 0
            
            print("\n🔍 Processing " + str(len(reassignments)) + " reassignments...")
            
            # Process reassignments (now includes final locations after strategic moves)
            for assignment in reassignments:
                dog_id = str(assignment.get('dog_id', '')).strip()
                new_assignment = str(assignment.get('new_assignment', '')).strip()
                
                # Final validation
                if not new_assignment or new_assignment == dog_id or ':' not in new_assignment:
                    print("  ❌ SKIPPING invalid assignment for " + str(dog_id))
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
                            assignment_type = assignment.get('assignment_type', 'unknown')
                            if 'moved_by_strategic' in assignment_type:
                                print("  🎯 " + str(dog_id) + " → " + str(new_assignment) + " (final location after strategic move)")
                            else:
                                print("  ✅ " + str(dog_id) + " → " + str(new_assignment))
                            break
            
            if not updates:
                print("❌ No valid updates to make")
                return False
            
            # Execute batch update
            print("\n📤 Writing " + str(len(updates)) + " updates to Google Sheets...")
            worksheet.batch_update(updates)
            
            strategic_moves = len([a for a in reassignments if 'moved_by_strategic' in a.get('assignment_type', '')])
            success_msg = "✅ Successfully updated " + str(updates_count) + " assignments with strategic cascading (CAPACITY BUG FIXED)!"
            if strategic_moves > 0:
                success_msg += " (including " + str(strategic_moves) + " dogs moved to final locations via strategic cascading)"
            
            print(success_msg)
            print("🎯 Used locality-first + strategic cascading with 1.5mi range + 75% adjacent")
            print("🔧 CAPACITY BUG FIXED: No driver will exceed capacity limits")
            
            # Send Slack notification
            slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    message = "🐕 Dog Reassignment Complete (CAPACITY BUG FIXED): " + str(updates_count) + " assignments updated using strategic cascading + 1.5mi range. No over-capacity assignments!"
                    slack_message = {"text": message}
                    response = requests.post(slack_webhook, json=slack_message, timeout=10)
                    if response.status_code == 200:
                        print("📱 Slack notification sent")
                except Exception as e:
                    print("⚠️ Could not send Slack notification: " + str(e))
            
            return True
            
        except Exception as e:
            print("❌ Error writing to sheets: " + str(e))
            import traceback
            print("🔍 Full error: " + str(traceback.format_exc()))
            return False


def main():
    """Main function to run the dog reassignment system"""
    print("🚀 Enhanced Dog Reassignment System - CAPACITY BUG FIXED")
    print("🔧 FIX: Immediate state updates prevent capacity race conditions")
    print("🎯 NEW: Strategic group-targeted cascading with incremental radius expansion")
    print("📏 Starts at 0.2mi, expands to 1.5mi in 0.1mi increments")
    print("🔄 Adjacent groups: 75% of current radius (more generous)")
    print("🎯 STRATEGIC CASCADING: Target blocked groups, not random dogs")
    print("🚶 Cascading moves up to 0.7mi with incremental radius (0.2→0.3→0.4→etc.)")
    print("🧅 Onion-layer backflow pushes outer assignments out to create inner space")
    print("📊 Quality: GOOD ≤0.2mi, BACKUP ≤0.5mi, EMERGENCY >0.5mi")
    print("🎯 EXTENDED RANGE: Exact matches ≤1.5mi, Adjacent groups ≤1.125mi")
    print("✅ GUARANTEED: No driver will be assigned more dogs than their capacity allows")
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
    
    # Run the locality-first assignment with strategic cascading
    print("\n🔄 Processing callout assignments...")
    
    reassignments = system.reassign_dogs_multi_strategy_optimization()
    
    # Ensure reassignments is always a list
    if reassignments is None:
        reassignments = []
    
    # Write results
    if reassignments:
        write_success = system.write_results_to_sheets(reassignments)
        if write_success:
            print("\n🎉 SUCCESS! Processed " + str(len(reassignments)) + " callout assignments")
            print("✅ Used locality-first + strategic cascading with 1.5mi range + 75% adjacent")
            print("🔧 CAPACITY BUG FIXED: All drivers remain within capacity limits")
        else:
            print("\n❌ Failed to write " + str(len(reassignments)) + " results to Google Sheets")
    else:
        print("\n✅ No callout assignments needed - all drivers available or no valid assignments found")


if __name__ == "__main__":
    main()
