#!/usr/bin/env python3
"""
🚀 Advanced Multi-Phase Dog Logistics Optimization System
==================================================
Handles complete driver day workflows with multiple field visits, 
strategic dog management, and complex routing optimization.
"""

import os
import json
import math
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from itertools import permutations
import gspread
from google.oauth2.service_account import Credentials

@dataclass
class LocationInfo:
    """Information about a location (dog, parking, field)"""
    dog_id: str
    location_type: str  # 'dog', 'parking', 'field'
    driver_name: str    # Which driver this belongs to
    groups: List[int]   # Which groups this location serves

@dataclass
class DogAssignment:
    """Complete dog assignment with group and driver info"""
    dog_id: str
    dog_name: str
    driver_name: str
    groups: List[int]
    num_dogs: int
    is_callout: bool = False

@dataclass
class DriverPhase:
    """Single phase of a driver's day (Group 1, 2, or 3)"""
    phase_num: int
    pickup_dogs: List[str]      # Dogs to pick up this phase
    dropoff_dogs: List[str]     # Dogs to drop off this phase
    retain_dogs: List[str]      # Dogs that stay at field for next phase
    field_location: str         # Field ID for this phase

@dataclass
class DriverDayPlan:
    """Complete day plan for a driver"""
    driver_name: str
    parking_location: str
    field_location: str
    phases: List[DriverPhase]
    total_dogs: int
    total_distance: float = 0.0

class AdvancedDogLogisticsSystem:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        self.distance_matrix = {}
        self.dog_assignments = {}
        self.driver_capacities = {}
        self.location_info = {}  # dog_id -> LocationInfo
        self.callout_dogs = []
        self.unavailable_drivers = set()  # Drivers who called out
        
        if not test_mode:
            self.setup_google_sheets()
            
    def setup_google_sheets(self):
        """Initialize Google Sheets connection"""
        try:
            service_account_info = json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'])
            credentials = Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.gc = gspread.authorize(credentials)
            # Get spreadsheet ID from environment or use your actual spreadsheet
            spreadsheet_id = os.getenv('GOOGLE_SHEETS_ID', '1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0')
            self.sheet = self.gc.open_by_key(spreadsheet_id)
            print("✅ Google Sheets client setup successful")
        except Exception as e:
            print(f"❌ Google Sheets setup failed: {e}")
            raise

    def load_all_data(self):
        """Load and parse all data including location types"""
        print("\n⬇️ Loading data from Google Sheets...")
        
        # Load distance matrix from distance spreadsheet
        print("📊 Loading distance matrix...")
        matrix_sheet = self.distance_sheet.worksheet("Distance Matrix")
        matrix_data = matrix_sheet.get_all_values()
        
        if not matrix_data:
            raise ValueError("❌ Distance matrix is empty")
        
        dog_ids = [cell.strip() for cell in matrix_data[0][1:] if cell.strip()]
        print(f"📊 Distance matrix shape: ({len(matrix_data)}, {len(matrix_data[0])})")
        print(f"📊 Found {len(dog_ids)} column Dog IDs")
        
        # Build distance matrix
        for i, row in enumerate(matrix_data[1:], 1):
            if i >= len(matrix_data):
                break
                
            row_dog_id = row[0].strip()
            if not row_dog_id:
                continue
                
            self.distance_matrix[row_dog_id] = {}
            for j, distance_str in enumerate(row[1:len(dog_ids)+1], 0):
                if j < len(dog_ids):
                    col_dog_id = dog_ids[j]
                    try:
                        distance = float(distance_str) if distance_str.strip() else 100.0
                    except (ValueError, AttributeError):
                        distance = 100.0
                    self.distance_matrix[row_dog_id][col_dog_id] = distance
        
        print(f"✅ Loaded distance matrix for {len(self.distance_matrix)} dogs")
        
        # Load dog assignments with location parsing from map spreadsheet
        print("🐕 Loading dog assignments...")
        map_sheet = self.map_sheet.worksheet("Map")
        map_data = map_sheet.get_all_values()
        
        print(f"📊 Map sheet shape: ({len(map_data)}, {len(map_data[0])})")
        
        headers = [h.strip().lower() for h in map_data[0]]
        dog_id_col = headers.index('dog id') if 'dog id' in headers else 0
        dog_name_col = headers.index('dog name') if 'dog name' in headers else 1  
        assignment_col = headers.index('assignment') if 'assignment' in headers else 2
        callout_col = headers.index('callout') if 'callout' in headers else None
        
        regular_assignments = 0
        callout_drivers = set()  # Track which drivers called out
        
        for row in map_data[1:]:
            if len(row) <= max(dog_id_col, dog_name_col, assignment_col):
                continue
                
            dog_id = row[dog_id_col].strip()
            dog_name = row[dog_name_col].strip() 
            assignment = row[assignment_col].strip()
            callout_info = row[callout_col].strip() if callout_col is not None and callout_col < len(row) else ""
            
            if not dog_id:
                continue
            
            # Extract callout driver from callout column
            if callout_info:
                # Parse "Aidan callout" or "Sara callout" etc.
                callout_parts = callout_info.lower().split()
                for part in callout_parts:
                    if part not in ['callout', 'call', 'out']:
                        callout_drivers.add(part.strip().title())  # Add driver name
                
            # Parse location type from dog name
            location_type = 'dog'  # default
            if 'parking' in dog_name.lower():
                location_type = 'parking'
            elif 'field' in dog_name.lower():
                location_type = 'field'
            
            # Parse driver and groups from assignment  
            driver_name, groups, num_dogs = self.parse_assignment(assignment)
            
            # Store location info
            self.location_info[dog_id] = LocationInfo(
                dog_id=dog_id,
                location_type=location_type,
                driver_name=driver_name if driver_name else "",
                groups=groups
            )
            
            # Store assignment info
            self.dog_assignments[dog_id] = DogAssignment(
                dog_id=dog_id,
                dog_name=dog_name,
                driver_name=driver_name if driver_name else "",
                groups=groups,
                num_dogs=num_dogs,
                is_callout=(not assignment or not driver_name)  # Empty assignment = callout
            )
            
            if driver_name:
                regular_assignments += 1
                
        print(f"✅ Loaded {regular_assignments} regular assignments")
        
        # Store unavailable drivers
        self.unavailable_drivers = callout_drivers
        if callout_drivers:
            print(f"🚫 Unavailable drivers: {', '.join(callout_drivers)}")
        
        # Load driver capacities from map spreadsheet
        print("👥 Loading driver capacities from map sheet columns R:W...")
        try:
            capacity_data = map_sheet.batch_get(['R1:W50'])[0]
            capacity_headers = [h.strip() for h in capacity_data[0] if h.strip()]
            
            for row in capacity_data[1:]:
                if len(row) > 0 and row[0].strip():
                    driver_name = row[0].strip()
                    try:
                        capacity = int(row[1]) if len(row) > 1 and row[1].strip() else 9
                        self.driver_capacities[driver_name] = capacity
                    except (ValueError, IndexError):
                        self.driver_capacities[driver_name] = 9
                        
            print(f"✅ Loaded capacities for {len(self.driver_capacities)} drivers")
        except Exception as e:
            print(f"⚠️ Could not load driver capacities: {e}")
            
    def parse_assignment(self, assignment: str) -> Tuple[str, List[int], int]:
        """Parse assignment string to extract driver, groups, and dog count"""
        if not assignment or assignment.strip() == "":
            return None, [], 1  # Empty assignment = callout
            
        assignment = assignment.strip()
        if assignment.lower() in ['callout', 'call out']:
            return None, [], 1
            
        # Handle formats like "Michelle:123", "Sara:2&3", "Court:1DD1"
        if ':' in assignment:
            driver_part, group_part = assignment.split(':', 1)
            driver_name = driver_part.strip()
            
            # Parse groups from group_part
            groups = []
            group_part = group_part.strip()
            
            # Handle special cases like "1DD1", "2DD1" 
            if 'DD' in group_part:
                # Extract base group (the first digit)
                base_group = int(group_part[0])
                groups = [base_group]
            elif '&' in group_part:
                # Handle "1&2", "2&3"
                for g in group_part.split('&'):
                    try:
                        groups.append(int(g.strip()))
                    except ValueError:
                        pass
            elif group_part == '123':
                groups = [1, 2, 3]
            else:
                # Single group or other format
                for char in group_part:
                    try:
                        if char.isdigit():
                            groups.append(int(char))
                    except ValueError:
                        pass
            
            # Remove duplicates and sort
            groups = sorted(list(set(groups)))
            
            # Estimate number of dogs (default to 1, could be enhanced)
            num_dogs = 1
            if 'DD' in group_part:
                num_dogs = 2  # DD usually means 2 dogs
                
            return driver_name, groups, num_dogs
            
        return None, [], 1

    def identify_callouts(self):
        """Identify dogs that need reassignment due to callouts"""
        self.callout_dogs = []
        
        for dog_id, assignment in self.dog_assignments.items():
            if assignment.is_callout:
                self.callout_dogs.append(assignment)
                
        print(f"🚨 Found {len(self.callout_dogs)} callouts that need drivers assigned:")
        for dog in self.callout_dogs:
            groups_str = '&'.join(map(str, dog.groups)) if len(dog.groups) > 1 else str(dog.groups[0]) if dog.groups else 'unknown'
            print(f"   - {dog.dog_name} ({dog.dog_id}) - {dog.num_dogs} dogs, groups [{groups_str}]")
            
        if self.unavailable_drivers:
            print(f"🚫 Excluding unavailable drivers: {', '.join(self.unavailable_drivers)}")

    def build_driver_day_plans(self) -> Dict[str, DriverDayPlan]:
        """Build complete day plans for each driver"""
        print("\n🗓️ Building driver day plans...")
        
        driver_plans = {}
        
        # Group assignments by driver
        driver_assignments = {}
        driver_parking = {}
        driver_fields = {}
        
        for dog_id, assignment in self.dog_assignments.items():
            if assignment.is_callout or not assignment.driver_name:
                continue
                
            location = self.location_info.get(dog_id)
            if not location:
                continue
                
            driver = assignment.driver_name
            
            if location.location_type == 'parking':
                driver_parking[driver] = dog_id
            elif location.location_type == 'field': 
                driver_fields[driver] = dog_id
            elif location.location_type == 'dog':
                if driver not in driver_assignments:
                    driver_assignments[driver] = []
                driver_assignments[driver].append(assignment)
        
        # Build day plan for each driver (excluding unavailable drivers)
        for driver_name, assignments in driver_assignments.items():
            # Skip drivers who called out
            if driver_name in self.unavailable_drivers:
                print(f"🚫 Skipping unavailable driver: {driver_name}")
                continue
                
            parking = driver_parking.get(driver_name, '')
            field = driver_fields.get(driver_name, '')
            
            # Group assignments by phase
            phase_1_dogs = [a for a in assignments if 1 in a.groups]
            phase_2_dogs = [a for a in assignments if 2 in a.groups] 
            phase_3_dogs = [a for a in assignments if 3 in a.groups]
            
            phases = []
            
            # Phase 1: Pick up all dogs with group 1
            if phase_1_dogs:
                pickup_dogs = [a.dog_id for a in phase_1_dogs]
                dropoff_dogs = [a.dog_id for a in phase_1_dogs if a.groups == [1]]  # Only group 1
                retain_dogs = [a.dog_id for a in phase_1_dogs if len(a.groups) > 1]  # Multi-group dogs
                
                phases.append(DriverPhase(
                    phase_num=1,
                    pickup_dogs=pickup_dogs,
                    dropoff_dogs=dropoff_dogs, 
                    retain_dogs=retain_dogs,
                    field_location=field
                ))
            
            # Phase 2: Pick up group 2 dogs
            if phase_2_dogs:
                pickup_dogs = [a.dog_id for a in phase_2_dogs if 2 in a.groups and 1 not in a.groups]  # New pickups
                dropoff_dogs = [a.dog_id for a in assignments if a.groups == [2] or a.groups == [1,2]]  # Group 2 and 1&2
                retain_dogs = [a.dog_id for a in assignments if 3 in a.groups and a.dog_id not in dropoff_dogs]  # Stay for group 3
                
                phases.append(DriverPhase(
                    phase_num=2,
                    pickup_dogs=pickup_dogs,
                    dropoff_dogs=dropoff_dogs,
                    retain_dogs=retain_dogs, 
                    field_location=field
                ))
            
            # Phase 3: Pick up group 3 dogs and finish
            if phase_3_dogs:
                pickup_dogs = [a.dog_id for a in phase_3_dogs if 3 in a.groups and 1 not in a.groups and 2 not in a.groups]  # New pickups
                dropoff_dogs = [a.dog_id for a in assignments if 3 in a.groups]  # All remaining dogs
                
                phases.append(DriverPhase(
                    phase_num=3,
                    pickup_dogs=pickup_dogs,
                    dropoff_dogs=dropoff_dogs,
                    retain_dogs=[],  # Day is over
                    field_location=field
                ))
            
            total_dogs = sum(a.num_dogs for a in assignments)
            
            driver_plans[driver_name] = DriverDayPlan(
                driver_name=driver_name,
                parking_location=parking,
                field_location=field,
                phases=phases,
                total_dogs=total_dogs
            )
            
        print(f"✅ Built day plans for {len(driver_plans)} drivers")
        
        # Debug output
        for driver_name, plan in driver_plans.items():
            print(f"\n👤 {driver_name} ({plan.total_dogs} dogs):")
            print(f"   🅿️ Parking: {plan.parking_location}")
            print(f"   🏈 Field: {plan.field_location}")
            for phase in plan.phases:
                print(f"   📋 Phase {phase.phase_num}: pickup {len(phase.pickup_dogs)}, dropoff {len(phase.dropoff_dogs)}, retain {len(phase.retain_dogs)}")
                
        return driver_plans

    def calculate_driver_day_distance(self, plan: DriverDayPlan) -> float:
        """Calculate total distance for a driver's complete day"""
        total_distance = 0.0
        current_location = plan.parking_location
        
        print(f"\n🚗 Calculating route for {plan.driver_name}:")
        print(f"   Starting at parking: {current_location}")
        
        for phase in plan.phases:
            print(f"\n   📋 Phase {phase.phase_num}:")
            
            # Pick up dogs
            if phase.pickup_dogs:
                print(f"      🐕 Picking up {len(phase.pickup_dogs)} dogs...")
                pickup_route = self.optimize_pickup_route(current_location, phase.pickup_dogs)
                pickup_distance = self.calculate_route_distance(pickup_route)
                total_distance += pickup_distance
                current_location = pickup_route[-1] if pickup_route else current_location
                print(f"      📏 Pickup distance: {pickup_distance:.1f}mi")
            
            # Go to field
            if phase.field_location:
                field_distance = self.get_distance(current_location, phase.field_location)
                total_distance += field_distance
                current_location = phase.field_location
                print(f"      🏈 To field: {field_distance:.1f}mi")
            
            # Drop off dogs (they get picked up from field)
            if phase.dropoff_dogs:
                print(f"      🏠 Dropping off {len(phase.dropoff_dogs)} dogs...")
                dropoff_route = self.optimize_dropoff_route(phase.field_location, phase.dropoff_dogs)
                dropoff_distance = self.calculate_route_distance(dropoff_route)
                total_distance += dropoff_distance
                current_location = dropoff_route[-1] if dropoff_route else current_location
                print(f"      📏 Dropoff distance: {dropoff_distance:.1f}mi")
                
                # Return to field if there are more phases
                if phase.retain_dogs:
                    return_distance = self.get_distance(current_location, phase.field_location)
                    total_distance += return_distance
                    current_location = phase.field_location
                    print(f"      🔄 Return to field: {return_distance:.1f}mi")
        
        # Return to parking at end of day
        if current_location != plan.parking_location:
            final_distance = self.get_distance(current_location, plan.parking_location)
            total_distance += final_distance
            print(f"   🅿️ Return to parking: {final_distance:.1f}mi")
        
        print(f"   🎯 Total day distance: {total_distance:.1f}mi")
        return total_distance

    def optimize_pickup_route(self, start_location: str, dog_ids: List[str]) -> List[str]:
        """Optimize the route for picking up multiple dogs"""
        if not dog_ids:
            return [start_location]
            
        # Simple nearest neighbor for now
        route = [start_location]
        remaining_dogs = dog_ids.copy()
        current_location = start_location
        
        while remaining_dogs:
            closest_dog = min(remaining_dogs, key=lambda dog: self.get_distance(current_location, dog))
            route.append(closest_dog)
            remaining_dogs.remove(closest_dog)
            current_location = closest_dog
            
        return route

    def optimize_dropoff_route(self, start_location: str, dog_ids: List[str]) -> List[str]:
        """Optimize the route for dropping off multiple dogs"""
        return self.optimize_pickup_route(start_location, dog_ids)

    def calculate_route_distance(self, route: List[str]) -> float:
        """Calculate total distance for a route"""
        if len(route) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(len(route) - 1):
            distance = self.get_distance(route[i], route[i + 1])
            total_distance += distance
            
        return total_distance

    def get_distance(self, from_id: str, to_id: str) -> float:
        """Get distance between two locations"""
        if from_id == to_id:
            return 0.0
            
        if from_id in self.distance_matrix and to_id in self.distance_matrix[from_id]:
            return self.distance_matrix[from_id][to_id]
        elif to_id in self.distance_matrix and from_id in self.distance_matrix[to_id]:
            return self.distance_matrix[to_id][from_id]
        else:
            return 100.0  # Default large distance

    def calculate_total_system_distance(self, driver_plans: Dict[str, DriverDayPlan]) -> float:
        """Calculate total distance across all drivers"""
        total_distance = 0.0
        
        print(f"\n📊 COMPLETE SYSTEM DISTANCE ANALYSIS:")
        for driver_name, plan in driver_plans.items():
            day_distance = self.calculate_driver_day_distance(plan)
            plan.total_distance = day_distance
            total_distance += day_distance
            
        print(f"\n🎯 TOTAL SYSTEM DISTANCE: {total_distance:.1f} miles")
        return total_distance

    def reassign_callout_dogs_advanced(self):
        """Advanced callout reassignment with full day workflow optimization"""
        if not self.callout_dogs:
            print("✅ No callout assignments needed - all drivers available")
            return
            
        print(f"\n🔄 Starting ADVANCED LOGISTICS OPTIMIZATION...")
        print(f"🎯 Optimizing complete driver day workflows with callout reassignments")
        
        # Get baseline driver plans (without callouts)
        baseline_plans = self.build_driver_day_plans()
        baseline_distance = self.calculate_total_system_distance(baseline_plans)
        
        print(f"\n📊 BASELINE SYSTEM (without callouts):")
        print(f"   📏 Total distance: {baseline_distance:.1f} miles")
        print(f"   👥 Active drivers: {len(baseline_plans)}")
        if self.unavailable_drivers:
            print(f"   🚫 Excluded drivers: {', '.join(self.unavailable_drivers)}")
        
        # Try different reassignment strategies
        strategies = [
            ("Advanced Greedy Walk", self.strategy_nearest_driver),
            ("Load Balanced with Domino", self.strategy_load_balanced),
            ("Minimal Distance Impact", self.strategy_minimal_impact)
        ]
        
        best_strategy = None
        best_distance = float('inf')
        best_plans = None
        
        for strategy_name, strategy_func in strategies:
            print(f"\n🔹 TESTING STRATEGY: {strategy_name}")
            
            try:
                # Create modified plans with callout assignments
                modified_plans = strategy_func(baseline_plans.copy())
                
                if modified_plans:
                    total_distance = self.calculate_total_system_distance(modified_plans)
                    distance_increase = total_distance - baseline_distance
                    
                    print(f"   📏 Total distance: {total_distance:.1f} miles (+{distance_increase:.1f})")
                    
                    if total_distance < best_distance:
                        best_distance = total_distance
                        best_strategy = strategy_name
                        best_plans = modified_plans
                else:
                    print(f"   ❌ Strategy failed to assign all callout dogs")
                    
            except Exception as e:
                print(f"   ❌ Strategy failed: {e}")
        
        if best_plans:
            print(f"\n🏆 OPTIMAL STRATEGY: {best_strategy}")
            print(f"   📏 Total system distance: {best_distance:.1f} miles")
            print(f"   📈 Distance increase: +{best_distance - baseline_distance:.1f} miles")
            
            self.display_final_assignments(best_plans)
        else:
            print("❌ No viable reassignment strategy found")

    def strategy_nearest_driver(self, baseline_plans: Dict[str, DriverDayPlan]) -> Dict[str, DriverDayPlan]:
        """Strategy: Advanced greedy walk with individual assignment and domino logic"""
        print("   🎯 Using advanced greedy walk with domino optimization...")
        modified_plans = baseline_plans.copy()
        
        # Sort callout dogs by priority (hardest to place first)
        prioritized_dogs = self.prioritize_callout_dogs()
        
        successful_assignments = 0
        domino_moves = 0
        
        for i, callout_dog in enumerate(prioritized_dogs, 1):
            print(f"\n   🐕 Assignment {i}/{len(prioritized_dogs)}: {callout_dog.dog_name} ({callout_dog.dog_id})")
            print(f"      Needs: {callout_dog.num_dogs} dogs, groups {callout_dog.groups}")
            
            # Calculate scores for all available drivers
            driver_scores = []
            
            for driver_name, plan in modified_plans.items():
                if driver_name in self.unavailable_drivers:
                    continue
                    
                score_info = self.calculate_driver_score(callout_dog, driver_name, plan)
                if score_info['can_assign']:
                    driver_scores.append((driver_name, plan, score_info))
            
            # Sort by score (higher is better)
            driver_scores.sort(key=lambda x: x[2]['score'], reverse=True)
            
            # Display top candidates
            print(f"      📊 Top candidates:")
            for driver_name, plan, score_info in driver_scores[:3]:
                status = "✅" if score_info['has_capacity'] else "🔄"
                print(f"         {status} {driver_name} - {score_info['effective_distance']:.1f}mi, "
                      f"{plan.total_dogs} dogs, score: {score_info['score']:.1f} {score_info['compatibility']}")
            
            # Try to assign to best driver
            assigned = False
            for driver_name, plan, score_info in driver_scores:
                if score_info['has_capacity']:
                    # Direct assignment
                    self.add_callout_to_plan(plan, callout_dog)
                    print(f"      🎯 DIRECT ASSIGNMENT: {callout_dog.dog_name} → {driver_name} ({score_info['effective_distance']:.1f}mi)")
                    successful_assignments += 1
                    assigned = True
                    break
                elif score_info['score'] > 70:
                    # Try domino moves for high-scoring drivers
                    print(f"      🔄 {driver_name} is full but good candidate - trying domino moves...")
                    domino_success = self.attempt_domino_chain(callout_dog, driver_name, plan, modified_plans)
                    if domino_success:
                        self.add_callout_to_plan(plan, callout_dog)
                        domino_moves += domino_success
                        print(f"      🎯 DOMINO SUCCESS: {callout_dog.dog_name} → {driver_name} (after {domino_success} moves)")
                        successful_assignments += 1
                        assigned = True
                        break
            
            if not assigned:
                print(f"      ❌ Could not assign {callout_dog.dog_name}")
        
        print(f"\n   📊 Assignment Results:")
        print(f"      ✅ Successful assignments: {successful_assignments}/{len(prioritized_dogs)}")
        print(f"      🔄 Domino moves executed: {domino_moves}")
        
        return modified_plans

    def prioritize_callout_dogs(self) -> List[DogAssignment]:
        """Sort callout dogs by priority (hardest to place first)"""
        def priority_score(dog):
            score = dog.num_dogs * 100  # Physical dogs
            score += len(dog.groups) * 10  # Number of groups
            if len(dog.groups) > 1:
                score += 20  # Multi-group bonus
            return score
        
        prioritized = sorted(self.callout_dogs, key=priority_score, reverse=True)
        
        print(f"\n   📋 CALLOUT DOG PRIORITIZATION:")
        print(f"      Processing order (hardest to place first):")
        for i, dog in enumerate(prioritized, 1):
            groups_str = '&'.join(map(str, dog.groups)) if len(dog.groups) > 1 else str(dog.groups[0]) if dog.groups else 'unknown'
            if len(dog.groups) > 1:
                desc = f"multi-group ({groups_str})"
            else:
                desc = f"group {groups_str}"
            
            if dog.num_dogs > 1:
                desc = f"{dog.num_dogs} dogs, " + desc
            else:
                desc = f"single dog, " + desc
                
            priority = priority_score(dog)
            print(f"         {i}. {dog.dog_name} - {desc} (priority: {priority})")
        
        return prioritized

    def calculate_group_specific_capacity(self, driver_name: str) -> Dict[int, int]:
        """Calculate available capacity per group for a driver"""
        if driver_name not in self.driver_capacities:
            return {1: 0, 2: 0, 3: 0}
        
        total_capacity = self.driver_capacities[driver_name]
        
        # Get driver's day plan to see current assignments
        driver_groups = set()
        current_dogs_by_group = {1: 0, 2: 0, 3: 0}
        
        # Count current dogs by group from all driver assignments
        for dog_id, assignment in self.dog_assignments.items():
            if assignment.driver_name == driver_name and not assignment.is_callout:
                for group in assignment.groups:
                    current_dogs_by_group[group] += assignment.num_dogs
                    driver_groups.add(group)
        
        # Calculate available capacity per group
        available_capacity = {}
        for group in [1, 2, 3]:
            if group in driver_groups:
                # Driver handles this group - capacity is total minus current
                available_capacity[group] = max(0, total_capacity - current_dogs_by_group[group])
            else:
                # Driver doesn't handle this group
                available_capacity[group] = 0
        
        return available_capacity

    def can_driver_handle_multi_group_dog(self, driver_name: str, callout_dog: DogAssignment) -> Tuple[bool, str]:
        """Check if driver can handle ALL required groups for a multi-group dog"""
        group_capacities = self.calculate_group_specific_capacity(driver_name)
        
        print(f"         🔍 Group capacity check for {driver_name}:")
        for group in [1, 2, 3]:
            if group_capacities[group] > 0:
                print(f"            Group {group}: {group_capacities[group]} spaces available")
        
        # Check if driver can handle ALL required groups
        missing_groups = []
        insufficient_capacity = []
        
        for required_group in callout_dog.groups:
            available = group_capacities.get(required_group, 0)
            if available == 0:
                if required_group not in [g for assignments in self.dog_assignments.values() 
                                        if assignments.driver_name == driver_name 
                                        for g in assignments.groups]:
                    missing_groups.append(required_group)
                else:
                    insufficient_capacity.append(required_group)
            elif available < callout_dog.num_dogs:
                insufficient_capacity.append(required_group)
        
        if missing_groups:
            return False, f"Missing group(s): {missing_groups}"
        elif insufficient_capacity:
            return False, f"No capacity in group(s): {insufficient_capacity}"
        else:
            return True, f"Compatible with groups {callout_dog.groups}"
        """Calculate comprehensive score for assigning dog to driver"""
        # Get minimum distance to driver's route
        min_distance = self.find_min_distance_to_driver(callout_dog.dog_id, plan)
        
        # Calculate group compatibility
        effective_distance, compatibility = self.calculate_effective_distance_with_groups(
            callout_dog.dog_id, callout_dog.groups, plan
        )
        
        # Check capacity
        max_capacity = self.driver_capacities.get(driver_name, 9)
        current_capacity = plan.total_dogs
        has_capacity = current_capacity + callout_dog.num_dogs <= max_capacity
        
        # Calculate base score
        base_score = max(0, 100 - effective_distance * 10)  # Distance penalty
        
        # Group compatibility bonus/penalty
        if "exact_match" in compatibility:
            base_score += 20
        elif "adjacent_match" in compatibility:
            base_score += 10
        else:
            base_score -= 30  # Heavy penalty for no group compatibility
        
        # Load balancing factor
        load_factor = 1 - (current_capacity / max_capacity)
        base_score *= (0.7 + 0.3 * load_factor)  # Prefer less loaded drivers
        
        return {
            'score': base_score,
            'effective_distance': effective_distance,
            'min_distance': min_distance,
            'compatibility': compatibility,
            'has_capacity': has_capacity,
            'can_assign': base_score > 0 and (has_capacity or effective_distance < 3.0)
        }

    def calculate_effective_distance_with_groups(self, callout_dog_id: str, needed_groups: List[int], plan: DriverDayPlan) -> Tuple[float, str]:
        """Calculate effective distance considering group compatibility"""
        min_distance = float('inf')
        compatibility = "no_match"
        
        # Get current groups handled by this driver
        driver_groups = set()
        for phase in plan.phases:
            for dog_id in phase.pickup_dogs + phase.dropoff_dogs + phase.retain_dogs:
                dog_info = self.dog_assignments.get(dog_id)
                if dog_info:
                    driver_groups.update(dog_info.groups)
        
        # Calculate group compatibility
        needed_set = set(needed_groups)
        
        if driver_groups.intersection(needed_set):
            # Has exact group match
            compatibility = f"exact_match_{list(needed_set)}"
            distance_multiplier = 1.0
        else:
            # Check for adjacent groups (within 1 of each other)
            adjacent = False
            for needed_group in needed_groups:
                for driver_group in driver_groups:
                    if abs(needed_group - driver_group) <= 1:
                        adjacent = True
                        break
                if adjacent:
                    break
            
            if adjacent:
                compatibility = f"adjacent_match_({','.join(map(str, needed_groups))},{','.join(map(str, driver_groups))})"
                distance_multiplier = 2.0  # Double distance penalty for adjacent groups
            else:
                compatibility = "no_group_compatibility"
                return 100.0, compatibility  # Reject if no compatibility
        
        # Find minimum physical distance to any dog in the route
        for phase in plan.phases:
            for dog_id in phase.pickup_dogs + phase.dropoff_dogs + phase.retain_dogs:
                distance = self.get_distance(callout_dog_id, dog_id)
                min_distance = min(min_distance, distance)
        
        # Include parking and field in distance calculation
        if plan.parking_location:
            distance = self.get_distance(callout_dog_id, plan.parking_location)
            min_distance = min(min_distance, distance)
        
        if plan.field_location:
            distance = self.get_distance(callout_dog_id, plan.field_location)
            min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            min_distance = 100.0
        
        effective_distance = min_distance * distance_multiplier
        return effective_distance, compatibility

    def attempt_domino_chain(self, callout_dog: DogAssignment, target_driver: str, target_plan: DriverDayPlan, all_plans: Dict[str, DriverDayPlan], max_depth: int = 3) -> int:
        """Attempt multi-level domino chain to free space for optimal assignment"""
        return self.recursive_domino_move(target_driver, target_plan, all_plans, callout_dog.num_dogs, max_depth, 0, set())

    def recursive_domino_move(self, driver_name: str, plan: DriverDayPlan, all_plans: Dict[str, DriverDayPlan], 
                            needed_capacity: int, max_depth: int, current_depth: int, visited_drivers: Set[str]) -> int:
        """Recursively attempt domino moves to free capacity with group constraints"""
        if current_depth >= max_depth or driver_name in visited_drivers:
            return 0
        
        visited_drivers.add(driver_name)
        
        print(f"         🔍 Level {current_depth}: Trying to free space in {driver_name}")
        
        # Find outlier dogs in this driver's route
        outliers = self.find_outlier_dogs(plan)
        
        if not outliers:
            print(f"         📋 No outliers found in {driver_name}'s route")
            return 0
        
        print(f"         📋 Found {len(outliers)} potential outliers:")
        for dog_id, outlier_distance in outliers[:2]:  # Show top 2
            dog_info = self.dog_assignments.get(dog_id, DogAssignment(dog_id, dog_id, "", [], 1))
            print(f"            {dog_info.dog_name} ({dog_id}): closest neighbor {outlier_distance:.1f}mi away, groups {dog_info.groups}")
        
        # Try to move each outlier (distance-first, not first-available!)
        for dog_id, outlier_distance in outliers:
            dog_info = self.dog_assignments.get(dog_id)
            if not dog_info:
                continue
            
            print(f"         🎯 Trying to move outlier: {dog_info.dog_name} (groups {dog_info.groups})")
            
            # Score ALL potential destinations, don't pick first available
            destination_scores = []
            
            for candidate_driver, candidate_plan in all_plans.items():
                if candidate_driver == driver_name or candidate_driver in self.unavailable_drivers:
                    continue
                
                # Calculate comprehensive score for moving outlier to this driver
                score_info = self.calculate_driver_score(dog_info, candidate_driver, candidate_plan)
                
                print(f"            📍 {candidate_driver}: {score_info['effective_distance']:.1f}mi, "
                      f"score {score_info['score']:.1f}, groups: {score_info['group_reason']}")
                
                if score_info['can_assign']:
                    if score_info['has_capacity']:
                        # Direct move possible - add to candidates
                        move_efficiency = score_info['score'] / max(score_info['effective_distance'], 0.1)
                        destination_scores.append({
                            'driver': candidate_driver,
                            'plan': candidate_plan,
                            'score': score_info['score'],
                            'distance': score_info['effective_distance'],
                            'efficiency': move_efficiency,
                            'is_direct': True,
                            'recursive_moves': 0
                        })
                    elif score_info['score'] > 70 and current_depth < max_depth - 1:
                        # Try recursive domino
                        print(f"            🔄 {candidate_driver} is full but good candidate - trying recursive domino...")
                        recursive_moves = self.recursive_domino_move(
                            candidate_driver, candidate_plan, all_plans, 
                            dog_info.num_dogs, max_depth, current_depth + 1, visited_drivers.copy()
                        )
                        if recursive_moves > 0:
                            # Recursive domino succeeded - add to candidates with penalty for complexity
                            move_efficiency = score_info['score'] / (max(score_info['effective_distance'], 0.1) * (1 + recursive_moves * 0.2))
                            destination_scores.append({
                                'driver': candidate_driver,
                                'plan': candidate_plan,
                                'score': score_info['score'],
                                'distance': score_info['effective_distance'],
                                'efficiency': move_efficiency,
                                'is_direct': False,
                                'recursive_moves': recursive_moves
                            })
            
            # Sort by efficiency (best score per distance) - DISTANCE-FIRST optimization
            destination_scores.sort(key=lambda x: x['efficiency'], reverse=True)
            
            if destination_scores:
                best_destination = destination_scores[0]
                print(f"            ✅ BEST DESTINATION: {best_destination['driver']} "
                      f"(efficiency: {best_destination['efficiency']:.1f}, "
                      f"distance: {best_destination['distance']:.1f}mi)")
                
                # Execute the best move
                if best_destination['is_direct']:
                    success = self.move_dog_between_plans(dog_id, plan, best_destination['plan'])
                    if success:
                        return 1
                else:
                    # Recursive domino - already executed, just return count
                    success = self.move_dog_between_plans(dog_id, plan, best_destination['plan'])
                    if success:
                        return 1 + best_destination['recursive_moves']
            else:
                print(f"            ❌ No suitable destination found for {dog_info.dog_name}")
        
        print(f"         ❌ Could not find suitable move for any outlier in {driver_name}")
        return 0

    def find_outlier_dogs(self, plan: DriverDayPlan, distance_threshold: float = 1.0) -> List[Tuple[str, float]]:
        """Find dogs that are outliers (>1 mile from any neighbor)"""
        outliers = []
        
        # Collect all dogs in this driver's route
        all_dogs = []
        for phase in plan.phases:
            all_dogs.extend(phase.pickup_dogs + phase.dropoff_dogs + phase.retain_dogs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_dogs = []
        for dog_id in all_dogs:
            if dog_id not in seen:
                unique_dogs.append(dog_id)
                seen.add(dog_id)
        
        # Check each dog for isolation
        for dog_id in unique_dogs:
            closest_neighbor_distance = float('inf')
            
            # Find closest neighbor dog
            for other_dog_id in unique_dogs:
                if other_dog_id != dog_id:
                    distance = self.get_distance(dog_id, other_dog_id)
                    closest_neighbor_distance = min(closest_neighbor_distance, distance)
            
            # If no neighbor within threshold, it's an outlier
            if closest_neighbor_distance > distance_threshold:
                outliers.append((dog_id, closest_neighbor_distance))
        
        # Sort by distance (most isolated first)
        outliers.sort(key=lambda x: x[1], reverse=True)
        return outliers

    def move_dog_between_plans(self, dog_id: str, from_plan: DriverDayPlan, to_plan: DriverDayPlan):
        """Move a dog from one driver's plan to another with group constraint validation"""
        dog_info = self.dog_assignments.get(dog_id)
        if not dog_info:
            print(f"         ❌ Could not find dog info for {dog_id}")
            return False
        
        # Validate group constraints before moving
        can_handle, reason = self.can_driver_handle_multi_group_dog(to_plan.driver_name, dog_info)
        if not can_handle:
            print(f"         ❌ Cannot move {dog_info.dog_name} to {to_plan.driver_name}: {reason}")
            return False
        
        print(f"         ✅ Group validation passed: {dog_info.dog_name} → {to_plan.driver_name}")
        
        # Remove from source plan
        removed = False
        for phase in from_plan.phases:
            if dog_id in phase.pickup_dogs:
                phase.pickup_dogs.remove(dog_id)
                removed = True
            if dog_id in phase.dropoff_dogs:
                phase.dropoff_dogs.remove(dog_id)
            if dog_id in phase.retain_dogs:
                phase.retain_dogs.remove(dog_id)
        
        if removed:
            from_plan.total_dogs -= dog_info.num_dogs
            print(f"         📤 Removed {dog_info.dog_name} from {from_plan.driver_name}")
            
            # Add to destination plan using existing group-aware logic
            self.add_callout_to_plan(to_plan, dog_info)
            print(f"         📥 Added {dog_info.dog_name} to {to_plan.driver_name}")
            return True
        else:
            print(f"         ❌ Could not find {dog_info.dog_name} in {from_plan.driver_name}'s plan")
            return False

    def strategy_load_balanced(self, baseline_plans: Dict[str, DriverDayPlan]) -> Dict[str, DriverDayPlan]:
        """Strategy: Advanced load balancing with greedy walk and domino logic"""
        print("   🎯 Using advanced load balancing with domino optimization...")
        modified_plans = baseline_plans.copy()
        
        # Sort callout dogs by priority
        prioritized_dogs = self.prioritize_callout_dogs()
        
        successful_assignments = 0
        domino_moves = 0
        
        for i, callout_dog in enumerate(prioritized_dogs, 1):
            print(f"\n   🐕 Assignment {i}/{len(prioritized_dogs)}: {callout_dog.dog_name} ({callout_dog.dog_id})")
            
            # Get available drivers sorted by current load (lowest first)
            available_drivers = [(name, plan) for name, plan in modified_plans.items() 
                               if name not in self.unavailable_drivers]
            sorted_drivers = sorted(available_drivers, key=lambda x: x[1].total_dogs)
            
            assigned = False
            for driver_name, plan in sorted_drivers:
                score_info = self.calculate_driver_score(callout_dog, driver_name, plan)
                
                if score_info['can_assign']:
                    if score_info['has_capacity']:
                        # Direct assignment to least loaded driver
                        self.add_callout_to_plan(plan, callout_dog)
                        print(f"      🎯 LOAD-BALANCED: {callout_dog.dog_name} → {driver_name} "
                              f"({plan.total_dogs-callout_dog.num_dogs}→{plan.total_dogs} dogs)")
                        successful_assignments += 1
                        assigned = True
                        break
                    elif score_info['score'] > 60:  # Lower threshold for load balancing
                        # Try domino for reasonable candidates
                        domino_success = self.attempt_domino_chain(callout_dog, driver_name, plan, modified_plans)
                        if domino_success:
                            self.add_callout_to_plan(plan, callout_dog)
                            domino_moves += domino_success
                            print(f"      🎯 DOMINO LOAD-BALANCE: {callout_dog.dog_name} → {driver_name} "
                                  f"(after {domino_success} moves)")
                            successful_assignments += 1
                            assigned = True
                            break
            
            if not assigned:
                print(f"      ❌ Could not assign {callout_dog.dog_name}")
        
        print(f"\n   📊 Load-Balanced Results:")
        print(f"      ✅ Successful assignments: {successful_assignments}/{len(prioritized_dogs)}")
        print(f"      🔄 Domino moves executed: {domino_moves}")
        
        return modified_plans

    def strategy_minimal_impact(self, baseline_plans: Dict[str, DriverDayPlan]) -> Dict[str, DriverDayPlan]:
        """Strategy: Minimize total distance impact with advanced optimization"""
        print("   🎯 Using minimal impact optimization with domino logic...")
        modified_plans = baseline_plans.copy()
        
        # Sort callout dogs by priority
        prioritized_dogs = self.prioritize_callout_dogs()
        
        successful_assignments = 0
        domino_moves = 0
        total_impact = 0.0
        
        for i, callout_dog in enumerate(prioritized_dogs, 1):
            print(f"\n   🐕 Assignment {i}/{len(prioritized_dogs)}: {callout_dog.dog_name} ({callout_dog.dog_id})")
            
            best_option = None
            best_impact = float('inf')
            best_is_domino = False
            
            # Evaluate all drivers for distance impact
            for driver_name, plan in modified_plans.items():
                if driver_name in self.unavailable_drivers:
                    continue
                
                score_info = self.calculate_driver_score(callout_dog, driver_name, plan)
                
                if score_info['can_assign']:
                    if score_info['has_capacity']:
                        # Calculate direct assignment impact
                        original_distance = plan.total_distance
                        temp_plan = self.copy_plan(plan)
                        self.add_callout_to_plan(temp_plan, callout_dog)
                        new_distance = self.calculate_driver_day_distance(temp_plan)
                        impact = new_distance - original_distance
                        
                        print(f"      📊 {driver_name}: +{impact:.1f}mi impact (direct)")
                        
                        if impact < best_impact:
                            best_impact = impact
                            best_option = (driver_name, plan, False, 0)
                    
                    elif score_info['score'] > 70:
                        # Calculate domino assignment impact
                        temp_plans = {name: self.copy_plan(p) for name, p in modified_plans.items()}
                        domino_moves_needed = self.attempt_domino_chain(callout_dog, driver_name, temp_plans[driver_name], temp_plans)
                        
                        if domino_moves_needed > 0:
                            # Calculate total system impact after domino moves
                            temp_plans[driver_name].total_dogs += callout_dog.num_dogs  # Simulate assignment
                            
                            original_system_distance = sum(p.total_distance for p in modified_plans.values())
                            new_system_distance = sum(self.calculate_driver_day_distance(p) for p in temp_plans.values())
                            impact = new_system_distance - original_system_distance
                            
                            print(f"      📊 {driver_name}: +{impact:.1f}mi impact (domino, {domino_moves_needed} moves)")
                            
                            if impact < best_impact:
                                best_impact = impact
                                best_option = (driver_name, plan, True, domino_moves_needed)
            
            # Execute best option
            if best_option:
                driver_name, plan, is_domino, moves_needed = best_option
                
                if is_domino:
                    domino_success = self.attempt_domino_chain(callout_dog, driver_name, plan, modified_plans)
                    if domino_success:
                        self.add_callout_to_plan(plan, callout_dog)
                        domino_moves += domino_success
                        print(f"      🎯 MINIMAL IMPACT (DOMINO): {callout_dog.dog_name} → {driver_name} "
                              f"(+{best_impact:.1f}mi, {domino_success} moves)")
                    else:
                        print(f"      ❌ Domino chain failed for {callout_dog.dog_name}")
                        continue
                else:
                    self.add_callout_to_plan(plan, callout_dog)
                    print(f"      🎯 MINIMAL IMPACT (DIRECT): {callout_dog.dog_name} → {driver_name} "
                          f"(+{best_impact:.1f}mi)")
                
                successful_assignments += 1
                total_impact += best_impact
            else:
                print(f"      ❌ Could not assign {callout_dog.dog_name}")
        
        print(f"\n   📊 Minimal Impact Results:")
        print(f"      ✅ Successful assignments: {successful_assignments}/{len(prioritized_dogs)}")
        print(f"      🔄 Domino moves executed: {domino_moves}")
        print(f"      📏 Total distance impact: +{total_impact:.1f}mi")
        
        return modified_plans

    def find_min_distance_to_driver(self, callout_dog_id: str, plan: DriverDayPlan) -> float:
        """Find minimum distance from callout dog to any location in driver's plan"""
        min_distance = float('inf')
        
        # Check distance to parking
        if plan.parking_location:
            distance = self.get_distance(callout_dog_id, plan.parking_location)
            min_distance = min(min_distance, distance)
        
        # Check distance to field
        if plan.field_location:
            distance = self.get_distance(callout_dog_id, plan.field_location)
            min_distance = min(min_distance, distance)
        
        # Check distance to all dogs in plan
        for phase in plan.phases:
            for dog_id in phase.pickup_dogs + phase.dropoff_dogs + phase.retain_dogs:
                distance = self.get_distance(callout_dog_id, dog_id)
                min_distance = min(min_distance, distance)
        
        return min_distance

    def add_callout_to_plan(self, plan: DriverDayPlan, callout_dog: DogAssignment):
        """Add a callout dog to a driver's day plan"""
        # Determine which phases this dog belongs in based on groups
        for group in callout_dog.groups:
            # Find the appropriate phase
            phase_idx = group - 1  # Group 1 = Phase 0, etc.
            
            if phase_idx < len(plan.phases):
                phase = plan.phases[phase_idx]
                
                # Add to pickup for this phase
                if callout_dog.dog_id not in phase.pickup_dogs:
                    phase.pickup_dogs.append(callout_dog.dog_id)
                
                # Determine when this dog gets dropped off
                if len(callout_dog.groups) == 1:
                    # Single group - drop off at end of this phase
                    if callout_dog.dog_id not in phase.dropoff_dogs:
                        phase.dropoff_dogs.append(callout_dog.dog_id)
                else:
                    # Multi-group - figure out retention logic
                    max_group = max(callout_dog.groups)
                    if group < max_group:
                        # Retain for later phases
                        if callout_dog.dog_id not in phase.retain_dogs:
                            phase.retain_dogs.append(callout_dog.dog_id)
                    else:
                        # Last group - drop off
                        if callout_dog.dog_id not in phase.dropoff_dogs:
                            phase.dropoff_dogs.append(callout_dog.dog_id)
        
        plan.total_dogs += callout_dog.num_dogs

    def copy_plan(self, plan: DriverDayPlan) -> DriverDayPlan:
        """Create a deep copy of a driver day plan"""
        return DriverDayPlan(
            driver_name=plan.driver_name,
            parking_location=plan.parking_location,
            field_location=plan.field_location,
            phases=[DriverPhase(
                phase_num=p.phase_num,
                pickup_dogs=p.pickup_dogs.copy(),
                dropoff_dogs=p.dropoff_dogs.copy(),
                retain_dogs=p.retain_dogs.copy(),
                field_location=p.field_location
            ) for p in plan.phases],
            total_dogs=plan.total_dogs,
            total_distance=plan.total_distance
        )

    def display_final_assignments(self, driver_plans: Dict[str, DriverDayPlan]):
        """Display the final optimized assignments"""
        print(f"\n📋 FINAL OPTIMIZED DRIVER ASSIGNMENTS:")
        
        total_dogs = 0
        total_distance = 0.0
        
        for driver_name, plan in driver_plans.items():
            print(f"\n👤 {driver_name} ({plan.total_dogs} dogs, {plan.total_distance:.1f}mi):")
            
            for phase in plan.phases:
                print(f"   📋 Group {phase.phase_num}:")
                if phase.pickup_dogs:
                    pickup_names = [self.dog_assignments.get(dog_id, DogAssignment(dog_id, dog_id, "", [], 1)).dog_name 
                                  for dog_id in phase.pickup_dogs]
                    print(f"      🐕 Pickup: {', '.join(pickup_names)}")
                if phase.dropoff_dogs:
                    dropoff_names = [self.dog_assignments.get(dog_id, DogAssignment(dog_id, dog_id, "", [], 1)).dog_name 
                                   for dog_id in phase.dropoff_dogs]
                    print(f"      🏠 Dropoff: {', '.join(dropoff_names)}")
                if phase.retain_dogs:
                    retain_names = [self.dog_assignments.get(dog_id, DogAssignment(dog_id, dog_id, "", [], 1)).dog_name 
                                  for dog_id in phase.retain_dogs]
                    print(f"      🔄 Retain: {', '.join(retain_names)}")
            
            total_dogs += plan.total_dogs
            total_distance += plan.total_distance
        
        print(f"\n🎯 SYSTEM TOTALS:")
        print(f"   🐕 Total dogs: {total_dogs}")
        print(f"   📏 Total distance: {total_distance:.1f} miles")
        print(f"   👥 Active drivers: {len(driver_plans)}")
        
        # Count advanced features used
        callout_count = len([dog for dog in self.dog_assignments.values() if dog.is_callout])
        print(f"\n🚀 ADVANCED OPTIMIZATION FEATURES:")
        print(f"   ✅ Multi-phase day planning with pickup/dropoff/retention logic")
        print(f"   ✅ Priority-based callout assignment (multi-dog & multi-group first)")
        print(f"   ✅ Group-aware distance calculation with adjacency penalties")
        print(f"   ✅ Domino chain optimization with outlier detection")
        print(f"   ✅ Dynamic route updates and greedy walk assignment")
        print(f"   🎯 Processed {callout_count} callout reassignments optimally")

def main():
    """Main execution function"""
    print("🚀 Advanced Dog Logistics Optimization System")
    print("=" * 50)
    
    # Initialize system
    test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
    system = AdvancedDogLogisticsSystem(test_mode=test_mode)
    
    if not test_mode:
        # Load all data
        system.load_all_data()
        
        # Identify callouts
        system.identify_callouts()
        
        # Run advanced optimization
        system.reassign_callout_dogs_advanced()
    else:
        print("🧪 Test mode - skipping actual optimization")

if __name__ == "__main__":
    main()
