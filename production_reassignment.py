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
            self.sheet = self.gc.open_by_key("1H4jg0pIApXAFWlP1G6WZDJaXJNXhD3ywqWa7KdLTrJk")
            print("✅ Google Sheets client setup successful")
        except Exception as e:
            print(f"❌ Google Sheets setup failed: {e}")
            raise

    def load_all_data(self):
        """Load and parse all data including location types"""
        print("\n⬇️ Loading data from Google Sheets...")
        
        # Load distance matrix
        print("📊 Loading distance matrix...")
        matrix_sheet = self.sheet.worksheet("Distance Matrix")
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
        
        # Load dog assignments with location parsing
        print("🐕 Loading dog assignments...")
        map_sheet = self.sheet.worksheet("Map")
        map_data = map_sheet.get_all_values()
        
        print(f"📊 Map sheet shape: ({len(map_data)}, {len(map_data[0])})")
        
        headers = [h.strip().lower() for h in map_data[0]]
        dog_id_col = headers.index('dog id') if 'dog id' in headers else 0
        dog_name_col = headers.index('dog name') if 'dog name' in headers else 1  
        assignment_col = headers.index('assignment') if 'assignment' in headers else 2
        
        regular_assignments = 0
        
        for row in map_data[1:]:
            if len(row) <= max(dog_id_col, dog_name_col, assignment_col):
                continue
                
            dog_id = row[dog_id_col].strip()
            dog_name = row[dog_name_col].strip() 
            assignment = row[assignment_col].strip()
            
            if not dog_id or not assignment:
                continue
            
            # Parse location type from dog name
            location_type = 'dog'  # default
            if 'parking' in dog_name.lower():
                location_type = 'parking'
            elif 'field' in dog_name.lower():
                location_type = 'field'
            
            # Parse driver and groups from assignment  
            driver_name, groups, num_dogs = self.parse_assignment(assignment)
            
            if driver_name:
                # Store location info
                self.location_info[dog_id] = LocationInfo(
                    dog_id=dog_id,
                    location_type=location_type,
                    driver_name=driver_name,
                    groups=groups
                )
                
                # Store assignment info
                self.dog_assignments[dog_id] = DogAssignment(
                    dog_id=dog_id,
                    dog_name=dog_name,
                    driver_name=driver_name, 
                    groups=groups,
                    num_dogs=num_dogs
                )
                regular_assignments += 1
                
        print(f"✅ Loaded {regular_assignments} regular assignments")
        
        # Load driver capacities
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
        if not assignment or assignment.lower() in ['callout', 'call out']:
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
            if not assignment.driver_name or assignment.driver_name.lower() in ['callout', 'call out']:
                assignment.is_callout = True
                self.callout_dogs.append(assignment)
                
        print(f"🚨 Found {len(self.callout_dogs)} callouts that need drivers assigned:")
        for dog in self.callout_dogs:
            groups_str = '&'.join(map(str, dog.groups)) if len(dog.groups) > 1 else str(dog.groups[0]) if dog.groups else 'unknown'
            print(f"   - {dog.dog_name} ({dog.dog_id}) - {dog.num_dogs} dogs, groups [{groups_str}]")

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
        
        # Build day plan for each driver
        for driver_name, assignments in driver_assignments.items():
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
        
        # Try different reassignment strategies
        strategies = [
            ("Nearest Driver Assignment", self.strategy_nearest_driver),
            ("Load Balanced Assignment", self.strategy_load_balanced),
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
        """Strategy: Assign each callout dog to nearest available driver"""
        modified_plans = baseline_plans.copy()
        
        for callout_dog in self.callout_dogs:
            best_driver = None
            best_distance = float('inf')
            
            for driver_name, plan in modified_plans.items():
                # Check capacity
                current_capacity = plan.total_dogs
                max_capacity = self.driver_capacities.get(driver_name, 9)
                
                if current_capacity + callout_dog.num_dogs <= max_capacity:
                    # Find minimum distance to this driver's route
                    min_distance = self.find_min_distance_to_driver(callout_dog.dog_id, plan)
                    
                    if min_distance < best_distance:
                        best_distance = min_distance
                        best_driver = driver_name
            
            if best_driver:
                # Add callout dog to this driver's plan
                self.add_callout_to_plan(modified_plans[best_driver], callout_dog)
                print(f"   ✅ {callout_dog.dog_name} → {best_driver} ({best_distance:.1f}mi)")
            else:
                print(f"   ❌ Could not assign {callout_dog.dog_name}")
                
        return modified_plans

    def strategy_load_balanced(self, baseline_plans: Dict[str, DriverDayPlan]) -> Dict[str, DriverDayPlan]:
        """Strategy: Balance load across all drivers"""
        modified_plans = baseline_plans.copy()
        
        # Sort drivers by current load
        sorted_drivers = sorted(modified_plans.items(), key=lambda x: x[1].total_dogs)
        
        for callout_dog in self.callout_dogs:
            # Find driver with lowest load who can handle this dog
            for driver_name, plan in sorted_drivers:
                current_capacity = plan.total_dogs
                max_capacity = self.driver_capacities.get(driver_name, 9)
                
                if current_capacity + callout_dog.num_dogs <= max_capacity:
                    self.add_callout_to_plan(plan, callout_dog)
                    print(f"   ✅ {callout_dog.dog_name} → {driver_name} (load balancing)")
                    # Re-sort for next assignment
                    sorted_drivers = sorted(modified_plans.items(), key=lambda x: x[1].total_dogs)
                    break
            else:
                print(f"   ❌ Could not assign {callout_dog.dog_name}")
                
        return modified_plans

    def strategy_minimal_impact(self, baseline_plans: Dict[str, DriverDayPlan]) -> Dict[str, DriverDayPlan]:
        """Strategy: Minimize total distance impact"""
        modified_plans = baseline_plans.copy()
        
        for callout_dog in self.callout_dogs:
            best_driver = None
            best_impact = float('inf')
            
            for driver_name, plan in modified_plans.items():
                # Check capacity
                current_capacity = plan.total_dogs
                max_capacity = self.driver_capacities.get(driver_name, 9)
                
                if current_capacity + callout_dog.num_dogs <= max_capacity:
                    # Calculate distance impact of adding this dog
                    original_distance = plan.total_distance
                    
                    # Create temporary plan with this callout
                    temp_plan = self.copy_plan(plan)
                    self.add_callout_to_plan(temp_plan, callout_dog)
                    new_distance = self.calculate_driver_day_distance(temp_plan)
                    
                    impact = new_distance - original_distance
                    
                    if impact < best_impact:
                        best_impact = impact
                        best_driver = driver_name
            
            if best_driver:
                self.add_callout_to_plan(modified_plans[best_driver], callout_dog)
                print(f"   ✅ {callout_dog.dog_name} → {best_driver} (+{best_impact:.1f}mi impact)")
            else:
                print(f"   ❌ Could not assign {callout_dog.dog_name}")
                
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
