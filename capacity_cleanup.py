# capacity_cleanup.py
# Aggressive proximity-focused capacity cleanup with cascading

import pandas as pd
import numpy as np
import requests
import json
import os
from typing import Dict, List, Tuple
import gspread
from google.oauth2.service_account import Credentials
import re

class CapacityCleanup:
    def __init__(self):
        """Initialize the aggressive proximity capacity cleanup"""
        # AGGRESSIVE DISTANCE THRESHOLDS - Much tighter for quality
        self.IDEAL_DISTANCE = 0.2      # Perfect assignments
        self.MAX_DIRECT_DISTANCE = 0.5    # Direct moves - tightened from 1.0
        self.MAX_ADJACENT_DISTANCE = 0.3  # Adjacent groups - tightened from 0.75
        self.MAX_CASCADE_DISTANCE = 0.6   # Cascading moves - tightened from 0.8
        
        # AGGRESSIVE RADIUS EXPANSION - Smaller steps for precision
        self.RADIUS_START = 0.1
        self.RADIUS_INCREMENT = 0.05  # Smaller increments for better precision
        
        # Data containers (will be populated from main system)
        self.distance_matrix = None
        self.dog_assignments = None
        self.driver_capacities = None
        self.sheets_client = None
        self.moves_made = []

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
        """AGGRESSIVE: Check compatibility with much tighter distance requirements"""
        dog_set = set(dog_groups)
        target_set = set(target_groups)
        
        # Perfect match - same groups (tighter threshold)
        if dog_set.intersection(target_set):
            return distance <= self.MAX_DIRECT_DISTANCE
        
        # Adjacent groups - much tighter threshold
        adjacent_pairs = [(1, 2), (2, 3), (2, 1), (3, 2)]
        for dog_group in dog_set:
            for target_group in target_set:
                if (dog_group, target_group) in adjacent_pairs:
                    return distance <= self.MAX_ADJACENT_DISTANCE
        
        return False

    def calculate_proximity_score(self, distance, has_capacity, is_perfect_group_match):
        """AGGRESSIVE: Heavy penalty for distance, big bonus for proximity"""
        if distance >= 100:  # Skip placeholders
            return -1000
        
        # Base score heavily penalizes distance
        base_score = 1000 - (distance * 1500)  # 1.5x penalty per mile
        
        # Capacity bonus
        if has_capacity:
            base_score += 500
        
        # Perfect group match bonus
        if is_perfect_group_match:
            base_score += 300
        
        # EXTREME proximity bonuses
        if distance <= 0.1:
            base_score += 1000  # Huge bonus for very close
        elif distance <= 0.2:
            base_score += 500   # Big bonus for close
        elif distance <= 0.3:
            base_score += 200   # Moderate bonus
        
        return base_score

    def build_driver_assignments(self, exclude_moves=None):
        """Build current assignments by driver, accounting for pending moves"""
        if exclude_moves is None:
            exclude_moves = []
        
        driver_assignments = {}
        
        for assignment in self.dog_assignments:
            combined = assignment.get('combined', '')
            if not combined or ':' not in combined:
                continue
            
            original_driver = combined.split(':', 1)[0].strip()
            assignment_part = combined.split(':', 1)[1].strip()
            groups = self.extract_groups_from_assignment(assignment_part)
            
            # Check if this dog has been moved in pending moves
            current_driver = original_driver
            for move in exclude_moves:
                if move['dog_id'] == assignment['dog_id']:
                    current_driver = move['to_driver']
                    break
            
            if current_driver not in driver_assignments:
                driver_assignments[current_driver] = []
            
            driver_assignments[current_driver].append({
                'dog_id': assignment['dog_id'],
                'dog_name': assignment['dog_name'],
                'groups': groups,
                'num_dogs': assignment['num_dogs']
            })
        
        return driver_assignments

    def analyze_capacity_violations(self):
        """Find all drivers who are over capacity"""
        print("\n🔍 Analyzing capacity violations with current assignments...")
        
        # Build current assignments by driver
        driver_assignments = self.build_driver_assignments()
        
        # Check each driver for violations
        violations = []
        
        for driver, assignments in driver_assignments.items():
            if driver not in self.driver_capacities:
                continue
            
            capacity = self.driver_capacities[driver]
            current_load = {'group1': 0, 'group2': 0, 'group3': 0}
            
            # Calculate current load
            for assignment in assignments:
                for group in assignment['groups']:
                    group_key = f'group{group}'
                    current_load[group_key] += assignment['num_dogs']
            
            # Check for violations
            for group_key in ['group1', 'group2', 'group3']:
                max_cap = capacity.get(group_key, 0)
                current = current_load[group_key]
                
                if current > max_cap:
                    overage = current - max_cap
                    group_num = int(group_key[-1])
                    
                    # Find dogs in this group that could be moved
                    moveable_dogs = []
                    for assignment in assignments:
                        if group_num in assignment['groups']:
                            moveable_dogs.append(assignment)
                    
                    violations.append({
                        'driver': driver,
                        'group': group_num,
                        'capacity': max_cap,
                        'current': current,
                        'overage': overage,
                        'moveable_dogs': moveable_dogs
                    })
                    
                    print(f"🚨 {driver} - Group {group_num}: {current}/{max_cap} ({overage} over)")
        
        print(f"📊 Found {len(violations)} capacity violations")
        return violations

    def find_optimal_targets_for_dog(self, dog_to_move, exclude_driver, pending_moves=None):
        """AGGRESSIVE: Find targets prioritizing extreme proximity"""
        if pending_moves is None:
            pending_moves = []
            
        targets = []
        
        # Build current assignments accounting for pending moves
        driver_assignments = self.build_driver_assignments(pending_moves)
        
        # Remove the source driver
        if exclude_driver in driver_assignments:
            del driver_assignments[exclude_driver]
        
        print(f"      🎯 Scanning {len(driver_assignments)} drivers for {dog_to_move['dog_name']}")
        
        # Check each driver with aggressive scoring
        for driver, assignments in driver_assignments.items():
            if driver not in self.driver_capacities:
                continue
            
            # Find closest distance to this driver
            closest_distance = float('inf')
            closest_via_dog = None
            
            for assignment in assignments:
                distance = self.get_distance(dog_to_move['dog_id'], assignment['dog_id'])
                if distance < closest_distance and distance < 100:  # Skip placeholders
                    closest_distance = distance
                    closest_via_dog = assignment
            
            if closest_via_dog is None or closest_distance > self.MAX_CASCADE_DISTANCE:
                continue
            
            # Check group compatibility with aggressive thresholds
            dog_groups = set(dog_to_move['groups'])
            target_groups = set(closest_via_dog['groups'])
            is_perfect_match = bool(dog_groups.intersection(target_groups))
            
            if not self.check_group_compatibility(dog_to_move['groups'], closest_via_dog['groups'], closest_distance):
                continue
            
            # Check capacity
            capacity = self.driver_capacities[driver]
            current_load = {'group1': 0, 'group2': 0, 'group3': 0}
            
            # Calculate current load
            for assignment in assignments:
                for group in assignment['groups']:
                    group_key = f'group{group}'
                    current_load[group_key] += assignment['num_dogs']
            
            # Check if we can add this dog
            can_accept = True
            for group in dog_to_move['groups']:
                group_key = f'group{group}'
                if current_load[group_key] + dog_to_move['num_dogs'] > capacity.get(group_key, 0):
                    can_accept = False
                    break
            
            # Calculate aggressive proximity score
            proximity_score = self.calculate_proximity_score(
                closest_distance, 
                can_accept, 
                is_perfect_match
            )
            
            targets.append({
                'driver': driver,
                'distance': closest_distance,
                'via_dog': closest_via_dog['dog_name'],
                'via_dog_id': closest_via_dog['dog_id'],
                'capacity_available': can_accept,
                'proximity_score': proximity_score,
                'is_perfect_match': is_perfect_match
            })
        
        # Sort by proximity score (highest first), then by distance (lowest first)
        targets.sort(key=lambda x: (-x['proximity_score'], x['distance']))
        
        if targets:
            print(f"         📊 Top 3 targets by proximity score:")
            for i, target in enumerate(targets[:3]):
                status = "✅ AVAILABLE" if target['capacity_available'] else "🔄 NEEDS CASCADING"
                match = "🎯 PERFECT" if target['is_perfect_match'] else "📍 ADJACENT"
                print(f"         {i+1}. {target['driver']} - {target['distance']:.2f}mi ({target['proximity_score']:.0f} pts) {match} {status}")
        
        return targets

    def find_cascading_targets_aggressive(self, dog_to_move, exclude_driver, pending_moves, max_distance):
        """AGGRESSIVE: Find targets using tight radius expansion"""
        current_radius = self.RADIUS_START
        
        print(f"        🔍 Aggressive radius search (start: {current_radius}mi, max: {max_distance}mi)")
        
        while current_radius <= max_distance:
            targets = self.find_optimal_targets_for_dog(dog_to_move, exclude_driver, pending_moves)
            
            # Filter to current radius and available capacity
            radius_targets = [
                t for t in targets 
                if t['distance'] <= current_radius and t['capacity_available']
            ]
            
            if radius_targets:
                best = radius_targets[0]
                print(f"        ✅ Found optimal target at {current_radius}mi: {best['driver']} ({best['proximity_score']:.0f} pts)")
                return radius_targets
            
            current_radius += self.RADIUS_INCREMENT
            current_radius = round(current_radius, 2)
        
        print(f"        ❌ No targets within {max_distance}mi")
        return []

    def attempt_cascading_move(self, dog_to_move, blocked_driver, pending_moves):
        """AGGRESSIVE: Attempt cascading with tight distance controls"""
        print(f"    🔄 Attempting AGGRESSIVE cascading for {dog_to_move['dog_name']} → {blocked_driver}")
        
        # Get current assignments for the blocked driver
        driver_assignments = self.build_driver_assignments(pending_moves)
        blocked_driver_dogs = driver_assignments.get(blocked_driver, [])
        
        if not blocked_driver_dogs:
            print(f"    ❌ No dogs found for {blocked_driver}")
            return None
        
        # Sort by aggressive proximity criteria (single group + close matches first)
        move_candidates = sorted(blocked_driver_dogs, key=lambda x: (
            len(x['groups']),           # Single group first
            x['num_dogs'],              # Fewer dogs first
            x['dog_name']               # Consistent ordering
        ))
        
        print(f"    📋 Trying to move {len(move_candidates)} dogs from {blocked_driver} (closest first):")
        
        # Try to move each dog with aggressive constraints
        for i, candidate in enumerate(move_candidates):
            if i >= 3:  # Only try top 3 candidates to keep it fast
                break
                
            print(f"      {i+1}. {candidate['dog_name']} (groups: {candidate['groups']}, {candidate['num_dogs']} dogs)")
            
            # Find targets using aggressive proximity search
            cascading_targets = self.find_cascading_targets_aggressive(
                candidate, 
                blocked_driver, 
                pending_moves, 
                self.MAX_CASCADE_DISTANCE
            )
            
            if cascading_targets:
                best_target = cascading_targets[0]
                print(f"      ✅ AGGRESSIVE MATCH: {candidate['dog_name']} → {best_target['driver']}")
                print(f"         📏 Distance: {best_target['distance']:.2f}mi")
                print(f"         🎯 Score: {best_target['proximity_score']:.0f} points")
                
                # Create the cascading move
                cascading_move = {
                    'dog_id': candidate['dog_id'],
                    'dog_name': candidate['dog_name'],
                    'from_driver': blocked_driver,
                    'to_driver': best_target['driver'],
                    'distance': best_target['distance'],
                    'proximity_score': best_target['proximity_score'],
                    'reason': f"aggressive_cascading_for_{dog_to_move['dog_name']}"
                }
                
                return cascading_move
            else:
                print(f"      ❌ No close targets for {candidate['dog_name']}")
        
        print(f"    ❌ Aggressive cascading failed for {blocked_driver}")
        return None

    def fix_capacity_violations(self):
        """AGGRESSIVE: Fix violations prioritizing extreme proximity"""
        print("\n🔧 AGGRESSIVE CAPACITY CLEANUP - Extreme proximity prioritization")
        print(f"📏 Thresholds: Direct ≤{self.MAX_DIRECT_DISTANCE}mi, Adjacent ≤{self.MAX_ADJACENT_DISTANCE}mi, Cascading ≤{self.MAX_CASCADE_DISTANCE}mi")
        
        violations = self.analyze_capacity_violations()
        if not violations:
            print("✅ No capacity violations found!")
            return []
        
        all_moves = []
        
        for violation in violations:
            print(f"\n🎯 AGGRESSIVE FIX: {violation['driver']} - Group {violation['group']}")
            print(f"   Need to move {violation['overage']} dogs from group {violation['group']}")
            
            # Sort by aggressive criteria (single group + fewer dogs first)
            moveable = sorted(violation['moveable_dogs'], key=lambda x: (
                len(x['groups']),
                x['num_dogs'],
                x['dog_name']
            ))
            
            dogs_moved = 0
            for dog in moveable:
                if dogs_moved >= violation['overage']:
                    break
                
                print(f"   📍 AGGRESSIVE PLACEMENT: {dog['dog_name']} (groups: {dog['groups']}, {dog['num_dogs']} dogs)")
                
                # Try direct move with aggressive proximity scoring
                targets = self.find_optimal_targets_for_dog(dog, violation['driver'], all_moves)
                available_targets = [t for t in targets if t['capacity_available']]
                
                if available_targets:
                    # Direct move possible
                    best_target = available_targets[0]
                    print(f"   ✅ DIRECT AGGRESSIVE MATCH: {best_target['driver']}")
                    print(f"      📏 Distance: {best_target['distance']:.2f}mi")
                    print(f"      🎯 Score: {best_target['proximity_score']:.0f} points")
                    
                    move = {
                        'dog_id': dog['dog_id'],
                        'dog_name': dog['dog_name'],
                        'from_driver': violation['driver'],
                        'to_driver': best_target['driver'],
                        'distance': best_target['distance'],
                        'proximity_score': best_target['proximity_score'],
                        'reason': f"aggressive_direct_group_{violation['group']}"
                    }
                    
                    all_moves.append(move)
                    dogs_moved += dog['num_dogs']
                
                else:
                    # Try aggressive cascading
                    blocked_targets = [t for t in targets if not t['capacity_available']]
                    
                    if blocked_targets:
                        best_blocked = blocked_targets[0]
                        print(f"   🔄 BLOCKED TARGET: {best_blocked['driver']} at {best_blocked['distance']:.2f}mi")
                        print(f"      🎯 Score: {best_blocked['proximity_score']:.0f} points - trying aggressive cascading")
                        
                        cascading_move = self.attempt_cascading_move(
                            dog, 
                            best_blocked['driver'], 
                            all_moves
                        )
                        
                        if cascading_move:
                            print(f"   ✅ AGGRESSIVE CASCADING SUCCESS!")
                            print(f"      🔄 Move 1: {cascading_move['dog_name']} → {cascading_move['to_driver']} ({cascading_move['distance']:.2f}mi)")
                            print(f"      🔄 Move 2: {dog['dog_name']} → {best_blocked['driver']} ({best_blocked['distance']:.2f}mi)")
                            
                            # Add both moves
                            all_moves.append(cascading_move)
                            
                            main_move = {
                                'dog_id': dog['dog_id'],
                                'dog_name': dog['dog_name'],
                                'from_driver': violation['driver'],
                                'to_driver': best_blocked['driver'],
                                'distance': best_blocked['distance'],
                                'proximity_score': best_blocked['proximity_score'],
                                'reason': f"aggressive_cascading_group_{violation['group']}"
                            }
                            
                            all_moves.append(main_move)
                            dogs_moved += dog['num_dogs']
                        else:
                            print(f"   ❌ Aggressive cascading failed for {dog['dog_name']}")
                    else:
                        print(f"   ❌ No targets found for {dog['dog_name']}")
        
        self.moves_made = all_moves
        
        # Enhanced summary with proximity stats
        if all_moves:
            direct_moves = [m for m in all_moves if 'direct' in m['reason']]
            cascading_moves = [m for m in all_moves if 'cascading' in m['reason']]
            avg_distance = sum(m['distance'] for m in all_moves) / len(all_moves)
            close_moves = len([m for m in all_moves if m['distance'] <= 0.3])
            
            print(f"\n📊 AGGRESSIVE CLEANUP SUMMARY:")
            print(f"   🎯 {len(all_moves)} total moves planned")
            print(f"   ➡️ {len(direct_moves)} direct moves")
            print(f"   🔄 {len(cascading_moves)} cascading moves")
            print(f"   📏 Average distance: {avg_distance:.2f}mi")
            print(f"   🎯 Close moves (≤0.3mi): {close_moves}/{len(all_moves)} ({close_moves/len(all_moves)*100:.0f}%)")
            print(f"   📈 Proximity optimization: {close_moves/len(all_moves)*100:.0f}% close placement")
        
        return all_moves

    def write_moves_to_sheets(self, moves):
        """Write the aggressive capacity fixes back to Google Sheets"""
        try:
            if not moves:
                print("✅ No moves to write")
                return True
            
            print(f"\n📝 Writing {len(moves)} aggressive capacity fixes to Google Sheets...")
            
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
            
            # Enhanced processing with proximity stats
            direct_moves = [m for m in moves if 'direct' in m['reason']]
            cascading_moves = [m for m in moves if 'cascading' in m['reason']]
            
            print(f"   📊 Processing {len(direct_moves)} direct + {len(cascading_moves)} cascading moves...")
            
            # Process each move
            for move in moves:
                dog_id = str(move['dog_id']).strip()
                to_driver = move['to_driver']
                distance = move['distance']
                score = move.get('proximity_score', 0)
                
                move_type = "🔄 CASCADING" if 'cascading' in move['reason'] else "➡️ DIRECT"
                proximity = "🎯 CLOSE" if distance <= 0.3 else "📍 MODERATE" if distance <= 0.5 else "📏 FAR"
                
                # Find the row for this dog
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
                                
                                updates.append({
                                    'range': cell_address,
                                    'values': [[new_combined]]
                                })
                                
                                print(f"  {move_type} {dog_id}: {move['from_driver']} → {to_driver} ({distance:.2f}mi, {score:.0f}pts) {proximity}")
                            break
            
            if updates:
                worksheet.batch_update(updates)
                
                # Enhanced success metrics
                avg_distance = sum(m['distance'] for m in moves) / len(moves)
                close_moves = len([m for m in moves if m['distance'] <= 0.3])
                
                print(f"\n✅ AGGRESSIVE CLEANUP SUCCESSFUL!")
                print(f"   📊 {len(updates)} total fixes applied")
                print(f"   📏 Average distance: {avg_distance:.2f}mi")
                print(f"   🎯 Close placements: {close_moves}/{len(moves)} ({close_moves/len(moves)*100:.0f}%)")
                print(f"   📈 Proximity excellence: {close_moves/len(moves)*100:.0f}% within 0.3mi")
                
                # Enhanced Slack notification
                slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')
                if slack_webhook:
                    try:
                        message = f"🔧 Aggressive Capacity Cleanup: {len(updates)} moves, {avg_distance:.2f}mi avg, {close_moves/len(moves)*100:.0f}% close placements"
                        slack_message = {"text": message}
                        requests.post(slack_webhook, json=slack_message, timeout=10)
                        print("📱 Enhanced proximity notification sent")
                    except:
                        pass
                
                return True
            else:
                print("❌ No valid updates to make")
                return False
            
        except Exception as e:
            print(f"❌ Error writing aggressive moves: {e}")
            return False


def main():
    """Main function for aggressive capacity cleanup"""
    print("🔧 Aggressive Proximity Capacity Cleanup")
    print("🎯 Extreme proximity prioritization with smart cascading")
    print("📏 Tight thresholds: ≤0.5mi direct, ≤0.3mi adjacent, ≤0.6mi cascading")
    print("🏆 Goal: Maximum proximity with zero tolerance for distant assignments")
    print("=" * 70)
    
    cleanup = CapacityCleanup()
    
    # This would be called from the main script, so no setup needed here
    print("⚠️ This cleanup should be called from the main script")
    print("   Use: run_capacity_cleanup(main_system) after main assignments")


if __name__ == "__main__":
    main()
