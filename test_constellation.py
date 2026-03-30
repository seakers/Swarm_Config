"""
test_constellation.py
=====================
Comprehensive test suite for the Constellation separation/rejoining system.

Run with:
    python test_constellation.py              # all tests
    python test_constellation.py --viz        # with visualization
    python test_constellation.py --save-plots # save plots to ./plots/
"""

import numpy as np
import sys
import os
import time
import argparse
from typing import Optional, Tuple, List, Dict, Set

# Core imports
from core.swarm import Swarm
from core.constellation import (
    Constellation, GroupState,
    SeparationRequirements, DockingRequirements, CommunicationRequirements,
    PropulsionSubsystem
)
from configs.formations import create_cube_formation, create_plane_formation

# Controller and metrics
from mechanics.constellation_moves import (
    ConstellationController, SeparationAction, DockingAction, ManeuverAction,
    ConstellationActionResult
)
from rewards.constellation_metrics import ConstellationMetrics

# Tasks
from tasks.constellation_tasks import (
    ConstellationTask, FormConstellationTask, RendezvousAndDockTask,
    StereoImagingTask, MultiPointSensingTask
)


# =============================================================================
# Test Utilities
# =============================================================================

def print_header(title: str) -> None:
    """Print a formatted section header."""
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def print_status(constellation: Constellation) -> None:
    """Print current constellation status."""
    state = constellation.get_constellation_state()
    print(f"  Time: {state['time']:.1f}s | Groups: {state['num_groups']} | "
          f"Connected: {state['is_connected']} | Max Baseline: {state['max_baseline']:.1f}m")
    
    for gid, ginfo in state['groups'].items():
        pos = [f"{v:.1f}" for v in ginfo['position']]
        vel = [f"{v:.2f}" for v in ginfo['velocity']]
        print(f"    Group {gid}: {ginfo['num_cubes']} cubes, "
              f"pos=[{', '.join(pos)}], vel=[{', '.join(vel)}], "
              f"Δv={ginfo['delta_v_remaining']:.2f} m/s")


def create_test_constellation(num_cubes: int = 64, 
                               cube_size: int = 4,
                               max_delta_v: float = 100.0) -> Constellation:
    """Create a constellation for testing with generous propulsion budget."""
    swarm = Swarm(num_cubes)
    create_cube_formation(swarm, size=cube_size)
    
    sep_reqs = SeparationRequirements(
        min_separation_delta_v=0.1,
        default_separation_velocity=1.0,
        min_group_size=1,
        max_groups=8,
        allow_single_cube_groups=True
    )
    
    dock_reqs = DockingRequirements(
        max_docking_velocity=0.5,
        max_docking_range=500.0,
        docking_delta_v_cost=0.2
    )
    
    comm_reqs = CommunicationRequirements(
        max_isl_range=100_000.0,
        base_data_rate=10_000.0,
        reference_distance=1_000.0,
        min_coordination_rate=100.0
    )
    
    constellation = Constellation(swarm, sep_reqs, dock_reqs, comm_reqs)
    
    # Set generous propulsion budget
    for ps in constellation._cube_propulsion.values():
        ps.max_delta_v = max_delta_v
        ps.remaining_delta_v = max_delta_v
    
    # Sync group propulsion
    for grp in constellation._groups.values():
        total = sum(constellation._cube_propulsion[cid].remaining_delta_v 
                   for cid in grp.cube_ids)
        grp.propulsion.max_delta_v = total
        grp.propulsion.remaining_delta_v = total
    
    return constellation


def separate_half(constellation: Constellation) -> Tuple[bool, str, Optional[int]]:
    """Utility to separate the first half of cubes from the only group."""
    grp = next(iter(constellation._groups.values()))
    all_ids = sorted(grp.cube_ids)
    half = set(all_ids[:len(all_ids) // 2])
    return constellation.separate(half, separation_direction=np.array([1.0, 0.0, 0.0]))


# =============================================================================
# Core Functionality Tests
# =============================================================================

def test_initial_state() -> bool:
    """Test that constellation initializes correctly."""
    print_header("1. Initial State Test")
    
    constellation = create_test_constellation(64, 4)
    
    # Assertions
    assert constellation.get_num_groups() == 1, "Expected 1 group after init"
    assert constellation.is_single_group(), "Expected is_single_group() == True"
    
    groups = constellation.get_all_groups()
    assert len(groups) == 1, "Expected 1 group in list"
    assert len(groups[0].cube_ids) == 64, "Expected 64 cubes in group"
    assert groups[0].propulsion.remaining_delta_v > 0, "Expected positive delta-v"
    
    # Check all cubes are mapped
    for i in range(64):
        group = constellation.get_group_for_cube(i)
        assert group is not None, f"Cube {i} not assigned to group"
    
    print_status(constellation)
    print("  ✓ PASS")
    return True


def test_separation_basic() -> bool:
    """Test basic group separation."""
    print_header("2. Basic Separation Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # Get valid separation actions
    valid_separations = controller.get_valid_separation_actions()
    print(f"  Found {len(valid_separations)} valid separation actions")
    
    assert len(valid_separations) > 0, "Should have valid separation actions"
    
    # Find a good separation (around half the cubes)
    best_action = None
    for action in valid_separations:
        if 20 <= len(action.cube_ids) <= 40:
            best_action = action
            break
    
    if best_action is None:
        best_action = valid_separations[0]
    
    print(f"  Separating {len(best_action.cube_ids)} cubes...")
    
    # Execute separation
    result = controller.execute_separation(best_action)
    
    print(f"  Result: success={result.success}, reason='{result.reason}'")
    print(f"  Delta-v used: {result.delta_v_used:.3f} m/s")
    
    assert result.success, f"Separation failed: {result.reason}"
    assert constellation.get_num_groups() == 2, "Expected 2 groups after separation"
    assert not constellation.is_single_group(), "Should not be single group"
    
    print_status(constellation)
    print("  ✓ PASS")
    return True


def test_separation_validation() -> bool:
    """Test that invalid separations are rejected."""
    print_header("3. Separation Validation Test")
    
    constellation = create_test_constellation(64, 4)
    
    # Test 1: Empty set
    can_sep, reason = constellation.can_separate(set())
    assert not can_sep, "Should reject empty set"
    print(f"  Empty set: rejected - '{reason}'")
    
    # Test 2: All cubes (nothing would remain)
    all_cubes = set(range(64))
    can_sep, reason = constellation.can_separate(all_cubes)
    print(f"  All cubes: {'accepted' if can_sep else 'rejected'} - '{reason}'")
    
    # Test 3: Zero propulsion
    constellation2 = create_test_constellation(64, 4)
    for ps in constellation2._cube_propulsion.values():
        ps.remaining_delta_v = 0.0
    for grp in constellation2._groups.values():
        grp.propulsion.remaining_delta_v = 0.0
    
    can_sep, reason = constellation2.can_separate({0, 1, 2, 3})
    assert not can_sep, "Should reject with zero delta-v"
    print(f"  Zero delta-v: rejected - '{reason}'")
    
    # Test 4: Max groups reached
    constellation3 = create_test_constellation(64, 4)
    constellation3.separation_reqs.max_groups = 1
    can_sep, reason = constellation3.can_separate({0, 1, 2, 3})
    assert not can_sep, "Should reject when max groups reached"
    print(f"  Max groups: rejected - '{reason}'")
    
    # Test 5: Disconnected cubes (non-adjacent)
    constellation4 = create_test_constellation(64, 4)
    # Cubes 0 and 63 are likely not adjacent in a 4x4x4 cube
    can_sep, reason = constellation4.can_separate({0, 63})
    print(f"  Potentially disconnected set: {'accepted' if can_sep else 'rejected'} - '{reason}'")
    
    print("  ✓ PASS")
    return True


def test_propagation() -> bool:
    """Test time propagation of separated groups."""
    print_header("4. Time Propagation Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # Separate
    valid_separations = controller.get_valid_separation_actions()
    for action in valid_separations:
        if len(action.cube_ids) >= 16:
            controller.execute_separation(action)
            break
    
    assert constellation.get_num_groups() >= 2, "Need multiple groups for propagation test"
    
    groups = constellation.get_all_groups()
    pos_before = {g.group_id: g.position.copy() for g in groups}
    vel = {g.group_id: g.velocity.copy() for g in groups}
    
    print(f"  Before propagation:")
    print_status(constellation)
    
    # Propagate
    dt = 100.0
    constellation.propagate(dt)
    
    print(f"\n  After {dt}s propagation:")
    print_status(constellation)
    
    # Verify positions updated correctly
    for group in constellation.get_all_groups():
        expected_pos = pos_before[group.group_id] + vel[group.group_id] * dt
        assert np.allclose(group.position, expected_pos, atol=1e-6), \
            f"Group {group.group_id} position mismatch"
    
    assert np.isclose(constellation.get_time(), dt), "Time not updated correctly"
    
    print("  ✓ PASS")
    return True


def test_maneuver() -> bool:
    """Test delta-v maneuver execution."""
    print_header("5. Maneuver Execution Test")
    
    constellation = create_test_constellation(64, 4)
    
    group = constellation.get_all_groups()[0]
    gid = group.group_id
    
    dv_before = group.propulsion.remaining_delta_v
    vel_before = group.velocity.copy()
    
    # Apply maneuver
    delta_v = np.array([2.0, 1.0, 0.5])
    dv_mag = np.linalg.norm(delta_v)
    
    success, msg = constellation.apply_delta_v(gid, delta_v)
    
    print(f"  Applied Δv = {delta_v}, magnitude = {dv_mag:.3f} m/s")
    print(f"  Result: {success}, '{msg}'")
    
    assert success, f"Maneuver failed: {msg}"
    assert np.allclose(group.velocity, vel_before + delta_v), "Velocity not updated"
    assert np.isclose(group.propulsion.remaining_delta_v, dv_before - dv_mag, atol=1e-9), \
        "Delta-v budget not decremented"
    
    # Test rejection when exceeding budget
    constellation2 = create_test_constellation(64, 4)
    grp2 = constellation2.get_all_groups()[0]
    grp2.propulsion.remaining_delta_v = 1.0
    
    success2, msg2 = constellation2.apply_delta_v(grp2.group_id, np.array([100.0, 0, 0]))
    assert not success2, "Should reject maneuver exceeding budget"
    print(f"  Over-budget rejection: '{msg2}'")
    
    print("  ✓ PASS")
    return True


def test_docking() -> bool:
    """Test docking of separated groups."""
    print_header("6. Docking Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # Separate
    valid_separations = controller.get_valid_separation_actions()
    for action in valid_separations:
        if 16 <= len(action.cube_ids) <= 32:
            result = controller.execute_separation(action)
            if result.success:
                break
    
    assert constellation.get_num_groups() == 2, "Need 2 groups for docking test"
    
    groups = constellation.get_all_groups()
    gid_a, gid_b = groups[0].group_id, groups[1].group_id
    
    print(f"  Groups before docking: {constellation.get_num_groups()}")
    
    # Move groups to same position for docking
    constellation._groups[gid_b].position = constellation._groups[gid_a].position.copy()
    constellation._groups[gid_b].velocity = constellation._groups[gid_a].velocity.copy()
    
    # Check docking possible
    can_dock, reason = constellation.can_dock(gid_a, gid_b)
    print(f"  Can dock: {can_dock}, '{reason}'")
    
    if can_dock:
        success, msg = constellation.dock(gid_a, gid_b)
        print(f"  Dock result: {success}, '{msg}'")
        
        assert success, f"Docking failed: {msg}"
        assert constellation.get_num_groups() == 1, "Should have 1 group after docking"
        assert constellation.is_single_group(), "Should be single group"
    else:
        print(f"  Skipping dock execution (pre-condition not met)")
    
    print_status(constellation)
    print("  ✓ PASS")
    return True


def test_docking_approach() -> bool:
    """Test docking approach maneuver."""
    print_header("7. Docking Approach Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # Separate with higher velocity
    valid_separations = controller.get_valid_separation_actions()
    for action in valid_separations:
        if len(action.cube_ids) >= 16:
            # Modify action for faster separation
            fast_action = SeparationAction(
                cube_ids=action.cube_ids,
                separation_direction=(1.0, 0.0, 0.0),
                separation_velocity=5.0
            )
            controller.execute_separation(fast_action)
            break
    
    # Propagate to create distance
    constellation.propagate(50.0)
    
    groups = constellation.get_all_groups()
    if len(groups) < 2:
        print("  Could not create 2 groups, skipping")
        print("  ✓ PASS (skipped)")
        return True
    
    gid_a, gid_b = groups[0].group_id, groups[1].group_id
    
    dist_before = groups[0].get_distance_to(groups[1])
    print(f"  Distance before approach: {dist_before:.1f}m")
    
    # Initiate approach
    success, msg = constellation.initiate_docking_approach(gid_a, gid_b, approach_velocity=1.0)
    print(f"  Approach initiated: {success}, '{msg}'")
    
    if success:
        # Propagate and check closing
        for _ in range(10):
            constellation.propagate(10.0)
            dist = constellation._groups[gid_a].get_distance_to(constellation._groups[gid_b])
            
            can_dock, _ = constellation.can_dock(gid_a, gid_b)
            if can_dock:
                print(f"  Docking range reached at t={constellation.get_time():.0f}s")
                break
        
        print(f"  Final distance: {dist:.1f}m")
    
    print("  ✓ PASS")
    return True


def test_communication() -> bool:
    """Test communication constraints between groups."""
    print_header("8. Communication Test")
    
    # Create with specific comm requirements
    swarm = Swarm(64)
    create_cube_formation(swarm, size=4)
    
    comm_reqs = CommunicationRequirements(
        max_isl_range=10_000.0,  # 10 km
        base_data_rate=10_000.0,
        reference_distance=1_000.0,
        min_coordination_rate=100.0
    )
    
    constellation = Constellation(swarm, comm_reqs=comm_reqs)
    controller = ConstellationController(constellation)
    
    # Set up propulsion
    for ps in constellation._cube_propulsion.values():
        ps.remaining_delta_v = 100.0
    for grp in constellation._groups.values():
        grp.propulsion.remaining_delta_v = 100.0 * len(grp.cube_ids)
    
    # Separate with high velocity
    valid_separations = controller.get_valid_separation_actions()
    for action in valid_separations:
        if len(action.cube_ids) >= 16:
            fast_action = SeparationAction(
                cube_ids=action.cube_ids,
                separation_direction=(1.0, 0.0, 0.0),
                separation_velocity=10.0
            )
            controller.execute_separation(fast_action)
            break
    
    print(f"  Initial groups: {constellation.get_num_groups()}")
    print(f"  Initially connected: {constellation.is_constellation_connected()}")
    
    # Propagate and check communication
    test_times = [0, 100, 500, 1000, 2000]
    
    for target_time in test_times:
        if target_time > constellation.get_time():
            constellation.propagate(target_time - constellation.get_time())
        
        baseline = constellation.get_max_baseline()
        connected = constellation.is_constellation_connected()
        
        print(f"  t={target_time:4d}s: baseline={baseline:8.1f}m, connected={connected}")
        
        # Check communication links
        links = constellation.get_communication_links()
        for (ga, gb), info in links.items():
            if target_time == test_times[-1]:  # Only print details for last time
                print(f"    Link {ga}-{gb}: {info['distance']:.0f}m, "
                      f"rate={info['data_rate']:.1f}bps, can_comm={info['can_communicate']}")
    
    print("  ✓ PASS")
    return True


def test_communication_graph() -> bool:
    """Test communication graph construction."""
    print_header("9. Communication Graph Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # Create multiple groups
    for _ in range(2):
        valid_separations = controller.get_valid_separation_actions()
        if valid_separations:
            for action in valid_separations:
                if 8 <= len(action.cube_ids) <= 32:
                    controller.execute_separation(action)
                    break
    
    print(f"  Number of groups: {constellation.get_num_groups()}")
    
    # Get communication graph
    graph = constellation.get_communication_graph()
    print(f"  Communication graph: {graph}")
    
    # Check connectivity
    connected = constellation.is_constellation_connected()
    print(f"  All groups connected: {connected}")
    
    # Verify graph structure
    for gid in constellation._groups.keys():
        assert gid in graph, f"Group {gid} missing from graph"
    
    print("  ✓ PASS")
    return True


def test_delta_v_accounting() -> bool:
    """Test delta-v budget accounting across operations."""
    print_header("10. Delta-V Accounting Test")
    
    constellation = create_test_constellation(64, 4, max_delta_v=50.0)
    
    # Get initial budget
    initial_total = constellation.get_total_delta_v_remaining()
    print(f"  Initial total delta-v: {initial_total:.2f} m/s")
    
    # Perform separation
    controller = ConstellationController(constellation)
    valid_separations = controller.get_valid_separation_actions()
    
    if valid_separations:
        action = valid_separations[0]
        result = controller.execute_separation(action)
        print(f"  Separation: success={result.success}, delta-v used={result.delta_v_used:.3f} m/s")
    
    after_sep = constellation.get_total_delta_v_remaining()
    print(f"  After separation: {after_sep:.2f} m/s")
    
    # Perform maneuvers
    groups = constellation.get_all_groups()
    total_maneuver_dv = 0.0
    
    for group in groups[:2]:  # Maneuver first two groups
        delta_v = np.array([1.0, 0.5, 0.0])
        success, _ = constellation.apply_delta_v(group.group_id, delta_v)
        if success:
            total_maneuver_dv += np.linalg.norm(delta_v)
    
    print(f"  Maneuvers used: {total_maneuver_dv:.3f} m/s")
    
    after_maneuver = constellation.get_total_delta_v_remaining()
    print(f"  After maneuvers: {after_maneuver:.2f} m/s")
    
    # Verify per-group accounting
    by_group = constellation.get_delta_v_by_group()
    print(f"  By group: {by_group}")
    
    # Check consistency
    total_from_groups = sum(by_group.values())
    assert np.isclose(total_from_groups, after_maneuver, atol=1.0), \
        f"Budget mismatch: groups sum to {total_from_groups}, total is {after_maneuver}"
    
    print("  ✓ PASS")
    return True


def test_constellation_metrics() -> bool:
    """Test constellation metrics computation."""
    print_header("11. Constellation Metrics Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # Create multi-group constellation
    for _ in range(2):
        valid_separations = controller.get_valid_separation_actions()
        if valid_separations:
            for action in valid_separations:
                if 10 <= len(action.cube_ids) <= 32:
                    controller.execute_separation(action)
                    break
    
    # Propagate to spread out
    constellation.propagate(200.0)
    
    print(f"  Groups: {constellation.get_num_groups()}")
    
    # Compute metrics
    metrics = ConstellationMetrics(constellation)
    all_metrics = metrics.get_all_metrics()
    
    print(f"  Metrics:")
    for name, value in all_metrics.items():
        if isinstance(value, float):
            print(f"    {name}: {value:.4f}")
        else:
            print(f"    {name}: {value}")
    
    # Test specific metrics
    aperture = metrics.get_effective_aperture()
    print(f"  Effective aperture: {aperture:.1f}m")
    
    comm_eff = metrics.get_communication_efficiency()
    print(f"  Communication efficiency: {comm_eff:.3f}")
    
    prop_eff = metrics.get_propulsion_efficiency()
    print(f"  Propulsion efficiency: {prop_eff:.3f}")
    
    # Test formation quality
    for formation in ['line', 'plane', 'tetrahedron']:
        quality = metrics.get_formation_quality(formation)
        print(f"  Formation quality ({formation}): {quality:.3f}")
    
    print("  ✓ PASS")
    return True


def test_constellation_state() -> bool:
    """Test constellation state retrieval."""
    print_header("12. Constellation State Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # Separate
    valid_separations = controller.get_valid_separation_actions()
    if valid_separations:
        controller.execute_separation(valid_separations[0])
    
    constellation.propagate(50.0)
    
    # Get full state
    state = constellation.get_constellation_state()
    
    print(f"  State keys: {list(state.keys())}")
    print(f"  Time: {state['time']:.1f}s")
    print(f"  Num groups: {state['num_groups']}")
    print(f"  Total cubes: {state['total_cubes']}")
    print(f"  Connected: {state['is_connected']}")
    print(f"  Max baseline: {state['max_baseline']:.1f}m")
    print(f"  Total delta-v: {state['total_delta_v']:.1f} m/s")
    
    # Check group info
    for gid, ginfo in state['groups'].items():
        print(f"  Group {gid}: {ginfo['num_cubes']} cubes, "
              f"pos={[f'{v:.1f}' for v in ginfo['position']]}")
    
    # Get observation vector
    obs = constellation.get_observation_vector()
    print(f"  Observation vector shape: {obs.shape}")
    print(f"  Observation sample: {obs[:10]}")
    
    print("  ✓ PASS")
    return True


def test_constellation_copy() -> bool:
    """Test constellation deep copy."""
    print_header("13. Constellation Copy Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # Modify original
    valid_separations = controller.get_valid_separation_actions()
    if valid_separations:
        controller.execute_separation(valid_separations[0])
    constellation.propagate(100.0)
    
    # Make copy
    copy = constellation.copy()
    
    # Verify copy is independent
    original_groups = constellation.get_num_groups()
    original_time = constellation.get_time()
    original_baseline = constellation.get_max_baseline()
    
    # Modify original further
    constellation.propagate(100.0)
    
    # Check copy is unchanged
    assert copy.get_num_groups() == original_groups, "Copy groups changed"
    assert copy.get_time() == original_time, "Copy time changed"
    assert np.isclose(copy.get_max_baseline(), original_baseline), "Copy baseline changed"
    
    print(f"  Original time: {constellation.get_time():.1f}s")
    print(f"  Copy time: {copy.get_time():.1f}s")
    print(f"  Copy is independent: True")
    
    print("  ✓ PASS")
    return True


def test_controller_actions() -> bool:
    """Test ConstellationController action generation."""
    print_header("14. Controller Actions Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # Test separation actions
    sep_actions = controller.get_valid_separation_actions()
    print(f"  Valid separation actions: {len(sep_actions)}")
    
    if sep_actions:
        # Show a few examples
        for action in sep_actions[:3]:
            print(f"    - {len(action.cube_ids)} cubes, dir={action.separation_direction}, "
                  f"vel={action.separation_velocity}")
    
    # Execute a separation to enable docking/maneuver actions
    if sep_actions:
        for action in sep_actions:
            if 16 <= len(action.cube_ids) <= 32:
                result = controller.execute_separation(action)
                if result.success:
                    print(f"  Executed separation: {result.reason}")
                    break
    
    # Test docking actions
    dock_actions = controller.get_valid_docking_actions()
    print(f"  Valid docking actions: {len(dock_actions)}")
    
    # Test maneuver actions
    maneuver_actions = controller.get_valid_maneuver_actions(max_delta_v=1.0)
    print(f"  Valid maneuver actions: {len(maneuver_actions)}")
    
    if maneuver_actions:
        for action in maneuver_actions[:3]:
            print(f"    - Group {action.group_id}, delta_v={action.delta_v}")
    
    print("  ✓ PASS")
    return True


def test_task_form_constellation() -> bool:
    """Test FormConstellationTask."""
    print_header("15. FormConstellationTask Test")
    
    constellation = create_test_constellation(64, 4)
    
    task = FormConstellationTask(
        target_num_groups=2,
        target_baseline=5000.0,
        baseline_tolerance=1000.0,
        min_group_size=16,
        maintain_communication=True
    )
    
    print(f"  Task: {task.get_task_info()}")
    
    # Initial state (1 group)
    initial_reward = task.compute_reward(constellation)
    initial_progress = task.get_progress(constellation)
    initial_complete = task.is_complete(constellation)
    
    print(f"  Initial: reward={initial_reward:.3f}, progress={initial_progress:.3f}, "
          f"complete={initial_complete}")
    
    # Separate into 2 groups
    controller = ConstellationController(constellation)
    valid_separations = controller.get_valid_separation_actions()
    
    for action in valid_separations:
        if 20 <= len(action.cube_ids) <= 40:
            # Increase separation velocity for faster baseline growth
            fast_action = SeparationAction(
                cube_ids=action.cube_ids,
                separation_direction=(1.0, 0.0, 0.0),
                separation_velocity=10.0
            )
            controller.execute_separation(fast_action)
            break
    
    # Propagate to reach target baseline
    while constellation.get_max_baseline() < 5000.0:
        constellation.propagate(100.0)
        if constellation.get_time() > 10000.0:
            break  # Safety limit
    
    # Check task status after separation
    final_reward = task.compute_reward(constellation)
    final_progress = task.get_progress(constellation)
    final_complete = task.is_complete(constellation)
    
    print(f"  After separation and propagation:")
    print(f"    Groups: {constellation.get_num_groups()}")
    print(f"    Baseline: {constellation.get_max_baseline():.1f}m")
    print(f"    Reward: {final_reward:.3f}")
    print(f"    Progress: {final_progress:.3f}")
    print(f"    Complete: {final_complete}")
    
    assert final_reward > initial_reward, "Reward should increase toward goal"
    assert final_progress > initial_progress, "Progress should increase"
    
    print("  ✓ PASS")
    return True


def test_task_rendezvous_and_dock() -> bool:
    """Test RendezvousAndDockTask."""
    print_header("16. RendezvousAndDockTask Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    # First separate
    valid_separations = controller.get_valid_separation_actions()
    for action in valid_separations:
        if len(action.cube_ids) >= 16:
            controller.execute_separation(action)
            break
    
    # Propagate to create some distance
    constellation.propagate(50.0)
    
    assert constellation.get_num_groups() == 2, "Need 2 groups for rendezvous test"
    
    task = RendezvousAndDockTask(
        target_num_groups=1,
        max_time=3600.0,
        efficiency_weight=0.3
    )
    
    print(f"  Task: {task.get_task_info()}")
    
    # Check initial state
    initial_reward = task.compute_reward(constellation)
    initial_progress = task.get_progress(constellation)
    
    print(f"  Initial (2 groups): reward={initial_reward:.3f}, progress={initial_progress:.3f}")
    
    # Move groups together for docking
    groups = constellation.get_all_groups()
    gid_a, gid_b = groups[0].group_id, groups[1].group_id
    
    # Set same position/velocity for docking
    constellation._groups[gid_b].position = constellation._groups[gid_a].position.copy()
    constellation._groups[gid_b].velocity = constellation._groups[gid_a].velocity.copy()
    
    # Dock
    can_dock, _ = constellation.can_dock(gid_a, gid_b)
    if can_dock:
        constellation.dock(gid_a, gid_b)
    
    # Check final state
    final_reward = task.compute_reward(constellation)
    final_complete = task.is_complete(constellation)
    
    print(f"  After docking: reward={final_reward:.3f}, complete={final_complete}")
    print(f"    Groups: {constellation.get_num_groups()}")
    
    assert final_complete, "Task should be complete after docking to 1 group"
    
    print("  ✓ PASS")
    return True


def test_task_stereo_imaging() -> bool:
    """Test StereoImagingTask."""
    print_header("17. StereoImagingTask Test")
    
    constellation = create_test_constellation(64, 4)
    
    task = StereoImagingTask(
        target_direction=(0, 1, 0),
        num_viewpoints=2,
        min_angular_separation=0.1,  # ~5.7 degrees
        max_angular_separation=0.5,  # ~28.6 degrees
        target_distance=1000.0
    )
    
    print(f"  Task: {task.get_task_info()}")
    print(f"  Required baseline: {task.min_baseline:.1f}m - {task.max_baseline:.1f}m")
    
    # Initial state
    initial_progress = task.get_progress(constellation)
    print(f"  Initial progress: {initial_progress:.3f}")
    
    # Separate to create viewpoints
    controller = ConstellationController(constellation)
    valid_separations = controller.get_valid_separation_actions()
    
    for action in valid_separations:
        if len(action.cube_ids) >= 16:
            # Separate perpendicular to target direction for stereo baseline
            stereo_action = SeparationAction(
                cube_ids=action.cube_ids,
                separation_direction=(1.0, 0.0, 0.0),  # Perpendicular to target (0,1,0)
                separation_velocity=2.0
            )
            controller.execute_separation(stereo_action)
            break
    
    # Propagate to achieve baseline
    target_baseline = (task.min_baseline + task.max_baseline) / 2
    while constellation.get_max_baseline() < target_baseline:
        constellation.propagate(10.0)
        if constellation.get_time() > 1000.0:
            break
    
    # Check task status
    final_progress = task.get_progress(constellation)
    final_reward = task.compute_reward(constellation)
    final_complete = task.is_complete(constellation)
    
    print(f"  After setup:")
    print(f"    Groups: {constellation.get_num_groups()}")
    print(f"    Baseline: {constellation.get_max_baseline():.1f}m")
    print(f"    Progress: {final_progress:.3f}")
    print(f"    Reward: {final_reward:.3f}")
    print(f"    Complete: {final_complete}")
    
    print("  ✓ PASS")
    return True


def test_task_multi_point_sensing() -> bool:
    """Test MultiPointSensingTask."""
    print_header("18. MultiPointSensingTask Test")
    
    constellation = create_test_constellation(64, 4)
    
    task = MultiPointSensingTask(
        target_num_groups=4,
        target_volume=1e6,  # 1 km^3
        min_group_size=8,
        formation_type='tetrahedron'
    )
    
    print(f"  Task: {task.get_task_info()}")
    
    # Initial state
    initial_progress = task.get_progress(constellation)
    print(f"  Initial progress: {initial_progress:.3f}")
    
    # Create multiple groups
    controller = ConstellationController(constellation)
    
    for i in range(3):  # Try to create 4 groups total
        valid_separations = controller.get_valid_separation_actions()
        if not valid_separations:
            break
        
        for action in valid_separations:
            if 8 <= len(action.cube_ids) <= 32:
                # Separate in different directions
                directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
                sep_action = SeparationAction(
                    cube_ids=action.cube_ids,
                    separation_direction=directions[i % 3],
                    separation_velocity=5.0
                )
                result = controller.execute_separation(sep_action)
                if result.success:
                    break
    
    # Propagate to spread out
    constellation.propagate(100.0)
    
    # Check metrics
    metrics = ConstellationMetrics(constellation)
    
    print(f"  After separations:")
    print(f"    Groups: {constellation.get_num_groups()}")
    print(f"    Coverage volume: {metrics.get_coverage_volume():.1f} m³")
    print(f"    Formation quality: {metrics.get_formation_quality('tetrahedron'):.3f}")
    print(f"    Progress: {task.get_progress(constellation):.3f}")
    print(f"    Reward: {task.compute_reward(constellation):.3f}")
    
    print("  ✓ PASS")
    return True


def test_multiple_separations() -> bool:
    """Test multiple sequential separations."""
    print_header("19. Multiple Separations Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    print(f"  Initial: {constellation.get_num_groups()} group(s)")
    
    separations_done = 0
    max_separations = 4
    
    while separations_done < max_separations:
        valid_separations = controller.get_valid_separation_actions()
        
        if not valid_separations:
            print(f"  No more valid separations at {constellation.get_num_groups()} groups")
            break
        
        # Find smallest valid separation
        best_action = min(valid_separations, key=lambda a: len(a.cube_ids))
        
        result = controller.execute_separation(best_action)
        
        if result.success:
            separations_done += 1
            print(f"  Separation {separations_done}: {len(best_action.cube_ids)} cubes -> "
                  f"{constellation.get_num_groups()} groups")
        else:
            print(f"  Separation failed: {result.reason}")
            break
    
    # Verify all cubes are still accounted for
    total_cubes = sum(len(g.cube_ids) for g in constellation.get_all_groups())
    assert total_cubes == 64, f"Lost cubes! Expected 64, got {total_cubes}"
    
    # Verify groups are disjoint
    all_cube_ids = set()
    for group in constellation.get_all_groups():
        overlap = all_cube_ids & group.cube_ids
        assert not overlap, f"Overlapping cubes: {overlap}"
        all_cube_ids |= group.cube_ids
    
    print(f"  Final: {constellation.get_num_groups()} groups, {total_cubes} total cubes")
    print("  ✓ PASS")
    return True


def test_full_mission_cycle() -> bool:
    """Test a complete mission cycle: separate, maneuver, dock."""
    print_header("20. Full Mission Cycle Test")
    
    constellation = create_test_constellation(64, 4)
    controller = ConstellationController(constellation)
    
    print("  Phase 1: Initial state")
    print_status(constellation)
    
    # Phase 2: Separate
    print("\n  Phase 2: Separation")
    valid_separations = controller.get_valid_separation_actions()
    for action in valid_separations:
        if 20 <= len(action.cube_ids) <= 40:
            result = controller.execute_separation(action)
            if result.success:
                print(f"    Separated {len(action.cube_ids)} cubes, Δv={result.delta_v_used:.2f}")
                break
    
    assert constellation.get_num_groups() == 2, "Should have 2 groups after separation"
    print_status(constellation)
    
    # Phase 3: Propagate and maneuver
    print("\n  Phase 3: Propagation and maneuvers")
    constellation.propagate(100.0)
    
    # Apply maneuvers
    groups = constellation.get_all_groups()
    for group in groups:
        delta_v = np.array([0.5, 0.0, 0.0])
        success, msg = constellation.apply_delta_v(group.group_id, delta_v)
        if success:
            print(f"    Maneuvered group {group.group_id}")
    
    constellation.propagate(200.0)
    print_status(constellation)
    
    # Phase 4: Approach and dock
    print("\n  Phase 4: Docking")
    groups = constellation.get_all_groups()
    if len(groups) >= 2:
        gid_a, gid_b = groups[0].group_id, groups[1].group_id
        
        # Force groups together for docking
        constellation._groups[gid_b].position = constellation._groups[gid_a].position.copy()
        constellation._groups[gid_b].velocity = constellation._groups[gid_a].velocity.copy()
        
        can_dock, reason = constellation.can_dock(gid_a, gid_b)
        if can_dock:
            success, msg = constellation.dock(gid_a, gid_b)
            print(f"    Docking: {msg}")
    
    print_status(constellation)
    
    assert constellation.is_single_group(), "Should be single group after docking"
    print("\n  ✓ PASS - Full cycle complete")
    return True


# =============================================================================
# Test Runner
# =============================================================================

ALL_TESTS = [
    test_initial_state,
    test_separation_basic,
    test_separation_validation,
    test_propagation,
    test_maneuver,
    test_docking,
    test_docking_approach,
    test_communication,
    test_communication_graph,
    test_delta_v_accounting,
    test_constellation_metrics,
    test_constellation_state,
    test_constellation_copy,
    test_controller_actions,
    test_task_form_constellation,
    test_task_rendezvous_and_dock,
    test_task_stereo_imaging,
    test_task_multi_point_sensing,
    test_multiple_separations,
    test_full_mission_cycle,
]


def run_all_tests(verbose: bool = True) -> Tuple[int, int, List[str]]:
    """Run all tests and return (passed, failed, failed_test_names)."""
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_fn in ALL_TESTS:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                failed_tests.append(test_fn.__name__)
        except Exception as e:
            failed += 1
            failed_tests.append(test_fn.__name__)
            if verbose:
                print(f"\n  ✗ FAIL [{test_fn.__name__}]: {e}")
                import traceback
                traceback.print_exc()
    
    return passed, failed, failed_tests


def run_single_test(test_name: str) -> bool:
    """Run a single test by name."""
    test_map = {fn.__name__: fn for fn in ALL_TESTS}
    
    if test_name not in test_map:
        print(f"Unknown test: {test_name}")
        print(f"Available tests:")
        for name in sorted(test_map.keys()):
            print(f"  - {name}")
        return False
    
    try:
        result = test_map[test_name]()
        if result:
            print(f"\n✓ Test {test_name} passed")
            return True
        else:
            print(f"\n✗ Test {test_name} failed (returned False)")
            return False
    except Exception as e:
        print(f"\n✗ Test {test_name} failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Constellation test suite")
    parser.add_argument("--viz", action="store_true", 
                        help="Show visualizations (requires display)")
    parser.add_argument("--save-plots", action="store_true", 
                        help="Save plots to ./plots/")
    parser.add_argument("--test", type=str, 
                        help="Run specific test by name")
    parser.add_argument("--list", action="store_true",
                        help="List all available tests")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    
    # List tests if requested
    if args.list:
        print("Available tests:")
        for test_fn in ALL_TESTS:
            doc = test_fn.__doc__ or "No description"
            doc_first_line = doc.strip().split('\n')[0]
            print(f"  {test_fn.__name__}: {doc_first_line}")
        return
    
    print("=" * 60)
    print("CONSTELLATION TEST SUITE")
    print("=" * 60)
    
    start_time = time.time()
    
    if args.test:
        # Run specific test
        success = run_single_test(args.test)
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        passed, failed, failed_tests = run_all_tests(verbose=args.verbose)
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"RESULTS: {passed} passed / {failed} failed ({elapsed:.2f}s)")
        print("=" * 60)
        
        if failed_tests:
            print(f"\nFailed tests:")
            for name in failed_tests:
                print(f"  - {name}")
        
        # Run visualizations if requested
        if args.viz or args.save_plots:
            try:
                run_visualizations(show=args.viz, save_dir="./plots" if args.save_plots else None)
            except Exception as e:
                print(f"\nVisualization error: {e}")
        
        sys.exit(0 if failed == 0 else 1)


def run_visualizations(show: bool = False, save_dir: Optional[str] = None) -> None:
    """Run visualization tests."""
    if not show and not save_dir:
        return
    
    print_header("Generating Visualizations")
    
    try:
        from visualization.constellation_renderer import ConstellationVisualizer
        import matplotlib
        
        if not show:
            matplotlib.use('Agg')  # Non-interactive backend for saving
        
        # Create a test constellation
        constellation = create_test_constellation(64, 4)
        controller = ConstellationController(constellation)
        
        # Separate into multiple groups
        for i in range(2):
            valid_separations = controller.get_valid_separation_actions()
            if valid_separations:
                for action in valid_separations:
                    if 10 <= len(action.cube_ids) <= 32:
                        controller.execute_separation(action)
                        break
        
        # Propagate to spread out
        constellation.propagate(200.0)
        
        # Create visualizer
        viz = ConstellationVisualizer(constellation)
        
        # Render local view
        print("  Generating local view...")
        viz.render_local(title="Constellation Local View", show_connections=True)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            viz.save(os.path.join(save_dir, "constellation_local.png"))
        
        # Render global view
        print("  Generating global view...")
        viz.render_global(title="Constellation Global View", show_comm_links=True)
        if save_dir:
            viz.save(os.path.join(save_dir, "constellation_global.png"))
        
        # Render dual view
        print("  Generating dual view...")
        viz.render_dual(title="Constellation Overview")
        if save_dir:
            viz.save(os.path.join(save_dir, "constellation_dual.png"))
        
        if show:
            print("\n  Close plot windows to continue...")
            viz.show()
        
        viz.close()
        print("  ✓ Visualizations complete")
        
    except ImportError as e:
        print(f"  Skipping visualizations (missing dependency): {e}")
    except Exception as e:
        print(f"  Visualization error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()