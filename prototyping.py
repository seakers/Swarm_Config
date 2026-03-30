from configs.formations import *
from core.connections import *
from core.cube import *
from core.grid import *
from core.swarm import *
from env.env import *
from env.multi_task_env import *
from mechanics.moves import *
from rewards.metrics import *
from tasks.tasks import *
from utils.utils import *
from visualization.renderer import *


# =============================================================================
# Test functions for face functions
# =============================================================================

def test_face_functions():
    """Test the face function system."""
    print("=" * 60)
    print("Testing Face Functions")
    print("=" * 60)
    
    # Create swarm
    swarm = Swarm(64)
    create_cube_formation(swarm, size=4)
    
    print(f"\nCreated 4x4x4 cube swarm")
    
    # Create analyzer
    analyzer = SwarmFaceAnalyzer(swarm)
    
    # Check exposure fractions for all functions
    print("\nFace function exposure fractions:")
    exposures = analyzer.get_all_exposure_fractions()
    for func, frac in exposures.items():
        print(f"  {func.name}: {frac:.2%}")
    
    # Test specific function exposure
    print("\nCubes with SOLAR_ARRAY exposed:")
    solar_cubes = analyzer.get_cubes_with_function_exposed(FaceFunction.SOLAR_ARRAY)
    print(f"  {len(solar_cubes)} cubes")
    
    # Test alignment analysis
    sun_direction = (0, 0, 1)  # Sun above
    print(f"\nSolar array alignment with sun at {sun_direction}:")
    solar_alignment = analyzer.compute_function_alignment(
        FaceFunction.SOLAR_ARRAY, sun_direction, only_exposed=True
    )
    for key, val in solar_alignment.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
    
    # Test solar efficiency
    print(f"\nSolar array efficiency:")
    efficiency = analyzer.compute_solar_array_efficiency(sun_direction)
    print(f"  Efficiency: {efficiency:.2%}")
    
    # Test antenna effectiveness
    earth_direction = (1, 0, 0)  # Earth to the right
    print(f"\nAntenna effectiveness (Earth at {earth_direction}):")
    antenna = analyzer.compute_antenna_effectiveness(earth_direction)
    for key, val in antenna.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
    
    # Test camera coverage
    target_direction = (0, 1, 0)  # Target in front
    print(f"\nCamera coverage (target at {target_direction}):")
    camera = analyzer.compute_camera_coverage(target_direction)
    for key, val in camera.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
    
    # Test thermal balance
    print(f"\nThermal balance:")
    thermal = analyzer.compute_thermal_balance(sun_direction)
    for key, val in thermal.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
    
    # Test power balance
    print(f"\nPower balance (10 AU from sun):")
    power = analyzer.compute_power_balance(sun_direction, sun_distance_au=10.0)
    for key, val in power.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
    
    return swarm, analyzer


def test_mission_mode_scoring():
    """Test mission mode scoring for different configurations."""
    print("\n" + "=" * 60)
    print("Testing Mission Mode Scoring")
    print("=" * 60)
    
    # Mission context
    sun_direction = (0, 0, -1)  # Sun below
    earth_direction = (1, 0, 0)  # Earth to the right
    target_direction = (0, 1, 0)  # Target in front
    sun_distance_au = 10.0  # At Saturn-ish distance
    
    # Test 1: Cube formation
    print("\n--- 4x4x4 Cube Formation ---")
    swarm_cube = Swarm(64)
    create_cube_formation(swarm_cube, size=4)
    scorer_cube = MissionModeScorer(swarm_cube)
    
    scores_cube = scorer_cube.get_all_mode_scores(
        sun_direction, earth_direction, target_direction, sun_distance_au
    )
    print("Mode scores:")
    for mode, score in scores_cube.items():
        if isinstance(score, float):
            print(f"  {mode}: {score:.3f}")
        else:
            print(f"  {mode}: {score}")
    
    # Test 2: Flat plane formation (good for communication)
    print("\n--- 8x8 Flat Plane Formation (Z-normal) ---")
    swarm_plane = Swarm(64)
    create_plane_formation(swarm_plane, width=8, height=8, normal_axis=2)
    scorer_plane = MissionModeScorer(swarm_plane)
    
    scores_plane = scorer_plane.get_all_mode_scores(
        sun_direction, earth_direction, target_direction, sun_distance_au
    )
    print("Mode scores:")
    for mode, score in scores_plane.items():
        if isinstance(score, float):
            print(f"  {mode}: {score:.3f}")
        else:
            print(f"  {mode}: {score}")
    
    # Test 3: Line formation (good for sparse aperture)
    print("\n--- 64-unit Line Formation (X-axis) ---")
    swarm_line = Swarm(64)
    create_line_formation(swarm_line, length=64, axis=0)
    scorer_line = MissionModeScorer(swarm_line)
    
    scores_line = scorer_line.get_all_mode_scores(
        sun_direction, earth_direction, target_direction, sun_distance_au
    )
    print("Mode scores:")
    for mode, score in scores_line.items():
        if isinstance(score, float):
            print(f"  {mode}: {score:.3f}")
        else:
            print(f"  {mode}: {score}")
    
    # Detailed breakdown for communication mode
    print("\n--- Detailed Communication Mode Analysis ---")
    print("\nCube formation:")
    comm_cube = scorer_cube.score_earth_communication_mode(
        earth_direction, sun_direction, sun_distance_au
    )
    print(f"  Total score: {comm_cube['total_score']:.3f}")
    print(f"  Antenna score: {comm_cube['antenna_score']:.3f}")
    print(f"  Power score: {comm_cube['power_score']:.3f}")
    print(f"  Thermal score: {comm_cube['thermal_score']:.3f}")
    
    print("\nPlane formation:")
    comm_plane = scorer_plane.score_earth_communication_mode(
        earth_direction, sun_direction, sun_distance_au
    )
    print(f"  Total score: {comm_plane['total_score']:.3f}")
    print(f"  Antenna score: {comm_plane['antenna_score']:.3f}")
    print(f"  Power score: {comm_plane['power_score']:.3f}")
    print(f"  Thermal score: {comm_plane['thermal_score']:.3f}")
    
    return swarm_cube, swarm_plane, swarm_line


def test_mission_mode_tasks():
    """Test the mission mode task classes."""
    print("\n" + "=" * 60)
    print("Testing Mission Mode Tasks")
    print("=" * 60)
    
    # Create swarm in cube formation
    swarm = Swarm(64)
    create_cube_formation(swarm, size=4)
    
    sun_direction = (0, 0, -1)
    earth_direction = (1, 0, 0)
    target_direction = (0, 1, 0)
    
    # Test single-mode task
    print("\n--- Single Mode Task (Communication) ---")
    comm_task = MissionModeTask(
        mode='communication',
        sun_direction=sun_direction,
        earth_direction=earth_direction,
        target_direction=target_direction,
        sun_distance_au=10.0,
        target_score=0.7
    )
    
    print(f"Task info: {comm_task.get_task_info()['mode']}")
    print(f"Reward: {comm_task.compute_reward(swarm):.3f}")
    print(f"Progress: {comm_task.get_progress(swarm):.3f}")
    print(f"Complete: {comm_task.is_complete(swarm)}")
    
    # Test multi-objective task
    print("\n--- Multi-Objective Task (Observation + Communication) ---")
    multi_task = MultiObjectiveMissionTask(
        objectives={
            'observation': 0.6,
            'communication': 0.4,
        },
        sun_direction=sun_direction,
        earth_direction=earth_direction,
        target_direction=target_direction,
        sun_distance_au=10.0,
        target_score=0.6
    )
    
    print(f"Objectives: {multi_task.objectives}")
    print(f"Reward: {multi_task.compute_reward(swarm):.3f}")
    print(f"Progress: {multi_task.get_progress(swarm):.3f}")
    print(f"Complete: {multi_task.is_complete(swarm)}")


if __name__ == "__main__":
    # Run tests
    swarm, analyzer = test_face_functions()
    swarm_cube, swarm_plane, swarm_line = test_mission_mode_scoring()
    test_mission_mode_tasks()


# # Create swarm
# swarm_cube_length = 4
# swarm = Swarm(num_cubes=swarm_cube_length**3)
# print(f"Created swarm: {swarm}")

# # Form into cube
# create_cube_formation(swarm, size=swarm_cube_length)
# print(f"After cube formation: {swarm}")
# print(f"  Bounds: {swarm.get_bounds()}")
# print(f"  Center of mass: {swarm.get_center_of_mass()}")
# print(f"  Surface area: {swarm.get_surface_area()}")
# print(f"  Connected: {swarm.is_connected()}")

# num_test_moves = 5
# print(f"\nTesting {num_test_moves} random moves:")
# viz = SwarmVisualizer(swarm)
# viz.render(title="Initial Formation", show_connections=True, show_ids=True)

# # Test movement system
# movement = MovementSystem(swarm, require_connectivity=False)
# valid_moves = movement.get_all_valid_moves()
# print(f"\nValid moves from cube formation: {len(valid_moves)}")

# # Execute a few moves
# print("\nExecuting some moves:")
# for i in range(num_test_moves):
#     if valid_moves:
#         np.random.shuffle(valid_moves)
#         move = valid_moves[0]
#         result = movement.execute_move(move)
#         print(f"  Move {i+1}: Cube {move.cube_id}, Edge {move.pivot_edge.name}, "
#                 f"Dir {move.direction} -> {result.success} ({result.reason if not result.success else 'OK'})")
#         if result.success:
#             viz.render(title=f"After Move {i+1}", show_connections=True, show_ids=True)
#         valid_moves = movement.get_all_valid_moves()

# print(f"\nAfter moves: {swarm}")
# print(f"  Still connected: {swarm.is_connected()}")

# viz.show()
# viz.close()