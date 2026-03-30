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


def test_basic_simulation():
    """Test basic swarm creation and manipulation."""
    print("=" * 60)
    print("Testing Basic Simulation")
    print("=" * 60)
    
    # Create swarm
    swarm = Swarm(num_cubes=64)
    print(f"Created swarm: {swarm}")
    
    # Form into cube
    create_cube_formation(swarm, size=4)
    print(f"After cube formation: {swarm}")
    print(f"  Bounds: {swarm.get_bounds()}")
    print(f"  Center of mass: {swarm.get_center_of_mass()}")
    print(f"  Surface area: {swarm.get_surface_area()}")
    print(f"  Connected: {swarm.is_connected()}")
    
    # Test metrics
    metrics = SwarmMetrics(swarm)
    print(f"\nMetrics:")
    for name, value in metrics.get_all_metrics().items():
        print(f"  {name}: {value:.4f}")
    
    # Test movement system
    movement = MovementSystem(swarm, require_connectivity=False)
    valid_moves = movement.get_all_valid_moves()
    print(f"\nValid moves from cube formation: {len(valid_moves)}")
    
    # Execute a few moves
    print("\nExecuting some moves:")
    for i in range(5):
        if valid_moves:
            move = valid_moves[0]
            result = movement.execute_move(move)
            print(f"  Move {i+1}: Cube {move.cube_id}, Edge {move.pivot_edge.name}, "
                  f"Dir {move.direction} -> {result.success} ({result.reason if not result.success else 'OK'})")
            valid_moves = movement.get_all_valid_moves()
    
    print(f"\nAfter moves: {swarm}")
    print(f"  Still connected: {swarm.is_connected()}")
    
    return swarm


def test_formations():
    """Test different formation types."""
    print("\n" + "=" * 60)
    print("Testing Formations")
    print("=" * 60)
    
    # Cube formation
    swarm_cube = Swarm(64)
    create_cube_formation(swarm_cube, size=4)
    metrics_cube = SwarmMetrics(swarm_cube)
    print(f"\n4x4x4 Cube:")
    print(f"  Bounds: {swarm_cube.get_bounds()}")
    print(f"  Surface area: {swarm_cube.get_surface_area()}")
    print(f"  Compactness: {metrics_cube.compactness():.4f}")
    print(f"  Max baseline: {metrics_cube.maximum_baseline():.4f}")
    
    # Plane formation
    swarm_plane = Swarm(64)
    create_plane_formation(swarm_plane, width=8, height=8, normal_axis=2)
    metrics_plane = SwarmMetrics(swarm_plane)
    print(f"\n8x8 Plane (normal to Z):")
    print(f"  Bounds: {swarm_plane.get_bounds()}")
    print(f"  Surface area: {swarm_plane.get_surface_area()}")
    print(f"  Compactness: {metrics_plane.compactness():.4f}")
    print(f"  Planarity (Z): {metrics_plane.planarity((0, 0, 1)):.4f}")
    print(f"  Planar area (Z): {metrics_plane.planar_coverage((0, 0, 1)):.4f}")
    
    # Line formation
    swarm_line = Swarm(64)
    create_line_formation(swarm_line, length=64, axis=0)
    metrics_line = SwarmMetrics(swarm_line)
    print(f"\n64-unit Line (along X):")
    print(f"  Bounds: {swarm_line.get_bounds()}")
    print(f"  Surface area: {swarm_line.get_surface_area()}")
    print(f"  Compactness: {metrics_line.compactness():.4f}")
    print(f"  Linearity (X): {metrics_line.linearity((1, 0, 0)):.4f}")
    print(f"  Max baseline: {metrics_line.maximum_baseline():.4f}")
    
    return swarm_cube, swarm_plane, swarm_line


def test_movement_mechanics():
    """Test the hinge movement mechanics in detail."""
    print("\n" + "=" * 60)
    print("Testing Movement Mechanics")
    print("=" * 60)
    
    # Create a small test case: 2x2x1 flat arrangement
    swarm = Swarm(4)
    positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    create_custom_formation(swarm, positions)
    
    print(f"\nInitial 2x2 flat formation:")
    for cube in swarm.get_all_cubes():
        print(f"  Cube {cube.cube_id}: pos={cube.position}")
    
    movement = MovementSystem(swarm, require_connectivity=False)
    
    # Get valid moves for cube 0 (corner cube)
    valid_moves_0 = movement.get_valid_moves(0)
    print(f"\nValid moves for cube 0 (at origin): {len(valid_moves_0)}")
    for move in valid_moves_0[:5]:  # Show first 5
        print(f"  Edge {move.pivot_edge.name}, direction {move.direction}")
    
    # Execute a move to make cube 0 climb on top
    print("\nLooking for a climb move...")
    for move in valid_moves_0:
        # Try to find a move that would put cube 0 on top of another
        result = movement._compute_move_result(move, dry_run=True)
        if result.success and result.new_position[2] > 0:
            print(f"  Found climb move: Edge {move.pivot_edge.name}, dir {move.direction}")
            print(f"  Would move to: {result.new_position}")
            
            # Execute it
            actual_result = movement.execute_move(move)
            print(f"  Executed: {actual_result.success}")
            break
    
    print(f"\nAfter move:")
    for cube in swarm.get_all_cubes():
        print(f"  Cube {cube.cube_id}: pos={cube.position}")
    
    print(f"  Connected: {swarm.is_connected()}")
    print(f"  Components: {swarm.get_connected_components()}")
    
    return swarm


def test_tasks():
    """Test task definitions and rewards."""
    print("\n" + "=" * 60)
    print("Testing Tasks")
    print("=" * 60)
    
    # Create swarm in cube formation
    swarm = Swarm(64)
    create_cube_formation(swarm, size=4)
    
    # Test FormCubeTask (should be complete already)
    task_cube = FormCubeTask(target_size=4)
    print(f"\nFormCubeTask on 4x4x4 cube:")
    print(f"  Reward: {task_cube.compute_reward(swarm):.4f}")
    print(f"  Progress: {task_cube.get_progress(swarm):.4f}")
    print(f"  Complete: {task_cube.is_complete(swarm)}")
    
    # Test FormPlaneTask (not complete yet)
    task_plane = FormPlaneTask(normal=(0, 0, 1), width=8, height=8)
    print(f"\nFormPlaneTask on 4x4x4 cube:")
    print(f"  Reward: {task_plane.compute_reward(swarm):.4f}")
    print(f"  Progress: {task_plane.get_progress(swarm):.4f}")
    print(f"  Complete: {task_plane.is_complete(swarm)}")
    
    # Test on actual plane
    swarm_plane = Swarm(64)
    create_plane_formation(swarm_plane, width=8, height=8, normal_axis=2)
    print(f"\nFormPlaneTask on 8x8 plane:")
    print(f"  Reward: {task_plane.compute_reward(swarm_plane):.4f}")
    print(f"  Progress: {task_plane.get_progress(swarm_plane):.4f}")
    print(f"  Complete: {task_plane.is_complete(swarm_plane)}")
    
    # Test FormLineTask
    task_line = FormLineTask(axis=(1, 0, 0), length=64)
    swarm_line = Swarm(64)
    create_line_formation(swarm_line, length=64, axis=0)
    print(f"\nFormLineTask on 64-line:")
    print(f"  Reward: {task_line.compute_reward(swarm_line):.4f}")
    print(f"  Progress: {task_line.get_progress(swarm_line):.4f}")
    print(f"  Complete: {task_line.is_complete(swarm_line)}")
    
    # Test MinimizeSurfaceTask
    task_surface = MinimizeSurfaceTask()
    print(f"\nMinimizeSurfaceTask on 4x4x4 cube:")
    print(f"  Reward: {task_surface.compute_reward(swarm):.4f}")
    print(f"  Complete: {task_surface.is_complete(swarm)}")
    
    print(f"\nMinimizeSurfaceTask on 8x8 plane:")
    print(f"  Reward: {task_surface.compute_reward(swarm_plane):.4f}")
    print(f"  Complete: {task_surface.is_complete(swarm_plane)}")
    
    # Test MaximizeSpreadTask
    task_spread = MaximizeSpreadTask(target_baseline=15.0)
    print(f"\nMaximizeSpreadTask (target=15) on 4x4x4 cube:")
    print(f"  Reward: {task_spread.compute_reward(swarm):.4f}")
    print(f"  Progress: {task_spread.get_progress(swarm):.4f}")
    metrics = SwarmMetrics(swarm)
    print(f"  Current baseline: {metrics.maximum_baseline():.4f}")
    
    print(f"\nMaximizeSpreadTask (target=15) on 64-line:")
    print(f"  Reward: {task_spread.compute_reward(swarm_line):.4f}")
    print(f"  Progress: {task_spread.get_progress(swarm_line):.4f}")
    metrics_line = SwarmMetrics(swarm_line)
    print(f"  Current baseline: {metrics_line.maximum_baseline():.4f}")


def test_environment():
    """Test the Gymnasium environment."""
    print("\n" + "=" * 60)
    print("Testing Gymnasium Environment")
    print("=" * 60)
    
    # Create environment
    env = SwarmReconfigurationEnv(
        num_cubes=64,
        task=FormPlaneTask(normal=(0, 0, 1), width=8, height=8),
        max_steps=100,
        initial_formation='cube',
        require_connectivity=False
    )
    
    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"\nAfter reset:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Valid moves: {info['valid_move_count']}")
    print(f"  Task progress: {info['task_progress']:.4f}")
    
    # Take some random valid actions
    print("\nTaking random valid actions:")
    total_reward = 0
    for step in range(10):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print(f"  Step {step}: No valid actions!")
            break
        
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"  Step {step}: action={action}, reward={reward:.4f}, "
              f"progress={info['task_progress']:.4f}, "
              f"success={info['move_success']}")
        
        if terminated or truncated:
            print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    print(f"\nTotal reward: {total_reward:.4f}")
    
    env.close()
    return env


def test_masked_environment():
    """Test the masked action space environment."""
    print("\n" + "=" * 60)
    print("Testing Masked Environment")
    print("=" * 60)
    
    env = MaskedSwarmEnv(
        num_cubes=64,
        task=FormPlaneTask(normal=(0, 0, 1)),
        max_steps=100,
        initial_formation='cube',
        max_valid_actions=256
    )
    
    print(f"\nMasked environment created:")
    print(f"  Action space: {env.action_space}")
    
    obs, info = env.reset(seed=42)
    print(f"\nAfter reset:")
    print(f"  Valid moves: {info['valid_move_count']}")
    
    # Take actions using indices into valid moves
    print("\nTaking sequential actions (0, 1, 2, ...):")
    for step in range(5):
        # Action 0 means "take the first valid move"
        action = step % info['valid_move_count']
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {step}: action={action}, reward={reward:.4f}, "
              f"valid_moves={info['valid_move_count']}")
        
        if terminated or truncated:
            break
    
    env.close()


def test_multi_task_environment():
    """Test the multi-task environment."""
    print("\n" + "=" * 60)
    print("Testing Multi-Task Environment")
    print("=" * 60)
    
    env = MultiTaskSwarmEnv(
        num_cubes=64,
        max_steps=50,
        initial_formation='cube'
    )
    
    print(f"\nMulti-task environment created with {len(env.tasks)} tasks")
    
    # Reset a few times to see different tasks
    print("\nSampling tasks:")
    for i in range(5):
        obs, info = env.reset(seed=i)
        print(f"  Reset {i}: task_type={info['sampled_task_type']}, "
              f"progress={info['task_progress']:.4f}")
    
    env.close()


def test_visualization():
    """Test the visualization system."""
    print("\n" + "=" * 60)
    print("Testing Visualization")
    print("=" * 60)
    
    # Create swarm
    swarm = Swarm(64)
    create_cube_formation(swarm, size=4)
    
    print("\nCreating visualizer...")
    viz = SwarmVisualizer(swarm)
    
    print("Rendering cube formation...")
    viz.render(title="4x4x4 Cube Formation", show_connections=True, show_ids=False)
    
    # Do some moves and visualize
    movement = MovementSystem(swarm, require_connectivity=False)
    
    print("Executing moves and updating visualization...")
    for i in range(10):
        valid_moves = movement.get_all_valid_moves()
        if valid_moves:
            move = valid_moves[np.random.randint(len(valid_moves))]
            result = movement.execute_move(move)
            if result.success:
                viz.render(
                    title=f"After move {i+1}",
                    highlight_cubes={move.cube_id},
                    show_connections=True
                )
    
    print("Visualization complete. Close the window to continue...")
    viz.show()
    viz.close()


def test_connectivity_constraint():
    """Test that connectivity constraints work properly."""
    print("\n" + "=" * 60)
    print("Testing Connectivity Constraints")
    print("=" * 60)
    
    # Create a small swarm for easier testing
    swarm = Swarm(8)
    positions = [
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
        (0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0)
    ]
    create_custom_formation(swarm, positions)
    
    print(f"\nInitial 4x2 formation:")
    print(f"  Connected: {swarm.is_connected()}")
    print(f"  Components: {len(swarm.get_connected_components())}")
    
    # Test with connectivity required
    movement_strict = MovementSystem(swarm.copy(), require_connectivity=True)
    valid_strict = movement_strict.get_all_valid_moves()
    print(f"\nWith connectivity required:")
    print(f"  Valid moves: {len(valid_strict)}")
    
    # Test without connectivity required
    movement_loose = MovementSystem(swarm.copy(), require_connectivity=False)
    valid_loose = movement_loose.get_all_valid_moves()
    print(f"\nWithout connectivity required:")
    print(f"  Valid moves: {len(valid_loose)}")
    
    # The loose version should have more or equal valid moves
    print(f"\nDifference: {len(valid_loose) - len(valid_strict)} additional moves when connectivity not required")


def test_split_groups_task():
    """Test the split groups task."""
    print("\n" + "=" * 60)
    print("Testing Split Groups Task")
    print("=" * 60)
    
    swarm = Swarm(64)
    create_cube_formation(swarm, size=4)
    
    task = SplitGroupsTask(num_groups=2, min_group_size=16)
    
    print(f"\nSplitGroupsTask (target: 2 groups, min size: 16)")
    print(f"  Initial reward: {task.compute_reward(swarm):.4f}")
    print(f"  Initial progress: {task.get_progress(swarm):.4f}")
    print(f"  Complete: {task.is_complete(swarm)}")
    print(f"  Components: {len(swarm.get_connected_components())}")
    
    # Manually break some connections to split the swarm
    # We'll disconnect along the middle plane
    print("\nManually splitting swarm...")
    
    # Find and remove connections between cubes at x=1 and x=2
    connections_to_remove = []
    for conn in swarm._connections.get_all_connections():
        cube1 = swarm.get_cube(conn.cube_id_1)
        cube2 = swarm.get_cube(conn.cube_id_2)
        
        # Check if this connection crosses the x=1.5 plane
        x1, x2 = cube1.position[0], cube2.position[0]
        if (x1 == 1 and x2 == 2) or (x1 == 2 and x2 == 1):
            connections_to_remove.append(conn)
    
    print(f"  Found {len(connections_to_remove)} connections to remove")
    
    for conn in connections_to_remove:
        swarm._connections.remove_connection(conn)
    
    print(f"\nAfter splitting:")
    print(f"  Components: {len(swarm.get_connected_components())}")
    components = swarm.get_connected_components()
    for i, comp in enumerate(components):
        print(f"    Component {i}: {len(comp)} cubes")
    
    print(f"  Reward: {task.compute_reward(swarm):.4f}")
    print(f"  Progress: {task.get_progress(swarm):.4f}")
    print(f"  Complete: {task.is_complete(swarm)}")
    
    return swarm


def test_orientation_tracking():
    """Test that cube orientations are tracked correctly through moves."""
    print("\n" + "=" * 60)
    print("Testing Orientation Tracking")
    print("=" * 60)
    
    # Create a small test swarm
    swarm = Swarm(4)
    positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    create_custom_formation(swarm, positions)
    
    # Get cube 0
    cube = swarm.get_cube(0)
    print(f"\nInitial orientation of cube 0:")
    print(f"  Position: {cube.position}")
    print(f"  Orientation: {cube.orientation}")
    
    # Check where each face is pointing
    print(f"  Face directions:")
    for face in Face:
        global_dir = cube.orientation.get_global_face_normal(face)
        print(f"    {face.name} -> {global_dir}")
    
    # Do a move and check orientation changed
    movement = MovementSystem(swarm, require_connectivity=False)
    valid_moves = movement.get_valid_moves(0)
    
    if valid_moves:
        move = valid_moves[0]
        print(f"\nExecuting move: Edge {move.pivot_edge.name}, direction {move.direction}")
        
        result = movement.execute_move(move)
        
        if result.success:
            cube = swarm.get_cube(0)  # Get updated cube
            print(f"\nAfter move:")
            print(f"  Position: {cube.position}")
            print(f"  Orientation: {cube.orientation}")
            print(f"  Face directions:")
            for face in Face:
                global_dir = cube.orientation.get_global_face_normal(face)
                print(f"    {face.name} -> {global_dir}")
    
    return swarm


def test_edge_support_detection():
    """Test that edge support detection works for various configurations."""
    print("\n" + "=" * 60)
    print("Testing Edge Support Detection")
    print("=" * 60)
    
    # Create L-shaped configuration
    swarm = Swarm(3)
    positions = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    create_custom_formation(swarm, positions)
    
    print(f"\nL-shaped configuration:")
    for cube in swarm.get_all_cubes():
        print(f"  Cube {cube.cube_id}: {cube.position}")
    
    movement = MovementSystem(swarm, require_connectivity=False)
    
    # Check which edges of cube 0 have support
    cube0 = swarm.get_cube(0)
    print(f"\nEdge support for cube 0 at {cube0.position}:")
    for edge in Edge:
        has_support = movement._edge_has_support(cube0, edge)
        print(f"  {edge.name}: {'Yes' if has_support else 'No'}")
    
    # Check cube 2 (corner of L)
    cube2 = swarm.get_cube(2)
    print(f"\nEdge support for cube 2 at {cube2.position}:")
    for edge in Edge:
        has_support = movement._edge_has_support(cube2, edge)
        print(f"  {edge.name}: {'Yes' if has_support else 'No'}")
    
    # Get valid moves for each cube
    for cube_id in range(3):
        valid = movement.get_valid_moves(cube_id)
        print(f"\nCube {cube_id} has {len(valid)} valid moves")


def test_swept_volume_collision():
    """Test that swept volume collision detection prevents invalid moves."""
    print("\n" + "=" * 60)
    print("Testing Swept Volume Collision")
    print("=" * 60)
    
    # Create configuration where a move would collide
    # 2x2 base with one cube on top
    swarm = Swarm(5)
    positions = [
        (0, 0, 0), (1, 0, 0),  # Bottom row
        (0, 1, 0), (1, 1, 0),  # Second row
        (0, 0, 1)              # One cube on top of (0,0,0)
    ]
    create_custom_formation(swarm, positions)
    
    print(f"\nConfiguration with potential collision:")
    for cube in swarm.get_all_cubes():
        print(f"  Cube {cube.cube_id}: {cube.position}")
    
    movement = MovementSystem(swarm, require_connectivity=False)
    
    # Try to move cube 1 up and over - should be blocked by cube 4
    cube1 = swarm.get_cube(1)
    print(f"\nChecking moves for cube 1 at {cube1.position}:")
    
    valid_moves = movement.get_valid_moves(1)
    print(f"  Valid moves: {len(valid_moves)}")
    
    # Try all possible moves and show which are blocked
    all_blocked = []
    for edge in Edge:
        for direction in [+1, -1]:
            move = HingeMove(1, edge, direction)
            result = movement._compute_move_result(move, dry_run=True)
            if not result.success:
                all_blocked.append((edge.name, direction, result.reason))
    
    print(f"\nBlocked moves for cube 1:")
    for edge_name, direction, reason in all_blocked[:10]:  # Show first 10
        print(f"  Edge {edge_name}, dir {direction}: {reason}")


def run_simple_episode():
    """Run a simple episode with random actions for demonstration."""
    print("\n" + "=" * 60)
    print("Running Simple Episode")
    print("=" * 60)
    
    env = SwarmReconfigurationEnv(
        num_cubes=64,
        task=FormPlaneTask(normal=(0, 0, 1), width=8, height=8),
        max_steps=200,
        initial_formation='cube',
        require_connectivity=False
    )
    
    obs, info = env.reset(seed=42)
    
    print(f"\nStarting episode:")
    print(f"  Task: Form 8x8 plane with normal (0, 0, 1)")
    print(f"  Initial progress: {info['task_progress']:.4f}")
    
    episode_reward = 0
    step = 0
    
    while True:
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            print(f"\nNo valid actions at step {step}!")
            break
        
        # Random action selection
        action = np.random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step += 1
        
        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"  Step {step}: progress={info['task_progress']:.4f}, "
                  f"reward={reward:.4f}, total={episode_reward:.4f}")
        
        if terminated:
            print(f"\nTask completed at step {step}!")
            break
        
        if truncated:
            print(f"\nEpisode truncated at step {step}")
            break
    
    print(f"\nEpisode summary:")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {episode_reward:.4f}")
    print(f"  Final progress: {info['task_progress']:.4f}")
    print(f"  Task complete: {info.get('task_complete', False)}")
    
    env.close()


def benchmark_move_computation():
    """Benchmark the speed of move computation."""
    print("\n" + "=" * 60)
    print("Benchmarking Move Computation")
    print("=" * 60)
    
    import time
    
    swarm = Swarm(64)
    create_cube_formation(swarm, size=4)
    movement = MovementSystem(swarm, require_connectivity=False)
    
    # Benchmark valid move computation
    print("\nBenchmarking get_all_valid_moves()...")
    num_iterations = 100
    
    start = time.time()
    for _ in range(num_iterations):
        valid_moves = movement.get_all_valid_moves()
    elapsed = time.time() - start
    
    print(f"  {num_iterations} iterations in {elapsed:.3f}s")
    print(f"  {elapsed/num_iterations*1000:.2f}ms per call")
    print(f"  Found {len(valid_moves)} valid moves")
    
    # Benchmark move execution
    print("\nBenchmarking move execution...")
    
    # Make a copy to reset to
    original_swarm = swarm.copy()
    
    num_moves = 100
    start = time.time()
    
    for i in range(num_moves):
        valid_moves = movement.get_all_valid_moves()
        if valid_moves:
            move = valid_moves[i % len(valid_moves)]
            movement.execute_move(move)
    
    elapsed = time.time() - start
    
    print(f"  {num_moves} moves in {elapsed:.3f}s")
    print(f"  {elapsed/num_moves*1000:.2f}ms per move (including validation)")
    
    # Benchmark environment step
    print("\nBenchmarking environment step...")
    
    env = SwarmReconfigurationEnv(
        num_cubes=64,
        task=FormPlaneTask(),
        max_steps=1000,
        initial_formation='cube'
    )
    
    obs, info = env.reset(seed=42)
    
    num_steps = 100
    start = time.time()
    
    for _ in range(num_steps):
        valid_actions = env.get_valid_actions()
        if valid_actions:
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    
    elapsed = time.time() - start
    
    print(f"  {num_steps} steps in {elapsed:.3f}s")
    print(f"  {elapsed/num_steps*1000:.2f}ms per step")
    print(f"  {1.0/(elapsed/num_steps):.1f} steps per second")
    
    env.close()


def demo_reconfiguration_sequence():
    """
    Demonstrate a scripted reconfiguration from cube to plane.
    
    This shows what the RL agent should learn to do.
    """
    print("\n" + "=" * 60)
    print("Demo: Scripted Reconfiguration (Cube -> Plane)")
    print("=" * 60)
    
    swarm = Swarm(64)
    create_cube_formation(swarm, size=4)
    
    print(f"\nInitial 4x4x4 cube:")
    metrics = SwarmMetrics(swarm)
    print(f"  Surface area: {swarm.get_surface_area()}")
    print(f"  Planarity (Z): {metrics.planarity((0, 0, 1)):.4f}")
    print(f"  Bounds: {swarm.get_bounds()}")
    
    movement = MovementSystem(swarm, require_connectivity=False)
    
    # Strategy: Move cubes from upper layers to expand the base
    # This is a simplified heuristic, not optimal
    
    moves_executed = 0
    max_moves = 500
    
    while moves_executed < max_moves:
        # Check if we've reached a plane
        metrics = SwarmMetrics(swarm)
        if metrics.planarity((0, 0, 1), tolerance=0.5) > 0.99:
            bounds = swarm.get_bounds()
            if bounds[0][2] == bounds[1][2]:  # All at same Z level
                print(f"\nReached planar configuration after {moves_executed} moves!")
                break
        
        # Find cubes not at z=0 and try to move them down/outward
        valid_moves = movement.get_all_valid_moves()
        
        # Prioritize moves that reduce height
        best_move = None
        best_score = -float('inf')
        
        for move in valid_moves:
            cube = swarm.get_cube(move.cube_id)
            result = movement._compute_move_result(move, dry_run=True)
            
            if not result.success:
                continue
            
            # Score: prefer moves that go down or spread out
            old_z = cube.position[2]
            new_z = result.new_position[2]
            
            # Prefer going down
            score = (old_z - new_z) * 10
            
            # Also prefer spreading out in XY
            old_dist = abs(cube.position[0] - 1.5) + abs(cube.position[1] - 1.5)
            new_dist = abs(result.new_position[0] - 3.5) + abs(result.new_position[1] - 3.5)
            score += (new_dist - old_dist)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None:
            print(f"\nNo improving move found at step {moves_executed}")
            break
        
        # Execute the best move
        result = movement.execute_move(best_move)
        moves_executed += 1
        
        if moves_executed % 50 == 0:
            metrics = SwarmMetrics(swarm)
            print(f"  Step {moves_executed}: planarity={metrics.planarity((0, 0, 1)):.4f}, "
                  f"bounds={swarm.get_bounds()}")
    
    print(f"\nFinal configuration after {moves_executed} moves:")
    metrics = SwarmMetrics(swarm)
    print(f"  Surface area: {swarm.get_surface_area()}")
    print(f"  Planarity (Z): {metrics.planarity((0, 0, 1)):.4f}")
    print(f"  Bounds: {swarm.get_bounds()}")
    print(f"  Connected: {swarm.is_connected()}")
    
    return swarm


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Run all tests and demos."""
    print("=" * 60)
    print("MODULAR SPACECRAFT SWARM SIMULATION")
    print("=" * 60)
    
    # Run tests
    # test_basic_simulation()
    # test_formations()
    # test_movement_mechanics()
    # test_tasks()
    # test_environment()
    # test_masked_environment()
    # test_multi_task_environment()
    # test_connectivity_constraint()
    # test_split_groups_task()
    # test_orientation_tracking()
    # test_edge_support_detection()
    # test_swept_volume_collision()
    
    # # Run benchmarks
    # benchmark_move_computation()
    
    # # Run demos
    # run_simple_episode()
    # demo_reconfiguration_sequence()
    
    # Optional: visualization test (requires display)
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on system
        
        response = input("\nRun visualization test? (y/n): ")
        if response.lower() == 'y':
            test_visualization()
    except Exception as e:
        print(f"\nSkipping visualization test: {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()