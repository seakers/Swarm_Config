from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from core.cube import Face, Cube, Orientation


class FaceFunction(Enum):
    """
    The primary function assigned to each face of a cubesat.
    
    Each cube has 6 faces, and each face has a dedicated function.
    When faces are bonded to other cubes, their function may be
    blocked or enhanced depending on the configuration.
    """
    
    # Communication
    ANTENNA_HIGH_GAIN = auto()    # For Earth communication (directional)
    ANTENNA_INTER_SAT = auto()    # For inter-satellite links within swarm
    
    # Power
    SOLAR_ARRAY = auto()          # Photovoltaic cells for power generation
    
    # Thermal
    RADIATOR = auto()             # Thermal rejection surface
    
    # Science/Sensing
    CAMERA = auto()               # Optical imaging instrument
    SCIENCE_INSTRUMENTS = auto()  # Magnetometer, particle detector, spectrometer, etc.
    
    # Utility (all faces have these in addition to primary function)
    # ELECTROMAGNET - implicit on all faces
    # PROXIMITY_SENSOR - implicit on all faces
    # LED_MARKERS - implicit on all faces


# =============================================================================
# Standard Face Assignment
# =============================================================================

# This defines which function is on which face in the LOCAL frame of each cube
# All cubes have identical face assignments in their local frame

STANDARD_FACE_ASSIGNMENT: Dict[Face, FaceFunction] = {
    Face.POS_Z: FaceFunction.ANTENNA_HIGH_GAIN,    # Top: Earth comms
    Face.NEG_Z: FaceFunction.SOLAR_ARRAY,          # Bottom: Solar power
    Face.POS_X: FaceFunction.CAMERA,               # Front: Primary imaging
    Face.NEG_X: FaceFunction.RADIATOR,             # Back: Thermal rejection
    Face.POS_Y: FaceFunction.ANTENNA_INTER_SAT,    # Left: Swarm comms
    Face.NEG_Y: FaceFunction.SCIENCE_INSTRUMENTS,  # Right: Science payload
}

# Reverse mapping: function -> face
FUNCTION_TO_FACE: Dict[FaceFunction, Face] = {
    v: k for k, v in STANDARD_FACE_ASSIGNMENT.items()
}


# =============================================================================
# Face Function Properties
# =============================================================================

@dataclass
class FaceFunctionProperties:
    """
    Properties of each face function that affect mission operations.
    """
    
    # Does this face need clear line-of-sight to operate?
    requires_los: bool = True
    
    # Does this face need to point at a specific target?
    requires_pointing: bool = False
    
    # What target does it need to point at? (if requires_pointing)
    pointing_target: Optional[str] = None  # 'earth', 'sun', 'target', 'cold_space'
    
    # Can multiple faces of this type combine for enhanced capability?
    can_combine: bool = False
    
    # Combination type if can_combine
    combination_effect: Optional[str] = None  # 'additive', 'aperture_synthesis', 'phased_array'
    
    # Does this face generate heat?
    heat_generation: float = 0.0  # Watts
    
    # Does this face need cooling?
    requires_cooling: bool = False
    
    # Power consumption when active
    power_consumption: float = 0.0  # Watts
    
    # Power generation (for solar arrays)
    power_generation: float = 0.0  # Watts at 1 AU (scales with distance)
    
    # Data generation rate when active
    data_rate: float = 0.0  # bits per second


# Properties for each face function
FACE_FUNCTION_PROPERTIES: Dict[FaceFunction, FaceFunctionProperties] = {
    
    FaceFunction.ANTENNA_HIGH_GAIN: FaceFunctionProperties(
        requires_los=True,
        requires_pointing=True,
        pointing_target='earth',
        can_combine=True,
        combination_effect='phased_array',  # Combining antennas increases gain
        heat_generation=2.0,
        requires_cooling=False,
        power_consumption=5.0,
        data_rate=1000.0,  # bits/sec transmit capability
    ),
    
    FaceFunction.ANTENNA_INTER_SAT: FaceFunctionProperties(
        requires_los=True,
        requires_pointing=False,  # Omnidirectional for nearby cubes
        can_combine=False,
        heat_generation=0.5,
        power_consumption=1.0,
        data_rate=10000.0,  # Higher rate for short-range
    ),
    
    FaceFunction.SOLAR_ARRAY: FaceFunctionProperties(
        requires_los=True,
        requires_pointing=True,
        pointing_target='sun',
        can_combine=True,
        combination_effect='additive',  # More panels = more power
        heat_generation=0.0,
        power_consumption=0.0,
        power_generation=2.0,  # Watts per face at 1 AU
    ),
    
    FaceFunction.RADIATOR: FaceFunctionProperties(
        requires_los=True,
        requires_pointing=True,
        pointing_target='cold_space',  # Should point away from sun
        can_combine=True,
        combination_effect='additive',
        heat_generation=-5.0,  # Negative = heat rejection
        requires_cooling=False,
    ),
    
    FaceFunction.CAMERA: FaceFunctionProperties(
        requires_los=True,
        requires_pointing=True,
        pointing_target='target',  # Science target
        can_combine=True,
        combination_effect='aperture_synthesis',  # Sparse aperture imaging
        heat_generation=1.0,
        requires_cooling=True,  # Cameras often need cooling
        power_consumption=3.0,
        data_rate=1000000.0,  # High data rate for imagery
    ),
    
    FaceFunction.SCIENCE_INSTRUMENTS: FaceFunctionProperties(
        requires_los=True,  # Magnetometer, particle detectors need exposure
        requires_pointing=False,  # Omnidirectional sensing
        can_combine=True,
        combination_effect='additive',  # More samples = better data
        heat_generation=0.5,
        power_consumption=2.0,
        data_rate=1000.0,
    ),
}


# =============================================================================
# Cubesat Subsystems
# =============================================================================

@dataclass
class PowerSubsystem:
    """
    Power generation and storage for a cubesat.
    """
    battery_capacity: float = 20.0  # Watt-hours
    battery_charge: float = 20.0    # Current charge (Watt-hours)
    battery_health: float = 1.0     # Degradation factor (0-1)
    
    # Power bus voltage
    bus_voltage: float = 5.0  # Volts
    
    # Can receive power from connected cubes
    power_sharing_enabled: bool = True
    
    def get_available_power(self) -> float:
        """Get currently available power."""
        return self.battery_charge * self.battery_health
    
    def consume_power(self, watt_hours: float) -> bool:
        """
        Consume power from battery.
        Returns True if sufficient power was available.
        """
        if watt_hours <= self.battery_charge:
            self.battery_charge -= watt_hours
            return True
        return False
    
    def charge(self, watt_hours: float) -> None:
        """Add charge to battery (from solar or power sharing)."""
        self.battery_charge = min(
            self.battery_capacity * self.battery_health,
            self.battery_charge + watt_hours
        )
    
    def get_charge_fraction(self) -> float:
        """Get battery charge as fraction of capacity."""
        return self.battery_charge / (self.battery_capacity * self.battery_health)


@dataclass
class ThermalSubsystem:
    """
    Thermal management for a cubesat.
    """
    # Current temperature (Celsius)
    temperature: float = 20.0
    
    # Acceptable temperature range
    min_operating_temp: float = -20.0
    max_operating_temp: float = 50.0
    
    # Thermal mass (determines how fast temperature changes)
    thermal_mass: float = 1.0  # J/°C
    
    # Internal heat generation from electronics
    base_heat_generation: float = 2.0  # Watts
    
    # Heater for cold conditions
    heater_power: float = 5.0  # Watts when active
    heater_active: bool = False
    
    def is_in_safe_range(self) -> bool:
        """Check if temperature is within operating limits."""
        return self.min_operating_temp <= self.temperature <= self.max_operating_temp
    
    def get_thermal_margin(self) -> float:
        """
        Get margin to nearest thermal limit.
        Positive = safe, negative = out of range.
        """
        margin_low = self.temperature - self.min_operating_temp
        margin_high = self.max_operating_temp - self.temperature
        return min(margin_low, margin_high)


@dataclass
class AttitudeSubsystem:
    """
    Attitude determination and control for a cubesat.
    """
    # Reaction wheel momentum (Nms) - 3 axis
    wheel_momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Maximum wheel momentum before saturation
    max_wheel_momentum: float = 0.01  # Nms
    
    # Wheel health
    wheel_health: np.ndarray = field(default_factory=lambda: np.ones(3))
    
    # Star tracker available
    star_tracker_functional: bool = True
    
    # Sun sensor available  
    sun_sensor_functional: bool = True
    
    def is_wheel_saturated(self) -> bool:
        """Check if any reaction wheel is near saturation."""
        return np.any(np.abs(self.wheel_momentum) > 0.9 * self.max_wheel_momentum)
    
    def get_saturation_fraction(self) -> float:
        """Get maximum saturation fraction across all wheels."""
        return np.max(np.abs(self.wheel_momentum)) / self.max_wheel_momentum


@dataclass
class DataSubsystem:
    """
    Data storage and processing for a cubesat.
    """
    # Storage capacity (bits)
    storage_capacity: float = 1e9  # 1 Gbit
    storage_used: float = 0.0
    
    # Processing capability
    processor_speed: float = 100e6  # 100 MHz
    processor_functional: bool = True
    
    def get_storage_fraction(self) -> float:
        """Get fraction of storage used."""
        return self.storage_used / self.storage_capacity
    
    def store_data(self, bits: float) -> bool:
        """
        Store data. Returns True if space was available.
        """
        if self.storage_used + bits <= self.storage_capacity:
            self.storage_used += bits
            return True
        return False
    
    def clear_data(self, bits: float) -> None:
        """Clear data (after transmission)."""
        self.storage_used = max(0, self.storage_used - bits)


@dataclass
class CommunicationSubsystem:
    """
    Communication capabilities for a cubesat.
    """
    # Transmitter
    transmit_power: float = 1.0  # Watts
    transmit_frequency: float = 8.4e9  # X-band (Hz)
    transmit_functional: bool = True
    
    # Receiver
    receiver_sensitivity: float = -120  # dBm
    receiver_functional: bool = True
    
    # Inter-satellite link
    isl_functional: bool = True
    
    # Current communication state
    is_transmitting: bool = False
    is_receiving: bool = False


# =============================================================================
# Enhanced Cube Class
# =============================================================================

@dataclass
class CubesatSubsystems:
    """
    All subsystems for a cubesat, bundled together.
    """
    power: PowerSubsystem = field(default_factory=PowerSubsystem)
    thermal: ThermalSubsystem = field(default_factory=ThermalSubsystem)
    attitude: AttitudeSubsystem = field(default_factory=AttitudeSubsystem)
    data: DataSubsystem = field(default_factory=DataSubsystem)
    comms: CommunicationSubsystem = field(default_factory=CommunicationSubsystem)
    
    def copy(self) -> 'CubesatSubsystems':
        """Create a deep copy."""
        return CubesatSubsystems(
            power=PowerSubsystem(
                battery_capacity=self.power.battery_capacity,
                battery_charge=self.power.battery_charge,
                battery_health=self.power.battery_health,
                bus_voltage=self.power.bus_voltage,
                power_sharing_enabled=self.power.power_sharing_enabled,
            ),
            thermal=ThermalSubsystem(
                temperature=self.thermal.temperature,
                min_operating_temp=self.thermal.min_operating_temp,
                max_operating_temp=self.thermal.max_operating_temp,
                thermal_mass=self.thermal.thermal_mass,
                base_heat_generation=self.thermal.base_heat_generation,
                heater_power=self.thermal.heater_power,
                heater_active=self.thermal.heater_active,
            ),
            attitude=AttitudeSubsystem(
                wheel_momentum=self.attitude.wheel_momentum.copy(),
                max_wheel_momentum=self.attitude.max_wheel_momentum,
                wheel_health=self.attitude.wheel_health.copy(),
                star_tracker_functional=self.attitude.star_tracker_functional,
                sun_sensor_functional=self.attitude.sun_sensor_functional,
            ),
            data=DataSubsystem(
                storage_capacity=self.data.storage_capacity,
                storage_used=self.data.storage_used,
                processor_speed=self.data.processor_speed,
                processor_functional=self.data.processor_functional,
            ),
            comms=CommunicationSubsystem(
                transmit_power=self.comms.transmit_power,
                transmit_frequency=self.comms.transmit_frequency,
                transmit_functional=self.comms.transmit_functional,
                receiver_sensitivity=self.comms.receiver_sensitivity,
                receiver_functional=self.comms.receiver_functional,
                isl_functional=self.comms.isl_functional,
                is_transmitting=self.comms.is_transmitting,
                is_receiving=self.comms.is_receiving,
            ),
        )


# =============================================================================
# Updated Cube class with face functions and subsystems
# =============================================================================

@dataclass
class EnhancedCube:
    """
    A cubesat with defined face functions and subsystems.
    
    This extends the basic Cube class with:
    - Face function assignments
    - Subsystem states
    - Methods to query face exposure and capability
    """
    
    cube_id: int
    position: Tuple[int, int, int]
    orientation: Orientation = field(default_factory=Orientation)
    
    # Subsystems
    subsystems: CubesatSubsystems = field(default_factory=CubesatSubsystems)
    
    # Face function assignment (default: standard assignment)
    face_functions: Dict[Face, FaceFunction] = field(
        default_factory=lambda: STANDARD_FACE_ASSIGNMENT.copy()
    )
    
    # Overall health/functionality
    is_functional: bool = True
    
    def get_face_function(self, local_face: Face) -> FaceFunction:
        """Get the function assigned to a face in local coordinates."""
        return self.face_functions.get(local_face, FaceFunction.SCIENCE_INSTRUMENTS)
    
    def get_face_for_function(self, function: FaceFunction) -> Optional[Face]:
        """Get the local face that has a given function."""
        for face, func in self.face_functions.items():
            if func == function:
                return face
        return None
    
    def get_global_direction_for_function(self, function: FaceFunction) -> Optional[Tuple[int, int, int]]:
        """
        Get the global direction that a function is pointing.
        
        Returns the unit vector in global coordinates that the specified
        function's face is pointing toward.
        """
        local_face = self.get_face_for_function(function)
        if local_face is None:
            return None
        
        return self.orientation.get_global_face_normal(local_face)
    
    def get_function_pointing_in_direction(self, global_dir: Tuple[int, int, int]) -> FaceFunction:
        """
        Get which function is pointing in a given global direction.
        """
        local_face = self.orientation.get_local_face_for_direction(global_dir)
        return self.get_face_function(local_face)
    
    def is_face_exposed(self, local_face: Face, 
                        occupied_positions: Set[Tuple[int, int, int]]) -> bool:
        """
        Check if a face is exposed (not blocked by another cube).
        
        Args:
            local_face: The face to check (in local coordinates)
            occupied_positions: Set of all occupied grid positions
            
        Returns:
            True if the face is exposed to space
        """
        global_dir = self.orientation.get_global_face_normal(local_face)
        adjacent_pos = (
            self.position[0] + global_dir[0],
            self.position[1] + global_dir[1],
            self.position[2] + global_dir[2]
        )
        return adjacent_pos not in occupied_positions
    
    def is_function_exposed(self, function: FaceFunction,
                            occupied_positions: Set[Tuple[int, int, int]]) -> bool:
        """
        Check if the face with a given function is exposed.
        """
        local_face = self.get_face_for_function(function)
        if local_face is None:
            return False
        return self.is_face_exposed(local_face, occupied_positions)
    
    def get_exposed_faces(self, occupied_positions: Set[Tuple[int, int, int]]) -> List[Face]:
        """Get list of all exposed faces."""
        exposed = []
        for face in Face:
            if self.is_face_exposed(face, occupied_positions):
                exposed.append(face)
        return exposed
    
    def get_exposed_functions(self, occupied_positions: Set[Tuple[int, int, int]]) -> List[FaceFunction]:
        """Get list of all exposed face functions."""
        exposed = []
        for face in self.get_exposed_faces(occupied_positions):
            exposed.append(self.get_face_function(face))
        return exposed
    
    def get_blocked_functions(self, occupied_positions: Set[Tuple[int, int, int]]) -> List[FaceFunction]:
        """Get list of all blocked (bonded) face functions."""
        blocked = []
        for face in Face:
            if not self.is_face_exposed(face, occupied_positions):
                blocked.append(self.get_face_function(face))
        return blocked
    
    def compute_alignment_score(self, function: FaceFunction,
                                target_direction: Tuple[float, float, float]) -> float:
        """
        Compute how well a function is aligned with a target direction.
        
        Args:
            function: The function to check
            target_direction: Desired pointing direction (will be normalized)
            
        Returns:
            Dot product of face normal with target direction (-1 to +1)
            +1 means perfectly aligned, -1 means opposite, 0 means perpendicular
        """
        face_dir = self.get_global_direction_for_function(function)
        if face_dir is None:
            return 0.0
        
        # Normalize target direction
        target = np.array(target_direction, dtype=float)
        target_norm = np.linalg.norm(target)
        if target_norm < 1e-10:
            return 0.0
        target = target / target_norm
        
        face_vec = np.array(face_dir, dtype=float)
        
        return float(np.dot(face_vec, target))
    
    def copy(self) -> 'EnhancedCube':
        """Create a deep copy of this cube."""
        return EnhancedCube(
            cube_id=self.cube_id,
            position=self.position,
            orientation=self.orientation.copy(),
            subsystems=self.subsystems.copy(),
            face_functions=self.face_functions.copy(),
            is_functional=self.is_functional,
        )
