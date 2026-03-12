"""Sewer environment parameters — Indian municipal infrastructure specs."""

from dataclasses import dataclass, field


@dataclass
class PipeParams:
    """Pipe geometry based on CPHEEO Manual on Sewerage and Sewage Treatment Systems."""

    diameter_min_mm: float = 600.0
    diameter_max_mm: float = 1200.0
    diameter_default_mm: float = 900.0
    length_m: float = 3.0
    wall_thickness_mm: float = 50.0
    materials: list[str] = field(default_factory=lambda: ["concrete", "brick", "corroded_metal"])


@dataclass
class SludgeParams:
    """Sludge properties from NEERI and IIT Bombay characterization studies."""

    total_solids_min_pct: float = 2.0
    total_solids_max_pct: float = 8.0
    specific_gravity_min: float = 1.02
    specific_gravity_max: float = 1.05
    # MuJoCo proxy: rigid body blocks
    block_count_min: int = 5
    block_count_max: int = 15
    block_size_min_mm: float = 50.0
    block_size_max_mm: float = 200.0
    block_density: float = 1200.0  # kg/m^3


@dataclass
class RobotParams:
    """Reference embodiment based on Bandicoot-class sewer robots."""

    base_width_mm: float = 600.0
    base_length_mm: float = 800.0
    base_height_mm: float = 300.0
    arm_dof: int = 4  # Phase 0 simplified; Phase 1 = 6
    scoop_width_mm: float = 200.0
    front_cam_resolution: tuple[int, int] = (480, 640)
    wrist_cam_resolution: tuple[int, int] = (224, 224)


@dataclass
class DomainRandomization:
    """Randomization ranges applied at each episode reset."""

    pipe_diameter: bool = True
    sludge_count: bool = True
    sludge_positions: bool = True
    lighting_intensity: bool = True
    lighting_intensity_range: tuple[float, float] = (0.3, 1.0)


@dataclass
class EnvConfig:
    """Complete environment configuration."""

    pipe: PipeParams = field(default_factory=PipeParams)
    sludge: SludgeParams = field(default_factory=SludgeParams)
    robot: RobotParams = field(default_factory=RobotParams)
    randomization: DomainRandomization = field(default_factory=DomainRandomization)
    max_episode_steps: int = 500
    control_frequency_hz: float = 10.0
    tasks: list[str] = field(
        default_factory=lambda: [
            "navigate to blockage",
            "assess and position for extraction",
            "extract sludge at current position",
            "deposit extracted material",
        ]
    )
