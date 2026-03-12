# Reference Robot Embodiment

## Base

- Type: Tracked differential drive
- Width: 600mm (fits Indian municipal sewer mains ≥600mm ID)
- Length: 800mm
- Height: 300mm
- Drive: 2-DOF — forward velocity, angular velocity

## Arm

- Type: Serial-chain articulated manipulator
- Phase 0 (MuJoCo): 4 revolute joints
- Phase 1 (Isaac Sim): 6 revolute joints
- Mounting: Shoulder on top of base, forward-facing
- Reach: sufficient to contact pipe floor from base height

## End-Effector

- Type: Scoop with hinged aperture
- Width: 200mm
- 1-DOF: aperture open/close

## Sensors

- Front camera: RGB-D, 640×480, mounted on base front, forward-facing
- Wrist camera: RGB, 224×224, mounted on last arm link
- IMU: on base
- Joint encoders: all arm joints

## Action Space

7-DOF flat vector, normalized to [-1, 1]:

| Index | Description | Used in |
|-------|-------------|---------|
| 0 | Base forward velocity | navigate |
| 1 | Base lateral velocity | navigate |
| 2 | Arm joint 1 velocity | position, extract, deposit |
| 3 | Arm joint 2 velocity | position, extract, deposit |
| 4 | Arm joint 3 velocity | position, extract, deposit |
| 5 | Arm joint 4 velocity | position, extract, deposit |
| 6 | Scoop aperture velocity | extract, deposit |

The model learns which dimensions to activate per task from the language prompt.

## Target Deployment Hardware

- NVIDIA Jetson Orin NX
- Budget: sub-₹2 lakh total robot cost
- Inference: <100ms per action chunk via TensorRT
- Power: battery-operated, no tethered power
- Connectivity: none required (fully edge)
