Self-Driving Reinforcement Learning using PPO
A Multi-Modal Autonomous Driving Simulation using Pure NumPy and PPO

1. Introduction
This project implements a complete self-driving car simulation environment using Reinforcement Learning (RL). The autonomous agent learns how to navigate roads, avoid obstacles, follow waypoints, and control vehicle movement through continuous interaction with a simulated environment.

Unlike many modern RL projects that rely heavily on deep learning frameworks such as TensorFlow or PyTorch, this project is built entirely using Python and NumPy. It demonstrates the internal mechanics of neural networks, PPO optimization, sensor fusion, and vehicle simulation from scratch.
2. Objectives
•	Build a realistic self-driving simulation environment.
•	Implement PPO reinforcement learning from scratch.
•	Integrate multiple autonomous vehicle sensors.
•	Train an agent to navigate safely and efficiently.
•	Demonstrate sensor fusion for autonomous navigation.
•	Create an educational RL framework without external DL libraries.
3. Key Features
•	Pure NumPy neural network implementation
•	Proximal Policy Optimization (PPO) agent
•	Multi-modal sensor fusion system
•	Dynamic traffic and obstacle simulation
•	Bird’s-eye semantic camera representation
•	Waypoint-based navigation system
•	Vehicle physics using bicycle dynamics
•	Checkpoint save/load functionality
•	Training, evaluation, and demo modes
•	Modular and extensible architecture
4. System Architecture

Sensors
  LiDAR
  Radar
  Ultrasonic Sensors
  Semantic Camera
 GPS

        

Sensor Encoders
  CNN Encoders
  MLP Encoders
  Feature Fusion

        

PPO Reinforcement Learning Agent
  Actor Network
  Critic Network

        

Vehicle Controller
  Steering
  Throttle
  Brake

5. Sensor System
LiDAR
•	360-degree rotating scan
•	Obstacle distance detection
•	Noise and dropout simulation
•	High-resolution environmental mapping
Radar
•	Forward object tracking
•	Relative velocity estimation
•	Distance and angle measurement
•	Long-range detection capability
Ultrasonic Sensors
•	Short-range collision avoidance
•	8-direction proximity sensing
•	Parking and close-range navigation
Semantic Camera
•	Bird’s-eye semantic segmentation
•	Road mask generation
•	Obstacle heatmap
•	Goal waypoint visualization
GPS Navigation
•	Waypoint routing
•	Position estimation
•	Heading calculation
•	Speed measurement
6. Observation Space
Sensor	Dimensions
LiDAR	360
Radar	32
Ultrasonic	8
GPS Navigation	6
Camera	64 × 64 × 3

7. Action Space
Action	Range
Steering	-1 to 1
Throttle	0 to 1
Brake	0 to 1
8. Reinforcement Learning
The project uses Proximal Policy Optimization (PPO), one of the most stable and widely used reinforcement learning algorithms for continuous control problems.

The PPO agent learns by interacting with the environment and receiving rewards based on driving performance. The model includes Actor-Critic architecture, Generalized Advantage Estimation (GAE), entropy regularization, and gradient-based policy updates.
9. Reward Function
•	Positive reward for moving toward waypoints
•	Large reward for reaching checkpoints
•	Reward for completing destination
•	Penalty for collisions
•	Penalty for off-road driving
•	Penalty for excessive speed
•	Penalty for sudden jerky actions
•	Penalty for idling
10. Neural Network Architecture
Different neural encoders process different sensor modalities:

• LiDAR Encoder → 1D CNN
• Camera Encoder → 2D CNN
• Radar Encoder → MLP
• Ultrasonic Encoder → MLP
• GPS Encoder → MLP

All encoded features are fused together into a shared latent representation before being passed into the Actor and Critic networks.
11. Installation and Usage
•	pip install numpy
•	python self_driving_rl.py
•	python self_driving_rl.py --demo
•	python self_driving_rl.py --eval checkpoints/best.npy
•	python self_driving_rl.py --load checkpoints/best.npy
12. Environment Details
•	Grid-based road network
•	Multiple navigation routes
•	Dynamic traffic vehicles
•	Boundary walls
•	Waypoint navigation system
•	Kinematic bicycle vehicle model
13. Future Improvements
•	PyTorch or TensorFlow integration
•	GPU acceleration
•	Real-time visualization
•	Lane detection
•	Traffic signal handling
•	Pedestrian simulation
•	3D simulation support
•	Real-world dataset integration
•	Multi-agent traffic environments
14. Educational Value
This project is highly suitable for students and researchers interested in Autonomous Vehicles, Reinforcement Learning, Artificial Intelligence, Sensor Fusion, Computer Vision, and Robotics.
15. Conclusion
This project demonstrates how a complete autonomous driving system can be built using reinforcement learning and multi-modal sensor fusion. The implementation provides a strong foundation for understanding PPO, autonomous navigation, vehicle dynamics, and AI-based control systems.
16. Author
Abishek Prabhu
AI & Machine Learning Student
