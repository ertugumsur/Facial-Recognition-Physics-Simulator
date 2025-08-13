# Facial-Recognition-Physics-Simulator

1. Real-Time Facial Recognition (PCA-Based)
The facial recognition algorithm uses Principal Component Analysis (PCA) to compare the state of faces and classify them into expressions such as smiling or sad.

Real-time processing: Implemented with the cv2 library to capture live frames from a webcam and assess facial states instantly.

Feature reduction: PCA is applied to reduce dimensionality while preserving expression-related features.

Classification: Faces are projected into PCA space, compared to reference states, and assigned an expression label.

2. Ion–Electron Physics Simulator (Hall Thruster Channel)
This simulation models the E×B drift inside Hall Thruster channels to study how channel length affects electron capturing.

In a Hall Thruster:

The goal is to trap an electron cloud near the outer side of the channel, where it ionizes neutral propellant and accelerates the resulting ions for thrust.

If the channel length is not at least an order of magnitude larger than the electron Larmor radius, electrons may short to the anode, damaging the thruster.

If the channel length is too long, ions may fail to escape the channel, resulting in zero thrust.

The simulator:

Models electron and ion trajectories under given E and B fields.

Determines the optimal channel length to balance electron confinement and ion acceleration.

Was used in the Olin Plasma Engineering Laboratory to help design a new Hall Thruster
