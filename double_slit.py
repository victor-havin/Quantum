# Double slit simulation
# Works in 2 stages:
# 1. Obtain interference pattern from a single qubit with phase modulation
# 2. Postprocess the result classically to simulate diffraction
# You can apply 'which-path' measurement, the interference pattern is destroyed.

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler import generate_preset_pass_manager
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION SWITCH ---
USE_SIMULATOR = False
# Set to False to run on real hardware
# ----------------------------

# 1. Define the backend based on the config variable
if USE_SIMULATOR:
    # Use the local Aer simulator
    backend = AerSimulator()
    print("Using local Aer simulator.")
else:
    # Use the IBM Quantum hardware via Qiskit Runtime
    service = QiskitRuntimeService() 
    # Select the least busy available QPU (Quantum Processing Unit)
    backend = service.least_busy(operational=True, simulator=False) 
    print(f"Using IBM hardware: {backend.name}")

sampler = Sampler(backend)
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

# Simulation parameters
shots_per_point = 256 # Reduced shots for speed in 2D grid
range_x = 10
range_y = 8
step_x  = 0.5
step_y  = 0.5
v_gauss = 8.0 # slot width
h_gauss = 4.0 # slot length

phi_param = Parameter('phi')

# Loop over every pixel coordinate on the virtual screen
# Define screen resolution (coarser grid for speed)
screen_y = np.arange(-range_y * np.pi, range_y * np.pi, step_y) 
screen_x = np.arange(-range_x, range_x, step_x)                 
num_y_points = len(screen_y)
num_x_points = len(screen_x)
# Initialize a 2D array to store the intensity data
intensity_map = np.zeros((num_y_points, num_x_points))

# Setup quantum circuit
phi_param_range = [-range_y * np.pi, range_y * np.pi, step_y]
phi_param = Parameter('phi')
qc = QuantumCircuit(1, 1)
qc.h(0)     # Prepare in superposition
# If you uncomment the next instruction, you simulate 
# 'which-path' measurment that destroys the interference pattern 
# qc.measure(0, 0)
qc.p(phi_param, 0) # Apply phase shift
qc.h(0)     # Apply second Hadamard gate to restore superposition
qc.measure_all() # Measure to obtain the interference pattern

# Prepare circuit to run
isa_circuit = pm.run(qc)

# Prepare parameters for batch
pubs = []
for value in screen_y:
    pubs.append((isa_circuit, [value], shots_per_point)) 

# Run the sampler and obtain result
job = sampler.run(pubs)
result = job.result()

# Process results clasically to build the 2D map
intensity_map = np.zeros((num_y_points, num_x_points))

for i, pub_result in enumerate(result):
    # Get probability of |0> each simulation
    counts = pub_result.data.meas.get_counts()
    prob_zero = counts.get('0', 0) / shots_per_point
    current_phi = screen_y[i]
    # Gaussian envelope for width (vertical position)
    vertical_envelope = np.exp(-0.3 * (current_phi)**2 / v_gauss**2)

    # Gaussian envelope for length (horizontal position)
    for j, x_pos in enumerate(screen_x):
        horizontal_envelope = np.exp(-0.3 * (x_pos)**2 / h_gauss**2)
        # Apply envelopes 
        intensity_map[i, j] = prob_zero * horizontal_envelope * vertical_envelope

# Plot the 2D interference pattern
plt.figure(figsize=(4, 4))
plt.imshow(intensity_map, extent=[-range_x, range_x, -range_y * np.pi, range_y * np.pi])
plt.xlabel('Horizontal Position (X)')
plt.ylabel('Vertical Position (Y))')
plt.title('Double-Slit Interference Pattern')
plt.colorbar(label='Intensity/Probability')
plt.show()
