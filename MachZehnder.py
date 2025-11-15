# Double slit simulation
# Works in 2 stages:
# 1. Obtain interference pattern from a single qubit with phase modulation
# 2. Postprocess the result classically to simulate diffraction
# You can apply 'which-path' measurement, the interference pattern is destroyed.

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import Parameter 
import numpy as np
import matplotlib.pyplot as plt

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

def MachZehnder(phi_values, num_shots, which_path=False):
    # Setup quantum circuit
    qr_beam1 = QuantumRegister(1, name = 'qr_beam1')
    qr_beam2 = QuantumRegister(1, name = 'qr_beam2')
    cr_combined = ClassicalRegister(2, name = 'cr_combined')

    qc = QuantumCircuit(qr_beam1, qr_beam2, cr_combined)
    phi_param = Parameter('phi')
    qc.h(qr_beam1)     # Prepare in superposition
    qc.cx(qr_beam1, qr_beam2)
    # 'which-path' measurment that destroys the interference pattern 
    if(which_path):
        qc.measure(qr_beam1, cr_combined[0])
    qc.rz(phi_param, qr_beam1) # Apply phase shift
    qc.h(qr_beam1)     # restore superposition on beam1
    qc.h(qr_beam2)     # restore superposition on beam2
    qc.measure([qr_beam1[0], qr_beam2[0]], [cr_combined[0], cr_combined[1]]) # Measure to obtain the interference pattern

    # Run the simulation
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuit = pm.run(qc)
    pubs = []
    for value in phi_values:
        pubs.append((isa_circuit, [value], num_shots)) 
    sampler = Sampler(backend)
    job = sampler.run(pubs)
    result = job.result()
    all_00_counts = []

    # process results
    for pub_result in result:
        combined_bit_array = pub_result.data.cr_combined
        count_00 = combined_bit_array.get_counts().get('00', 0)
        all_00_counts.append(count_00)
    return np.array(all_00_counts)

phis = np.linspace(0, 2 * np.pi, 256)
beam1_no_wp = []
beam1_with_wp = []

counts_no_wp = MachZehnder(phi_values=phis, num_shots=256, which_path=False)
counts_with_wp = MachZehnder(phi_values=phis, num_shots=256, which_path=True)


# Plotting
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)

ax1.plot(phis, counts_no_wp, label="No which-path", color='blue')
ax1.set_ylabel("Beam brightness")
ax1.set_title("Without Which-Path Measured")
ax1.grid(True)

ax2.plot(phis, counts_with_wp, label="With which-path", color='red')
ax2.set_xlabel("Phase φ (radians)")
ax2.set_ylabel("Beam brightness")
ax2.set_title("With Which-Path Measured)")
ax2.grid(True)

plt.tight_layout()
plt.show()

