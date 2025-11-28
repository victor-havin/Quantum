# CoinToss.py
# My version of a "Hello Quantum World" program, 
# a quantum coin toss simulation.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# --- CONFIGURATION SWITCH ---
USE_SIMULATOR = True
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

# Quantum circuit with 1 qubit and 1 classical bit
coin = QuantumCircuit(1, 1)

# Hadamard gate to create superposition
coin.h(0)

# Measure:
# Collapse the wavefunction or split the universe.
# Depends on your interpretation of quantum mechanics :)
coin.measure(0, 0)

job = backend.run(coin, shots=10000)
result = job.result()

# Get the counts and plot the histogram
counts = result.get_counts(coin)
plot_histogram(counts)
print(counts)
plt.title("Quantum Coin Toss")
plt.show()