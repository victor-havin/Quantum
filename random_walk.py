from collections import Counter
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization import plot_histogram, plot_state_qsphere
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

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

nqubits = 16
def run_random_walk(entangled=False, shots=2048):
    qc = QuantumCircuit(nqubits, nqubits)

    # Superposition
    qc.h(range(nqubits))

    # Optional entanglement
    if entangled:
        for i in range(nqubits - 1):
            qc.cx(i, i+1)
        for n in range(nqubits//2):
            qc.ry(math.pi/4, n)
    ''' Debug: Show partial trace of first half qubits 
    state = Statevector.from_instruction(qc)
    rho = partial_trace(state, range(nqubits//2))
    df = pd.DataFrame(rho.data.real)
    print(f"Partial trace of first half qubits:\n{df}")    
    '''
    qc.measure(range(6), range(6))

    # Execute
    backend = Aer.get_backend('qasm_simulator')
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Displacement histogram
    histogram = Counter()
    for bitstring, freq in counts.items():
        displacement = sum(1 if bit == '1' else -1 for bit in bitstring)
        histogram[displacement] += freq

    return histogram

# Run both versions
hist_unentangled = run_random_walk(entangled=False)
hist_entangled = run_random_walk(entangled=True)

print(hist_unentangled) 
print(hist_entangled)

# Plotting
def plot_histogram(hist, title, color):
    xs = sorted(hist.keys())
    ys = [hist[x] for x in xs]
    plt.bar(xs, ys, color=color, alpha=0.6, label=title)

plt.figure(figsize=(nqubits, 6))
plot_histogram(hist_unentangled, "Unentangled Walk", "blue")
plot_histogram(hist_entangled, "Entangled Walk", "red")
plt.xlabel("Displacement")
plt.ylabel("Frequency")
plt.title("Quantum Random Walk: Entangled vs Unentangled")
plt.legend()
plt.grid(True)
plt.show()
