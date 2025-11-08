# Schrodinger Cat Simulation with Decoherence on the Cat Qubit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Kraus, DensityMatrix, partial_trace
import numpy as np
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


# --- Parameters ---
num_steps = 1000
gamma_step = 0.01
cumulative_gamma = np.linspace(gamma_step, num_steps * gamma_step, num_steps)
cat_probs = []

# --- Kraus operators for amplitude damping ---
def amplitude_damping_kraus(gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return Kraus([K0, K1])

# --- Build initial entangled state: (|0⟩|0⟩ + |1⟩|1⟩)/√2 ---
# Using DensityMatrix directly from the state vector for simplicity
# Qubit 0: Isotope Atom, Qubit 1: Cat
#initial_state_vector = (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1])) / np.sqrt(2)
# We start in with atom at 0.1 decayed and cat at 0.9 alive
initial_state_vector = np.sqrt(0.01)*np.kron([1, 0], [1, 0]) + np.sqrt(0.99)*np.kron([0, 1], [0, 1])
state = DensityMatrix(initial_state_vector)

# --- Simulate decoherence on the CAT qubit (qubit 1) over time ---
kraus_step = amplitude_damping_kraus(gamma_step)

for _ in range(num_steps):
    # Apply the fixed-step Kraus operator to the ayom qubit (index 0)
    state = state.evolve(kraus_step, qargs=[0])
    
    # Trace out the evolved state to get the cat's (index 1) reduced state
    cat_state = partial_trace(state, qargs=[1]) 
    
    # P(cat dead) is the probability of the |0> state in the cat's reduced density matrix
    # Accordingly, P(cat alive) =is the probability of the |1> state
    # |0> is the top-left element of the 2x2 matrix. |1> is bottom right.
    p_dead = np.real(cat_state.data[0, 0])
    cat_probs.append(p_dead)

# Plotting the results:
# Probability of the cat being dead (in state |0⟩) vs cumulative decoherence strength
# The horisontal axis can be interpreted as time under constant decoherence rate
# The dashed line indicates a 3-sigma threshold for practical purposes

plt.figure(figsize=(8, 5))
plt.plot(cumulative_gamma, cat_probs, label="P(cat dead) (Qubit 1)", color="purple")
plt.axhline(0.97, color='gray', linestyle='--', label="3σ threshold (~0.97)")
plt.xlabel("Cumulative Decoherence strength γ (∝ time/2)")
plt.ylabel("Probability cat is dead (state |0⟩)")
plt.title("Schrodinger Cat: Decoherence Over Time on Cat Qubit")
plt.legend()
plt.grid(True)
plt.show()
