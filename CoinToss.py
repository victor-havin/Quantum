# CoinToss.py
# My version of a "Hello Quantum World" program, 
# a quantum coin toss simulation.

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Quantum circuit with 1 qubit and 1 classical bit
coin = QuantumCircuit(1, 1)

# Hadamard gate to create superposition
coin.h(0)

# Measure:
# Collapse the wavefunction or split the universe.
# Depends on your interpretation of quantum mechanics :)
coin.measure(0, 0)

# Simulate the circuit
backend = Aer.get_backend('qasm_simulator')
job = execute(coin, backend, shots=100)
result = job.result()

# Get the counts and plot the histogram
counts = result.get_counts(coin)
plot_histogram(counts)
print(counts)
plt.title("Quantum Coin Toss")
plt.show()