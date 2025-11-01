# 1. Import necessary libraries
import numpy as np
from qiskit import QuantumCircuit,execute
from qiskit_aer import Aer 
from qiskit.visualization import plot_histogram, plot_state_qsphere
import matplotlib.pyplot as plt

def flip(on: bool):
    # 2. Create a quantum circuit with one qubit and one classical bit
    qc = QuantumCircuit(2, 2)

    # 3. Apply a Hadamard gate to put the qubit into an equal superposition state (|psi1>)
    #    This creates |psi1> = (1/sqrt(2)) * (|0> + |1>)
    qc.h(0)
    qc.h(1)

    # 4. Apply a Z-gate to the qubit.
    #    The Z-gate applies a phase shift of -1 (or e^i*pi) to the |1> state.
    #    The state becomes (1/sqrt(2)) * (|0> - |1>).
    #    This is |psi2> = (1/sqrt(2)) * (|0> + e^i*pi*|1>)
    qc.z(0)
    qc.cx(0,1)
    # 5. Apply another Hadamard gate.
    #    This second Hadamard gate performs the addition/interference.
    #    It sends the (|0> - |1>) state back to the |1> state.
    #    The two superpositions have now cancelled out, leaving the qubit in the |1> state.
    #    If the Z-gate was not present, two Hadamards would return the qubit to the |0> state.
    qc.h(0)
    qc.h(1)
    if not on:
        # Now the key part: to flip the outcome, 
        # we apply a rotation around the Y-axis by π (180 degrees).
        qc.ry(3.1415, 0)  

    # 6. Measure the qubits
    qc.measure([0, 1], [0, 1])

    # 7. Draw the circuit
    print("The quantum circuit:")
    print(qc.draw())

    # 8. Run the circuit on a simulator and get the results
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    filtered_counts = {
        'off': counts.get('00', 0),
        'on' : counts.get('01', 0)
    }
    # 9. Print and plot the results
    print("\nMeasurement counts:", counts)
    plot_histogram(filtered_counts)
    plt.title("Measurement Results (Filtered)")
    plt.show()
    # 10. Optional: Visualize the final state on a Q-sphere
    final_state_sim = Aer.get_backend('statevector_simulator')
    job_state = execute(qc, final_state_sim)
    result_state = job_state.result()
    statevector = result_state.get_statevector()  
    print("\nFinal statevector:", statevector)

flip(True);
flip(False);