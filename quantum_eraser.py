#===============================================================================
# Quantum Eraser Experiment using Qiskit
# This script simulates a quantum eraser setup where which-path information
# can be preserved or erased, demonstrating the effects on interference patterns.
# This quantum circuit is based on Kim et al.'s delayed choice quantum eraser 
# experiment.
# Signal and idler photons are entangled, and phase shifts are applied.
# The results are visualized to show the difference between preserved and erased
# which-path information.
#===============================================================================

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

class QuantumEraser:
    
    # Setup the experiment
    def setup(self, phi_values):
        self.phi_param = Parameter('phi')
        # Setup quantum circuit
        self.qr_signal = QuantumRegister(1, name = 'qr_signal')
        self.qr_idler = QuantumRegister(1, name = 'qr_idler')
        self.cr_combined = ClassicalRegister(2, name = 'cr_combined')
        self.qc_preserved = QuantumCircuit(self.qr_signal, self.qr_idler, self.cr_combined)
        self.experiment()
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_preserved = pm.run(self.qc_preserved)
        isa_erased = pm.run(self.qc_erased)

        # Prepare parameterized circuits for all phi values        
        self.pubs = []
        for value in phi_values:
            bound_preserved = isa_preserved.assign_parameters({self.phi_param: value})
            self.pubs.append(bound_preserved)
            bound_erased = isa_erased.assign_parameters({self.phi_param: value})
            self.pubs.append(bound_erased)
        # Setup sampler
        return Sampler(backend)
        
        
    # Quantum circuit for the quantum eraser experiment
    def experiment(self):
        # Prepare entangled state with phase shift
        self.qc_preserved.h(self.qr_signal)
        self.qc_preserved.cx(self.qr_signal, self.qr_idler)
        self.qc_preserved.rz(self.phi_param, self.qr_signal)
  
        # Copy and modify for erasure branch
        # Which pass information erased by applying Hadamard gates
        self.qc_erased = self.qc_preserved.copy()
        self.qc_erased.h(self.qr_signal[0])
        self.qc_erased.h(self.qr_idler[0])

        # Measure
        self.qc_preserved.measure(
            [self.qr_signal[0], self.qr_idler[0]],
            [self.cr_combined[0], self.cr_combined[1]])
        self.qc_erased.measure(
            [self.qr_signal[0], self.qr_idler[0]],
            [self.cr_combined[0], self.cr_combined[1]])
        

    # Run the simulation
    def run_experiment(self, phi_values, num_shots):
        sampler = self.setup(phi_values)
        job = sampler.run(self.pubs, shots=num_shots)
        result = job.result()
        return result

class Data:
    def __init__(self, phi_values, result):
        # Filter and store results
        self.phi_values = phi_values
        self.counts_combined = []
        self.counts_preserved_00 = []
        self.counts_preserved_01 = []
        self.counts_erased_00 = []
        self.counts_erased_01 = []
        # Filter out data from sampler results
        for i, pub_result in enumerate(result):
            # Sampler combines results, so even indices are 'preserved' and odd are 'erased'
            combined_bit_array = pub_result.data.cr_combined
            count_00 = combined_bit_array.get_counts().get('00', 0)
            count_01 = combined_bit_array.get_counts().get('01', 0)
            if i % 2 == 0 :
                self.counts_preserved_00.append(count_00)
                self.counts_preserved_01.append(count_01)
            else:
                self.counts_erased_00.append(count_00)
                self.counts_erased_01.append(count_01)
        # Calculate combined counts
        for i in range(len(self.counts_preserved_00)):   
            total = (self.counts_preserved_00[i] + self.counts_preserved_01[i] +
                     self.counts_erased_00[i] + self.counts_erased_01[i])
            self.counts_combined.append(total)

class Plotter:
    @staticmethod
    def plot_results(data):
        # Create subplots
        _, (totals, plt1, plt2) = plt.subplots(nrows=3, figsize=(8, 6), sharex=True)
        
        # Plot total counts
        totals.plot(data.phi_values, data.counts_combined, label='Total', color='green')
        totals.set_title('Total Counts Across All Measurements')
        totals.set_ylabel('Counts')
        totals.legend(loc='upper right')
        totals.grid(True)
        
        # Plot preserved which-path information
        plt1.plot(data.phi_values, data.counts_preserved_00, marker='o', label='00', color='red')
        plt1.plot(data.phi_values, data.counts_preserved_01, marker='o', label='01', color='blue')
        plt1.set_title('With Which-Path Information Preserved')
        plt1.set_ylabel('Counts')
        plt1.legend(loc='center right')
        plt1.grid(True)

        # Plot erased which-path information
        plt2.plot(data.phi_values, data.counts_erased_00, marker='o', label='00', color='red')
        plt2.plot(data.phi_values, data.counts_erased_01, marker='o', label='01', color='blue')
        plt2.set_title('With Which-Path Information Erased')
        plt2.set_xlabel('Phase Shift (radians)')
        plt2.set_ylabel('Counts')
        plt2.legend(loc='center right')
        plt2.grid(True)
        
        # Finalize and show plot
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    # Define parameters
    phi_values = np.linspace(0, 2 * np.pi, 100)
    num_shots = 512

    # Initialize Quantum Eraser experiment
    qe = QuantumEraser()
    # Run experiment
    result = qe.run_experiment(phi_values, num_shots=num_shots)
    # Process results
    data = Data(phi_values, result)
    # Plot results
    Plotter.plot_results(data)