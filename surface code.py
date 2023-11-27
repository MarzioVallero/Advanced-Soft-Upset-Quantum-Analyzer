from copy import deepcopy
from objprint import op
import numpy as np
import json
from qiskit import transpile, execute, Aer, QuantumRegister, QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.fake_provider import FakeJakarta
from qiskit_aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator

# An output bistring is considered correct if it starts with 1
def percentage_correct_outputs(counts):
    corrected = 0
    total = 0
    for basis, count in counts.items():
        if basis.startswith("1"):
            corrected += count
        total += count

    return corrected / total

# Define a dummy inj_gate to be used as an anchor for the NoiseModel
gate_qc = QuantumCircuit(1, name='id')
gate_qc.id(0)
inj_gate = gate_qc.to_gate()
inj_gate.label = "inj_gate"
inj_gate.name = "inj_gate"
inj_gate.duration = 0.0

# Repetition qubit surface code
from qtcodes import RepetitionQubit
qubit = RepetitionQubit({"d":3},"t")
qubit.reset_z()
# # Add dummy custom identity gate that performs injection
# # If you do it at the logical level, the dummy identity gat gets removed by the transpiler
# # Solution: add the custom gate to the device basis_gates list before transpiling
# for i in range(len(qubit.circ.qubits)):
#     qubit.circ.append(inj_gate, [i])
qubit.stabilize()
qubit.x()
qubit.stabilize()
qubit.readout_z()

circ = qubit.circ

# Minimal working example circuit
# circ = QuantumCircuit(1, 1)
# # circ.append(inj_gate, [0])
# circ.id([0])
# circ.measure([0], [0])
# print(circ.draw(fold=-1))

device_backend = FakeJakarta()

# qasm_sim = Aer.get_backend('qasm_simulator')
qasm_sim = AerSimulator()
noise_model_backend = NoiseModel.from_backend(device_backend)
basis_gates = noise_model_backend.basis_gates
coupling_map = device_backend.configuration().coupling_map
basis_gates = device_backend.configuration().basis_gates

# Add the dummy inj_gate and the id to the basis gates of the device
# basis_gates.append("inj_gate")
basis_gates.append("id")

# Get the transpiled version of the circuit, if needed
t_circ = transpile(
    circ,
    device_backend,
    scheduling_method='asap',
    initial_layout=list(range(len(circ.qubits))),
    seed_transpiler=42,
    basis_gates=basis_gates
)

# t_circ = circ

print(t_circ.draw())

# Execute the circuit WITHOUT noise
probs_ideal = execute(t_circ, qasm_sim, coupling_map=coupling_map, shots=1024)
# Execute the circuit WITH noise
probs_noisy = execute(t_circ, qasm_sim, noise_model=noise_model_backend, coupling_map=coupling_map, shots=1024)

print(f"Ideal counts: {percentage_correct_outputs(probs_ideal.result().get_counts())}")
print(f"Noisy counts: {percentage_correct_outputs(probs_noisy.result().get_counts())}")

# Create a transient error syndrome
from qiskit.providers.aer.noise import reset_error, pauli_error
transient_error = reset_error(0.0, 1.0)
# transient_error = pauli_error([('X', 1.0), ('I', 0.0)])

# The noisy AerSimulator doesn't recognise the inj_gate.
# Possible solution: map the error to an Identity Gate, wihtout using a custom gate
# Create a column of id gates as injection points, then append the rest of the transpiled circuit afterwards
new_t_circ = deepcopy(t_circ)
new_t_circ.clear()
for i in range(len(circ.qubits)):
    new_t_circ.id([i])
new_t_circ.compose(t_circ, inplace=True)
t_circ = new_t_circ

# Execute the circuit WITHOUT noise, but with the transient_fault, with respect to each qubit
for inj_index in range(len(circ.qubits)):
    noise_model_only_transient = NoiseModel()
    noise_model_only_transient.add_quantum_error(transient_error, instructions=["id"], qubits=[inj_index])

    qasm_sim = AerSimulator(noise_model=noise_model_only_transient, basis_gates=basis_gates)

    # qasm_sim = AerSimulator(noise_model=noise_model_only_transient,
    #         coupling_map=coupling_map,
    #         basis_gates=basis_gates)
    # t_circ = transpile(circ, qasm_sim)
    
    # print(t_circ.draw())

    probs_noisy_only_transient = qasm_sim.run(t_circ)

    # probs_noisy_only_transient = execute(t_circ, qasm_sim, noise_model=noise_model_only_transient, coupling_map=coupling_map, shots=1024)

    print(f"Only transient on qubit {inj_index} counts: {percentage_correct_outputs(probs_noisy_only_transient.result().get_counts())}\n{probs_noisy_only_transient.result().get_counts()}")

# Execute the circuit WITH noise and with the transient_fault
for inj_index in range(len(circ.qubits)):
    noise_model_with_transient = NoiseModel.from_backend(device_backend)
    noise_model_with_transient.add_quantum_error(transient_error, instructions=["id"], qubits=[inj_index])

    qasm_sim = AerSimulator(noise_model=noise_model_with_transient, basis_gates=basis_gates)

    # qasm_sim = AerSimulator(noise_model=noise_model_with_transient,
    #         coupling_map=coupling_map,
    #         basis_gates=basis_gates)
    # t_circ = transpile(circ, qasm_sim)
    
    # print(t_circ.draw())

    probs_noisy_with_transient = qasm_sim.run(t_circ)

    # probs_noisy_with_transient = execute(t_circ, qasm_sim, noise_model=noise_model_with_transient, coupling_map=coupling_map, shots=1024)

    print(f"Noisy with transient on qubit {inj_index} counts: {percentage_correct_outputs(probs_noisy_with_transient.result().get_counts())}")