#%%
from copy import deepcopy
from qiskit.providers.fake_provider import ConfigurableFakeBackend
from qiskit.providers.fake_provider import FakeMumbai
from qiskit.providers.aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.device.parameters import thermal_relaxation_values, readout_error_values
from qiskit import transpile, QuantumCircuit

def FakeSycamore25():
    n_qubits = 25

    cmap_sycamore_25 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 9], [9, 14], [14, 19], [19, 24], [24, 23], [23, 18], [18, 13], [13, 8], [8, 7], [7, 6], [6, 5], [5, 10], [10, 15], [15, 20], [20, 21], [21, 16], [11, 16], [0, 5], [1, 6], [6, 11], [11, 12], [12, 7], [7, 2], [3, 8], [8, 9], [14, 13], [13, 12], [12, 17], [18, 17], [16, 17], [17, 22], [23, 22], [22, 21], [10, 11], [16, 15], [18, 19]]
    cmap_sycamore_53 = [[0, 5], [5, 1], [1, 6], [6, 2], [2, 7], [7, 14], [14, 8], [8, 3], [3, 9], [9, 4], [4, 10], [10, 16], [16, 9], [9, 15], [15, 8], [5, 11], [11, 17], [17, 23], [23, 29], [29, 35], [35, 41], [41, 47], [41, 48], [48, 42], [42, 49], [49, 43], [43, 50], [50, 44], [44, 51], [51, 45], [45, 52], [52, 46], [46, 40], [40, 45], [45, 39], [39, 44], [44, 38], [38, 43], [43, 37], [37, 42], [42, 36], [36, 41], [36, 29], [29, 24], [24, 17], [17, 12], [12, 5], [12, 6], [6, 13], [13, 7], [12, 18], [18, 13], [13, 19], [19, 14], [14, 20], [20, 15], [15, 21], [21, 16], [16, 22], [22, 28], [28, 21], [21, 27], [27, 20], [20, 26], [26, 19], [19, 25], [25, 18], [18, 24], [24, 30], [30, 25], [25, 31], [31, 26], [26, 32], [32, 27], [27, 33], [33, 28], [28, 34], [34, 40], [40, 33], [33, 39], [39, 32], [32, 38], [38, 31], [31, 37], [37, 30], [30, 36]]

    ibm_device_backend = FakeMumbai()

    qubit_properties_backend = ibm_device_backend.properties()
    single_qubit_gates = set(ibm_device_backend.configuration().basis_gates).intersection(NoiseModel()._1qubit_instructions)
    single_qubit_gates.add("reset")
    # single_qubit_gates.add("measure")

    qubit_t1 = [item[0] for q_index, item in enumerate(thermal_relaxation_values(qubit_properties_backend)) if q_index in set(range(25))]
    qubit_t2 = [item[1] for q_index, item in enumerate(thermal_relaxation_values(qubit_properties_backend)) if q_index in set(range(25))]
    qubit_frequency = [item[2] for q_index, item in enumerate(thermal_relaxation_values(qubit_properties_backend)) if q_index in set(range(25))]
    qubit_readout_error = [item[0] for q_index, item in enumerate(readout_error_values(qubit_properties_backend)) if q_index in set(range(25))]
    qubit_readout_length = [item[1]["readout_length"] for item in qubit_properties_backend._qubits.items()]
    
    basis_gates = ibm_device_backend.configuration().basis_gates

    device_backend = ConfigurableFakeBackend(name="FakeSycamore", n_qubits=n_qubits, version=1, 
                                             coupling_map=cmap_sycamore_25, basis_gates=basis_gates, 
                                             qubit_t1=qubit_t1, qubit_t2=qubit_t2,
                                             qubit_frequency=qubit_frequency, 
                                             qubit_readout_error=qubit_readout_error,
                                             single_qubit_gates=single_qubit_gates, 
                                             dt=None)

    for q, props in device_backend._properties._qubits.items():
        if "readout_length" not in props and q in set(range(n_qubits)):
            props["readout_length"] = qubit_readout_length[q]
    for g, g_props in deepcopy(device_backend._properties._gates).items():
        for q, q_props in g_props.items():
            if len(q) > 1:
                device_backend._properties._gates[g][(q[1], q[0])] = q_props

    return device_backend


device_backend = FakeSycamore25()
noise_model = NoiseModel.from_backend(device_backend)

circ = QuantumCircuit(25, 25)
circ.x(range(25))
circ.measure(range(25), range(25))
circ.name = "Google Experiment 25 qubits"

# Transpile the circuits at the start to not reapeat transpilation at every injection campaing
t_circ = transpile(circ, device_backend, scheduling_method='asap',
            initial_layout=list(range(len(circ.qubits))), seed_transpiler=42)

qasm_sim = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
try:
    qasm_sim.set_options(device='GPU')
except:
    pass
probs = qasm_sim.run(t_circ, shots=1024).result()
print(probs)

# %%
