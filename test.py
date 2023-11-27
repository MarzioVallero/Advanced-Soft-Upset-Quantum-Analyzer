from mqt import ddsim
import numpy as np
import json
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from quantum_gates.utilities import DeviceParameters
from qiskit.providers.fake_provider import FakeManila

circ = QuantumCircuit(2,2)
circ.h(0)
circ.cx(0,1)
circ.barrier(range(2))
circ.measure(range(2),range(2))
circ.draw()

config = {
    "backend": {
        "hub": "ibm-q",
        "group": "open",
        "project": "main",
        "device_name": "ibmq_manila"
    },
    "run": {
        "shots": 1000,
        "qubits_layout": [0, 1],
        "psi0": [1, 0, 0, 0]
    }
}

backend_config = config["backend"]
backend = FakeManila()
run_config = config["run"]

qubits_layout = run_config["qubits_layout"]
device_param = DeviceParameters(qubits_layout)
device_param.load_from_backend(backend)
device_param_lookup = device_param.__dict__()

sim = ddsim.DDSIMProvider().get_backend("qasm_simulator")

t_circ = transpile(
    circ,
    backend,
    scheduling_method='asap',
    initial_layout=qubits_layout,
    seed_transpiler=42,
    basis_gates=None
)

print(t_circ.draw(fold=-1))

probs = sim.run(quantum_circuits=t_circ, 
    qubits_layout=qubits_layout, 
    psi0=np.array(run_config["psi0"]), 
    shots=run_config["shots"], 
    device_param=device_param_lookup,
    nqubit=2).result()

print(probs)