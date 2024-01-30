# %% Test different noise models
from injector_par import *

nm = bitphase_flip_noise_model(0.1, 1)
print(nm._local_quantum_errors["x"][(0,)])
nm.add_quantum_error(reset_to_zero(1), "x", [0])
print(nm._local_quantum_errors["x"][(0,)])

# %% Test transpilation and simulation time for different topologies
from injector_par import *
from qtcodes import RepetitionQubit
from qiskit.transpiler import PassManager

ts = time()
log(f"Job started at {datetime.fromtimestamp(ts)}.")
# Repetition qubit surface code
d = 5
T = 1
repetition_q = RepetitionQubit({"d":d},"t")
repetition_q.reset_z()
repetition_q.stabilize()
repetition_q.x()
repetition_q.stabilize()
repetition_q.readout_z()
repetition_q.circ.name = "Repetition Qubit"

# Backend and circuit selection
target_circuit = repetition_q.circ
qtcodes_circ = repetition_q

line_edge_list = [[i, i+1] for i in range(25)]
complete_edge_list = [[i, j] for i in range(25) for j in range(i) if i != j]
mesh_edge_list = [[(i*5)+j, (i*5)+j+1] for i in range(5) for j in range(4)] +  [[((i)*5)+j, ((i+1)*5)+j] for i in range(4) for j in range(5)]
topologies = {"linear":line_edge_list, "complete":complete_edge_list, "square_mesh":mesh_edge_list}
for topology_name, topology in topologies.items():
    ts = time()
    device_backend = CustomBackend(n_qubits=target_circuit.num_qubits, coupling_map=topology)

    transpiled_circuit = transpile(target_circuit, device_backend, scheduling_method='asap',
            initial_layout=list(range(len(target_circuit.qubits))), seed_transpiler=42)
    log(f"Transpilation done for {topology_name} ({time() - ts} elapsed)")

    noise_model_simulator = bitphase_flip_noise_model(0.01, target_circuit.num_qubits)
    for target_name, target in {"original":target_circuit, "transpiled_circuit":transpiled_circuit}.items():
        ts = time()
        sim = AerSimulator(noise_model=deepcopy(noise_model_simulator), basis_gates=noise_model_simulator.basis_gates)
        try:
            sim.set_options(device='GPU')
            counts = sim.run(target, shots=100000).result().get_counts()
        except RuntimeError:
            sim.set_options(device='CPU')
            counts = sim.run(target, shots=100000).result().get_counts()
        log(f"counts {target_name}: {counts}")
        log(f"Simulation done for {topology_name} ({time() - ts} elapsed)")
    log("-----------------------------------------------------------------------------------")

# %% Plot topologies from IBM devices
from injector_par import *
from qiskit.providers.fake_provider import FakeProvider
from qiskit.visualization import plot_coupling_map

# Filter out all "small" backends (less than size qubits)
size = 18
backends = {backend.name():backend for backend in FakeProvider().backends() if backend.configuration().n_qubits > size}
qubit_range = set(range(30))
graphs = {}
for name, backend in backends.items():
    isomorphic = False
    coupling_map = backend.configuration().coupling_map
    for edge in deepcopy(coupling_map):
        if (edge[0] not in qubit_range) or (edge[1] not in qubit_range):
            coupling_map.remove(edge)
    G = nx.Graph(coupling_map)

    for graph in graphs.values():
        if nx.is_isomorphic(G, graph[1]):
            isomorphic = True
            break
    if isomorphic or not nx.is_connected(G):
        continue

    graphs[name] = (coupling_map, G)

nr = int(np.ceil(np.sqrt(len(graphs.values()))))
fig = plt.figure(figsize=(12*nr, 12*nr)); plt.clf()
fig, ax = plt.subplots(nr, nr, num=1)

for i, (name, (cm, graph)) in enumerate(graphs.items()):
    ix = np.unravel_index(i, ax.shape)
    pos = nx.spring_layout(graph, weight=0.0001, iterations=1000, threshold=0.00001)
    nx.draw_networkx(graph, pos=pos, with_labels=True, ax=ax[ix])
    ax[ix].set_title(name, fontsize=30)
plt.show()

#%% Get subgraphs of overall used qubits after the transpilation process
from injector_par import *
from qtcodes import XXZZQubit, RepetitionQubit
from qiskit.converters import circuit_to_dag

xxzzd3 = XXZZQubit({'d':3})
xxzzd3.stabilize()
xxzzd3.stabilize()
xxzzd3.x()
xxzzd3.stabilize()
xxzzd3.readout_z()
xxzzd3.circ.name = "XXZZ d3 Qubit"

target_circuit = xxzzd3.circ

transpiled_graphs = {}
for name, (cm, graph) in graphs.items():
    log(f"Transpiling with {name}'s topology")
    for optimization in range(4):
        ts = time()
        device_backend = CustomBackend(n_qubits=len(graph.nodes), coupling_map=cm)
        transpiled_circuit = transpile(target_circuit, device_backend, scheduling_method='asap',
                                optimization_level=optimization,
                                # initial_layout=list(range(len(target_circuit.qubits))),
                                seed_transpiler=42)
        active_qubits = []
        operations = list(reversed(list(enumerate(transpiled_circuit.data))))
        for idx, _instruction in operations:
            if _instruction.operation.name not in ["delay", "barrier"]:
                for _qubit in _instruction.qubits:
                    if _qubit.index not in active_qubits:
                        active_qubits.append(_qubit.index)
        idle_qubits = [ q for q in range(len(graph.nodes)) if q not in active_qubits ]
        reduced_graph = deepcopy(graph)
        reduced_graph.remove_nodes_from(idle_qubits)
        transpiled_graphs[f"{name} ol={optimization}"] = (cm, reduced_graph)
        active_qubits.sort()
        log(f"      ol={optimization} with {len(operations)} ops done in {timedelta(seconds=time() - ts)}\n        active={active_qubits}\n    len(active)={len(active_qubits)}")
    log("\n")

nr = int(np.ceil(np.sqrt(len(transpiled_graphs.values()))))
fig = plt.figure(figsize=(12*4, 12*len(transpiled_graphs)/4)); plt.clf()
fig, ax = plt.subplots(len(transpiled_graphs)//4, 4, num=1)

for i, (name, (cm, graph)) in enumerate(transpiled_graphs.items()):
    ix = np.unravel_index(i, ax.shape)
    pos = nx.spring_layout(graph, weight=0.0001, iterations=1000, threshold=0.00001)
    nx.draw_networkx(graph, pos=pos, with_labels=True, ax=ax[ix])
    ax[ix].set_title(name, fontsize=30)
plt.show()
# %%
