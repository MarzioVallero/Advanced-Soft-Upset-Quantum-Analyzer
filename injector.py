#%%
from copy import deepcopy
from functools import partial
import pandas as pd
import pickle, bz2
import re
from ast import literal_eval as make_tuple
from time import time, sleep
from datetime import datetime, timedelta
from math import exp, inf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from os.path import isdir, dirname
from os import mkdir
from subprocess import Popen, PIPE, STDOUT
import tqdm
from mycolorpy import colorlist as mcp
from scipy.interpolate import griddata
from qiskit import transpile, QuantumCircuit
from qiskit import Aer
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import  FakeBrooklyn, FakeProvider
from qiskit_aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit_aer.noise.device.parameters import thermal_relaxation_values, readout_error_values
from qiskit.providers.fake_provider import ConfigurableFakeBackend
from qiskit.providers.aer.noise import reset_error, pauli_error
from qiskit.circuit.library import IGate
from qiskit.quantum_info.analysis import hellinger_distance, hellinger_fidelity
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

file_logging = False
logging_filename = "./asuqa.log"
console_logging = True
#TODO: move to CONST file
line_edge_list = [[i, i+1] for i in range(30)]
complete_edge_list = [[i, j] for i in range(30) for j in range(i) if i != j]
mesh_edge_list = [[(i*5)+j, (i*5)+j+1] for i in range(6) for j in range(4)] +  [[((i)*5)+j, ((i+1)*5)+j] for i in range(5) for j in range(5)]

def CustomBackend(active_qubits=list(range(30)), coupling_map=mesh_edge_list, backend_name="Mesh"):
    """Custom backend that uses the noise profile of FakeBrooklyn and maps it to a custom coupling map. 
    Accepts a maximum of 30 qubits, as that is the simulation limit for the AerSimulator."""
    n_qubits = min(len(active_qubits), len(set(flatten(coupling_map))))
    if n_qubits > 30:
        log("No more than 30 qubits can be simulated at a time.")
        raise Exception
    qubit_remapper = {original:new for original, new in zip(active_qubits, range(n_qubits))}
    remapped_cm = []
    for edge in coupling_map:
        if edge[0] in active_qubits and edge[1] in active_qubits:
            remapped_cm.append([qubit_remapper[edge[0]], qubit_remapper[edge[1]]])
            # Transpiler uses the coupling map as a directed graph, so reverse edge is needed
            remapped_cm.append([qubit_remapper[edge[1]], qubit_remapper[edge[0]]])

    G = nx.Graph(remapped_cm)
    ibm_device_backend = FakeBrooklyn()

    qubit_properties_backend = ibm_device_backend.properties()
    # single_qubit_gates = set(ibm_device_backend.configuration().basis_gates).intersection(NoiseModel()._1qubit_instructions)
    single_qubit_gates = set(get_standard_gate_name_mapping().keys())
    single_qubit_gates.add("reset")
    # single_qubit_gates.add("measure")

    qubit_t1 = [item[0] for q_index, item in enumerate(thermal_relaxation_values(qubit_properties_backend)) if q_index in range(n_qubits)]
    qubit_t2 = [item[1] for q_index, item in enumerate(thermal_relaxation_values(qubit_properties_backend)) if q_index in range(n_qubits)]
    qubit_frequency = [item[2] for q_index, item in enumerate(thermal_relaxation_values(qubit_properties_backend)) if q_index in range(n_qubits)]
    qubit_readout_error = [item[0] for q_index, item in enumerate(readout_error_values(qubit_properties_backend)) if q_index in range(n_qubits)]
    qubit_readout_length = [value["readout_length"] for key, value in qubit_properties_backend._qubits.items() if key in range(n_qubits)]
    
    # basis_gates = ibm_device_backend.configuration().basis_gates
    basis_gates = single_qubit_gates

    device_backend = ConfigurableFakeBackend(name=backend_name, n_qubits=n_qubits, version=1, 
                                             coupling_map=remapped_cm, basis_gates=basis_gates, 
                                             qubit_t1=qubit_t1, qubit_t2=qubit_t2,
                                             qubit_frequency=qubit_frequency, 
                                             qubit_readout_error=qubit_readout_error,
                                             single_qubit_gates=single_qubit_gates, 
                                             dt=None)

    for q, props in deepcopy(device_backend._properties._qubits).items():
        if "readout_length" not in props and q in range(n_qubits):
            props["readout_length"] = qubit_readout_length[q]
    for g, g_props in deepcopy(device_backend._properties._gates).items():
        for q, q_props in g_props.items():
            if g in NoiseModel()._2qubit_instructions:
                for neighbour in nx.all_neighbors(G, q[0]):
                    device_backend._properties._gates[g][(q[0], neighbour)] = q_props
                    device_backend._properties._gates[g][(neighbour, q[0])] = q_props
                del device_backend._properties._gates[g][(q[0], )]

    return device_backend

def get_coupling_maps(min_size, qubit_range=set(range(30))):
    """Returns a dictionary{name: coupling_map} containing all the coupling maps from the IBM backends according to the specified parameters, 
    plus three ideal 30-qubit coupling maps (linear, complete, mesh)."""
    graphs = {"linear":(line_edge_list, nx.Graph(line_edge_list)),
              "complete":(complete_edge_list, nx.Graph(complete_edge_list)),
              "square_mesh":(mesh_edge_list, nx.Graph(mesh_edge_list))}

    backends = {backend.name():backend for backend in FakeProvider().backends() if backend.configuration().n_qubits > min_size}
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
        # If isomorphic graph has already been selected or the output graph is not connected, skip
        if isomorphic or not nx.is_connected(G):
            continue

        graphs[name] = (coupling_map, G)

    return {name:cm for name, (cm, g) in graphs.items()}

def get_active_qubits(circuit):
    """Get qubit lines from a transpiled quantum circuit that are "logically" active, i.e. used at least once in computation.
    Throws an Exception if the circuit is not scheduled, i.e. if it has not been transpiled."""
    active_qubits = []
    operations = list(reversed(list(enumerate(circuit.data))))
    for idx, _instruction in operations:
        if _instruction.operation.name not in ["delay", "barrier"]:
            for _qubit in _instruction.qubits:
                if _qubit.index not in active_qubits:
                    active_qubits.append(_qubit.index)
    active_qubits.sort()
    return active_qubits

def filter_coupling_map(coupling_map, active_qubits):
    """Remove from all qubits not in active_qubits from <coupling_map>."""
    reduced_coupling_map = []
    for edge in coupling_map:
        if edge[0] in active_qubits and edge[1] in active_qubits:
            reduced_coupling_map.append(edge)
    return reduced_coupling_map

def bitphase_flip_noise_model(p_error, n_qubits):
    """Returns a simple noise model on all qubits in range(n_qubits) made of the independent composition of an X and a Z PauliError, both with probability p_error."""
    bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
    phase_flip = pauli_error([('Z', p_error), ('I', 1 - p_error)])
    bitphase_flip = bit_flip.compose(phase_flip)
    noise_model = NoiseModel()
    for q_index in range(n_qubits):
        noise_model.add_quantum_error(bitphase_flip, instructions=list(NoiseModel()._1qubit_instructions), qubits=[q_index])

    # "Do you think God stays in heaven because he, too, lives in fear of what he's created here on earth?"
    setattr(noise_model,"__name__", f"bitphase_flip_noise_model {p_error}")

    return noise_model

def flatten(container):
    """Yields the recursively flattened list of a multiply nested list."""
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def log(content):
    """Logging wrapper, can redirect both to stdout and a file."""
    if file_logging:
        fp = open(logging_filename, "a")
        fp.write(content+'\n')
        fp.flush()
        fp.close()
    if console_logging:
        print(content)

def plot_coupling_map(device_backend):
    """Extract the coupling map from a Qiskit backend object and plot its coupling map."""
    coupling_map = device_backend.configuration().coupling_map
    G = nx.Graph(coupling_map)
    try:
        pos = nx.spring_layout(G, weight=0.1, iterations=500, threshold=0.00001)
    except Exception:
        pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G, pos=pos, with_labels=True)
    plt.show()

# Define a custom transient error, parametrised by the probability p of being applied.
# Errors can be composed or tensored together to achieve more complex behaviours:
# - Composition: E(ρ)=E2(E1(ρ)) -> error = error1.compose(error2)
# - Tensor: E(ρ)=(E1⊗E2)(ρ) -> error error1.tensor(error2)
def reset_to_zero(p):
    return reset_error(p, 0.0)

# Define a custom damping function parametrised by the distance from the root impact qubit
def square_damping(depth=0):
    n = 1
    return n**2/((depth+n)**2)

def error_probability_decay(input, transient_duration_ns):
    # factor = 1/(4*1.0e6*25.0e6)
    factor = 10
    x = input/transient_duration_ns
    return exp(-(factor)*x)

def get_inj_gate(inj_duration=0.0):
    """Define a dummy inj_gate to be used as an anchor for the NoiseModel. Currently unused, as errors are applied to all gates."""
    inj_gate = IGate()
    inj_gate.label = "inj_gate"
    inj_gate.duration = inj_duration

    return inj_gate

def subprocess_pool(command, args_dict, max_processes):
    """Custom subprocess pool, of up to max_processes simultaneous workers. Queue elements follow the syntax of Popen(subprocess)."""
    output_list = []
    proc_list = []
    with tqdm.tqdm(total=len(args_dict.keys())) as pbar:
        for i, args in args_dict.items():
            p = Popen(command, bufsize=4096, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
            stdout_data = p.communicate(input=bz2.compress(pickle.dumps(args)))[0]
            proc_list.append((p, i))
            if len(proc_list) == max_processes:
                wait = True
                while wait:
                    done, (p,i), o = check_for_done(proc_list)
                    if done:
                        output_list.append((i, o))
                        proc_list.remove((p,i))
                        wait = False
                    else:
                        sleep(1)
                pbar.update()
        while len(proc_list) > 0:
            done, (p,i), o = check_for_done(proc_list)
            if done:
                output_list.append((i, o))
                proc_list.remove((p,i))
                pbar.update()
    
    return output_list

def check_for_done(l):
    """Check if a process in the supplied process list has terminated. Returns True and the index of the first process in the list that has terminated."""
    for p, i in l:
        if p.poll() is not None:
            raw_output = p.communicate()
            output = pickle.loads(bz2.decompress(raw_output[0]))
            return True, (p, i) , output
    return False, False, None

def spread_transient_error(noise_model, sssp_dict_of_qubit, root_inj_probability, spread_depth, transient_error_function, damping_function):
    """Recursively spread the transient error according to the sssp_dict_of_qubit retrieved from the coupling map of the device_backend and add it to the NoiseModel object."""
    single_qubits_injected = []
    for path in sssp_dict_of_qubit:
        if len(path)-1 <= spread_depth:
            transient_error = transient_error_function(root_inj_probability*damping_function(depth=len(path)-1))
            # To add an error to specific gate on the inj_gate, make sure to have prepended them in run_injection_campaign() !
            # noise_model.add_quantum_error(transient_error, instructions=["inj_gate"], qubits=[path[-1]])

            # Add error to all gates on specific qubit, avoid adding the error to the same signgle qubit twice
            if path[-1] not in single_qubits_injected:
                single_qubits_injected.append(path[-1])
                target_instructions = list(noise_model._1qubit_instructions.intersection(noise_model.basis_gates))
                target_instructions.append("inj_gate")
                noise_model.add_quantum_error(transient_error, instructions=target_instructions, qubits=[path[-1]], warnings=False)

            if len(path) >= 2:
                target_instructions_2q = list(noise_model._2qubit_instructions.intersection(noise_model.basis_gates))
                two_qubit_targets = (path[-2], path[-1])
                noise_model.add_quantum_error(transient_error.tensor(transient_error), instructions=target_instructions_2q, qubits=two_qubit_targets, warnings=False)
                inverted_two_qubit_targets = (path[-1], path[-2])
                noise_model.add_quantum_error(transient_error.tensor(transient_error), instructions=target_instructions_2q, qubits=inverted_two_qubit_targets, warnings=False)

    return noise_model

def error_spread_map(coupling_map, injection_point, spread_depth):
    """Returns a list of all the paths from <injection_point> to all other nodes up to distance <spread_depth>, allowing partially repeating paths so long as they differ in the second to last node in the path.
    This is needed in order to correctly address all error paths in a Mesh topology."""
    error_list = []
    G = nx.Graph(coupling_map)
    distances = list(nx.single_target_shortest_path_length(G, injection_point, cutoff=spread_depth))
    for n, d in distances:
        simple_paths = list(nx.all_simple_paths(G, source=injection_point, target=n, cutoff=d))
        last_edges = []
        for path in simple_paths:
            last_edge = (path[-1], path[-2])
            if last_edge not in last_edges:
                last_edges.append(last_edge)
                error_list.append(path)
        if injection_point == n:
            error_list.append([injection_point])

    return error_list

def check_if_sublist(source, sublist):
    """Recursively checks if <sublist> is contained in <source>."""
    if len(sublist) == 0:
        return True
    if len(source) == 0:
        return False
    if source[0] == sublist[0]:
        return check_if_sublist(source[1:], sublist[1:])
    else:
        return check_if_sublist(source[1:], sublist)

def merge_injection_paths(error_spread_map0, error_spread_map1):
    """Merges two <error_spread_map> lists of paths by keeping the shortest common subpaths among all paths.
    Works correctly only if the supplied error_spread_maps refer to adjacent nodes."""
    merge = []
    for source_path in error_spread_map0:
        merge.append(source_path)
        for subpath in error_spread_map1:
            if subpath not in merge:
                if check_if_sublist(source_path, subpath):
                    merge.append(subpath)
                    if len(source_path) > 2:
                        merge.remove(source_path)
                if len(subpath) == 1:
                    merge.append(subpath)
    return merge

def get_shot_execution_time_ns(circuit):
    """Return the execution time for a given quantum circuit in nanoseconds. The method has high precision on transpiled (scheduled) circuits, while it produces an estimate using static parameters for non transpiled ones."""
    basis_gates = list(NoiseModel()._1qubit_instructions)

    # Check if the circuit has been scheduled (transpiled) already
    try:
        min_time = []
        max_time = []
        for qubit in circuit.qubits:
            min_time.append(circuit.qubit_start_time(qubit))
            max_time.append(circuit.qubit_stop_time(qubit))
        shot_duration = max(max_time) - min(min_time)
    except Exception as e:
        single_gate_avg_duration = 5.0 #ns
        multi_gate_avg_duration = 20.0 #ns
        measure_duration = 13.0 #ns
        qubit_total_duration = [0] * circuit.width()
        for instruction in reversed(list((circuit.data))):
            for qubit in instruction.qubits:
                if instruction.operation.name == "barrier":
                    pass
                elif instruction.operation.name == "measure":
                    qubit_total_duration[qubit.index] += measure_duration
                elif len(instruction.qubits) == 1:
                    qubit_total_duration[qubit.index] += single_gate_avg_duration
                else:
                    qubit_total_duration[qubit.index] += multi_gate_avg_duration
        shot_duration = max(qubit_total_duration)   

    return shot_duration

def get_shots_and_probability_per_batch(circuit, n_quantised_steps, transient_error_duration_ns):
    """Given a quantum circuit's execution time, computes how many shots of the circuit can be carried out in <transient_error_duration_ns> time, 
    which is in turn divided by the <n_quantised_steps>, returning the number of shots performed for each time batch.
    The <error_probability_decay> is sampled and converted into a step function, returned as a list of probabilities associated to each time batch."""
    shot_time_per_circuit = get_shot_execution_time_ns(circuit)
    total_shots = transient_error_duration_ns // shot_time_per_circuit
    shots_per_time_batch = total_shots // n_quantised_steps
    
    probability_per_batch = []
    for batch in range(n_quantised_steps):
        time_step = batch*shots_per_time_batch*shot_time_per_circuit
        probability = error_probability_decay(time_step, transient_error_duration_ns)
        probability_per_batch.append(probability)

    return shots_per_time_batch, probability_per_batch

def listify(obj):
    """Return a list containing <obj> if the object itself is not a list, else return it unaltered."""
    if isinstance(obj, list):
        return obj
    else:
        return [obj]

def run_injection(circuit, injection_point=0, transient_error_function=reset_to_zero, root_inj_probability=1.0, time_step=0, spread_depth=0, damping_function=None, device_backend=None, noise_model=None, shots=1024, execution_type="fault"):
    """Run one injection according to the specified input parameters."""
    if noise_model is not None:
        noise_model_simulator = noise_model
    elif device_backend is not None:
        noise_model_simulator = NoiseModel.from_backend(device_backend)
    else:
        noise_model_simulator = NoiseModel()
        
    if noise_model.is_ideal():
        noise_model_simulator.add_basis_gates(list(NoiseModel._1qubit_instructions))

    noise_model_simulator.basis_gates.append("id")

    # LEGACY reference for gate based injection point selection:
    # Prepend to the circuit an Identity gate labeled "inj_gate" to be used as target by the NoiseModel
    # Use deepcopy() to preserve the QuantumRegister structure of the original circuit and allow composition
    # inj_circuit = deepcopy(circuit)
    # inj_circuit.clear()
    # for i in range(len(circuit.qubits)):
    #     inj_circuit.append(get_inj_gate(), [i])
    # inj_circuit.compose(circuit, inplace=True)
    inj_circuit = circuit

    if execution_type == "golden":
        sim = AerSimulator(noise_model=deepcopy(noise_model_simulator), basis_gates=noise_model_simulator.basis_gates)
    else:
        # List of paths from injection point to all other nodes such that there are no duplicates in the last two nodes of the path
        if isinstance(injection_point, list):
            spread_map = None
            for q in injection_point:
                if spread_map == None:
                    spread_map = error_spread_map(device_backend.configuration().coupling_map, q, inf)
                else:
                    q_map = error_spread_map(device_backend.configuration().coupling_map, q, spread_depth)
                    spread_map = merge_injection_paths(spread_map, q_map)
            spread_map = [path for path in spread_map if (len(path) <= spread_depth+1 or (path[-1] in injection_point and path[-2] in injection_point))]
        else:
            spread_map = error_spread_map(device_backend.configuration().coupling_map, injection_point, inf)

        noise_model = deepcopy(noise_model_simulator)
        noise_model.add_basis_gates(["inj_gate"])
        noise_model = spread_transient_error(noise_model, spread_map, root_inj_probability, spread_depth, transient_error_function, damping_function)
        sim = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
    try:
        sim.set_options(device='GPU')
        counts = sim.run(inj_circuit, shots=shots).result().get_counts()
    except (RuntimeError, QiskitError) as e:
        sim.set_options(device='CPU')
        counts = sim.run(inj_circuit, shots=shots).result().get_counts()
    
    return counts

def injection_campaign(circuits, device_backends=None, noise_models=None, injection_points=0, transient_error_functions = reset_to_zero, spread_depths = 0, damping_functions = square_damping, transient_error_duration_ns = 25000000, n_quantised_steps = 4, processes=1):
    """Run a set of injections as the product of all interables in the arguments with a custom parallel subprocess pool.
    Every argument is supplied as either a single object or as a list."""
    
    if not (len(listify(circuits)) == len(listify(device_backends)) == len(listify(noise_models))):
        raise Exception("circuits, device_backends and noise_models must have the same length.") 
    
    subprocess_args = {}
    index_subprocesses = 0
    for circuit, device_backend, noise_model in zip(listify(circuits), listify(device_backends), listify(noise_models)):
        shots_per_time_batch, probability_per_batch = get_shots_and_probability_per_batch(circuit, n_quantised_steps, transient_error_duration_ns)
        # Add the golden executions (ingnore args relative to injection)
        index_subprocesses += 1
        subprocess_args[index_subprocesses] = {"circuit":circuit, "injection_point":None, "transient_error_function":None, "root_inj_probability":0.0, 
                                                "time_step":0, "spread_depth":0, "damping_function":None, "device_backend":device_backend, 
                                                "noise_model":noise_model, "shots":int(shots_per_time_batch*n_quantised_steps),"execution_type":"golden"}
        for injection_point in listify(injection_points):
            for quantised_step_index, root_inj_probability in enumerate(probability_per_batch):
                for transient_error_function in listify(transient_error_functions):
                    for spread_depth in listify(spread_depths):
                        for damping_function in listify(damping_functions):
                            index_subprocesses += 1
                            subprocess_args[index_subprocesses] = {"circuit":circuit, "injection_point":injection_point, "transient_error_function":transient_error_function, "root_inj_probability":root_inj_probability, 
                                                                    "time_step":int(shots_per_time_batch)*quantised_step_index, "spread_depth":spread_depth, "damping_function":damping_function, 
                                                                    "device_backend":device_backend, "noise_model":noise_model, "shots":int(shots_per_time_batch), "execution_type":"injection"}

    if processes == 1:
        for p_index, args in deepcopy(subprocess_args.items()):
            subprocess_args[p_index]["counts"] = run_injection(**args)
    else:
        command = ['python3', 'subprocess_run_injection.py']
        process_output_list = subprocess_pool(command=command, args_dict=subprocess_args, max_processes=processes)
        for p_index, counts in process_output_list:
            subprocess_args[p_index]["counts"] = counts

    results_list = []
    for args in subprocess_args.values():
        results_list.append({"circuit_name":args["circuit"].name, 
                             "p2v_map":{k:v._register.name+str(v._index) for k, v in args["circuit"]._layout.initial_layout._p2v.items()} if args["circuit"]._layout is not None else None,
                             "injection_point":tuple(listify(args["injection_point"])), 
                             "transient_error_function":args["transient_error_function"].__name__ if args["transient_error_function"] is not None else None, 
                             "root_inj_probability":args["root_inj_probability"], 
                             "time_step":args["time_step"],
                             "spread_depth":args["spread_depth"], 
                             "damping_function":args["damping_function"].__name__ if args["damping_function"] is not None else None, 
                             "device_backend_name":args["device_backend"].name(),
                             "coupling_map":args["device_backend"].configuration().coupling_map,
                             "noise_model":args["noise_model"].__name__,
                             "shots":args["shots"],
                             "execution_type":args["execution_type"],
                             "counts":args["counts"]
                             })
    result_df = pd.DataFrame(flatten(results_list))

    return result_df

def percentage_collapses_error(golden_counts, inj_counts):
    """Google's test: ratio between the zeroes measured and the total number of qubits measured. Returns a percentage."""
    zeros_measured = 0
    total_measurements = 0
    for bitstring, count in inj_counts.items():
        zeros_in_bitstring = bitstring.count("0")
        zeros_measured += zeros_in_bitstring*count
        total_measurements += len(bitstring)*count
    return (zeros_measured / total_measurements)

def count_collapses_error(golden_counts, inj_counts):
    """Google's test: ratio between the zeroes measured and the total number of qubits measured. Returns an absolute value, averaged across shots."""
    zeros_measured = 0
    total_measurements = 0
    for bitstring, count in inj_counts.items():
        zeros_in_bitstring = bitstring.count("0")
        zeros_measured += zeros_in_bitstring*count
        total_measurements += len(bitstring)*count
    return (zeros_measured / total_measurements) * len(bitstring)

# TODO: untested!
def plot_transient(result_df, compare_function):
    """Plot the results of an injection campaign according to the supplied compare_function(golden_counts, inj_counts) over time. The results are saved under the ./plots directory."""
    
    golden_counts = result_df.loc[result_df['execution_type'] == "golden"]
    result_df["logical_error"] = result_df["counts"].apply(lambda row: compare_function(golden_counts, row))
    result_df.sort_values(by=['time_step'])
    df_golden = result_df.loc[result_df['execution_type'] == "golden"]
    df_injection = result_df.loc[result_df['execution_type'] == "injection"]

    sns.set_theme(style="whitegrid", palette="deep")    
    plt.figure(figsize=(20,10))

    # Golden
    golden_counts = df_golden["counts"].iloc[0]
    golden_bitstring = max(golden_counts, key=golden_counts.get)
    golden_y = []
    time_steps = df_injection["time_step"].drop_duplicates()
    golden_logical_error = compare_function(golden_counts, golden_counts)
    for time_step in time_steps:
        golden_y.append(golden_logical_error)
    plt.plot(time_steps, golden_y, label=f"Golden", color="gold")

    df_injection_point = [x for _, x in df_injection.groupby(["injection_point"])]
    for df in df_injection_point:
        injected_qubit = df["injection_point"].iloc[0][0]
        plt.plot(df["time_step"], df["logical_error"], label=f"Transient on qubit {injected_qubit}") 
    plt.xlabel(f'discretised time (shots)')
    plt.ylabel(f'{compare_function.__name__}')
    plt.title(f'Transient logical error evolution over time')
    plt.legend()
    
    circuit_name = re.sub("[^a-zA-Z]", "", df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/transient_error_over_time_{compare_function.__name__}_{circuit_name} on {df["device_backend_name"].iloc[0]}'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename)
    plt.close()

def plot_injection_logical_error(result_df, compare_function_generator, log=True):
    """Plot the results of an injection campaign according to the supplied compare_function(golden_counts, inj_counts) generator over the injection probability of each point. The results are saved under the ./plots directory."""
    plot_df_list = []
    df_circuit_name = [x for _, x in result_df.groupby(["circuit_name"])]
    for df in df_circuit_name:
        golden_counts = df.loc[df['execution_type'] == "golden"]
        d = make_tuple(re.sub("[^0-9(),]", "", golden_counts["circuit_name"].values[0]))
        compare_error_function = compare_function_generator(d=d)
        df["logical_error"] = df["counts"].apply(lambda row: compare_error_function(golden_counts, row))
        df.sort_values(by=['time_step'])
        inj_logical_error = df.loc[df['execution_type'] == "injection"]
        plot_df_list.append(inj_logical_error[["circuit_name", "device_backend_name", "root_inj_probability", "logical_error"]])
        
    linestyle = {"ideal":"--", "noisy":"-"}
    colours = mcp.gen_color(cmap="cool", n=len(plot_df_list))
    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(20,10))

    for index, df in enumerate(plot_df_list):
        plt.plot(df["root_inj_probability"], df["logical_error"], label=df["circuit_name"].iloc[0], 
                    marker="s", color=colours[index], linestyle="-")
        
    plt.plot(df["root_inj_probability"], df["root_inj_probability"], label=f"Breakeven performance", 
                marker="d", color="red", linestyle="--")
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.xlabel(f'injection probability')
    plt.ylabel(f'{compare_function_generator().__name__}')
    plt.title(f'{df["circuit_name"].iloc[0]} on {df["device_backend_name"].iloc[0]}')
    plt.legend()

    circuit_name = re.sub("[^a-zA-Z]", "", df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/logical_physical_error_{compare_function_generator().__name__}_{circuit_name} on {df["device_backend_name"].iloc[0]}'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename)
    plt.close()

def plot_histogram_error(result_df, compare_function):
    """Plot the results of a <circuit_name>_histogram_affected_qubits results file. The results are saved under the ./plots directory."""
    
    def get_label(row):
        return f"{row['device_backend_name']} {'damped' if row['spread_depth'] != 0 else 'constant'}"

    def get_len(row):
        return len(row["injection_point"])

    # code distance analysis
    df_device_backend_name = [x for _, x in result_df.groupby(["device_backend_name"])]
    merged_df = pd.DataFrame()
    for df_backend in df_device_backend_name:
        golden_counts = df_backend.loc[df_backend['execution_type'] == "golden"].head(1)
        df_backend["logical_error"] = df_backend["counts"].apply(lambda row: compare_function(golden_counts, row))
        df_backend.sort_values(by=['time_step'])
        golden_counts = df_backend.loc[df_backend['execution_type'] == "golden"].head(1) #["logical_error"]
        df_backend = df_backend.loc[df_backend['execution_type'] == "injection"] #["logical_error"]
        df_backend["label"] = df_backend.apply(get_label, axis=1)
        df_backend["total_injected_qubits"] = df_backend.apply(get_len, axis=1)
        merged_df = pd.concat([merged_df, df_backend], ignore_index=True)

    df = pd.pivot(merged_df, index="label", columns="total_injected_qubits", values="logical_error")
    df = df.loc[:, ~(df.eq(1) | df.isna()).all()]
    n_unique_injection_points = len(df.columns)
    df = df.unstack().reset_index(name="logical_error")
    sns.set_theme(style="whitegrid", palette="deep")
    g = sns.catplot(x="total_injected_qubits", y="logical_error", hue="label", data=df, kind="bar", legend_out=False, height=10, aspect=n_unique_injection_points/10)

    circuit_name = re.sub("[^a-zA-Z]", "", merged_df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/histogram_spread_depth_error_{compare_function.__name__}_{circuit_name} on {merged_df["device_backend_name"].iloc[0]}'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename)
    plt.close()

def plot_3d_surface(result_df, compare_function, ip=1):
    """Plot the results of a <circuit_name>_surfaceplot results file. The results are saved under the ./plots directory."""

    df_data_list = []
    df_noise_model = [x for _, x in result_df.groupby(["noise_model"])]
    for df in df_noise_model:
        golden_counts = df.loc[df['execution_type'] == "golden"]
        df["logical_error"] = df["counts"].apply(lambda row: compare_function(golden_counts, row))
        df.sort_values(by=['time_step'])
        inj_logical_error = df.loc[df['execution_type'] == "injection"][["noise_model", "root_inj_probability", "logical_error"]]
        inj_logical_error["physical_error"] = inj_logical_error.apply(lambda row: float(re.findall("-?[\d.]+(?:e-?\d+)?", row["noise_model"])[0]), axis=1)
        df_data_list.append(inj_logical_error)

    df = pd.concat(df_data_list, ignore_index=True)
    sns.set_style("whitegrid", {'axes.grid' : False})
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.invert_xaxis()

    # ax.yaxis.set_major_formatter(mticker.FuncFormatter(percentage_tick_formatter_no_decimal))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(percentage_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    x1 = 10**(np.linspace(np.log10(df['physical_error'].min()), np.log10(df['physical_error'].max()), ip*len(df['physical_error'].drop_duplicates()) ))
    # y1 = np.linspace(df['root_inj_probability'].min(), df['root_inj_probability'].max(), ip*len(df['root_inj_probability'].drop_duplicates()))
    y1 = 10**(np.linspace(np.log10(df['root_inj_probability'].min()), np.log10(df['root_inj_probability'].max()), ip*len(df['root_inj_probability'].drop_duplicates()) ))
    X, Y = np.meshgrid(x1, y1)
    Z = griddata(((df['physical_error']), df['root_inj_probability']), df['logical_error'], (X, Y), method="linear")
    g = ax.plot_surface(np.log10(X), np.log10(Y), Z, rstride=1, cstride=1, cmap="Spectral_r", linewidth=0.0, antialiased=True)

    # g = ax.plot_trisurf(np.log10(df.physical_error), df.root_inj_probability, df.logical_error, cmap="Spectral_r", linewidth=0.2, antialiased=True)
    ax.set_xlabel('\nPhysical error rate')
    ax.set_ylabel('\nInjection probability')
    ax.set_zlabel('\nLogical error', labelpad=12)

    circuit_name = re.sub("[^a-zA-Z]", "", result_df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/3d_surfaceplot_{compare_function.__name__}_{circuit_name} on {result_df["device_backend_name"].iloc[0]}'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename)
    plt.close()

def plot_topology_injection_point_error(result_df, compare_function):
    """Plot the results of a <circuit_name>_topologies_analysis results file. The results are saved under the ./plots directory."""
    vmax = 0.0
    graphs_with_data = []
    df_device_backend_name = [x for _, x in result_df.groupby(["device_backend_name"])]
    for df_all_nodes in df_device_backend_name:
        golden_counts = df_all_nodes.loc[df_all_nodes['execution_type'] == "golden"]
        df_all_nodes["logical_error"] = df_all_nodes["counts"].apply(lambda row: compare_function(golden_counts, row))
        df_all_nodes.sort_values(by=['time_step'])
        df_inj_point = df_all_nodes.loc[df_all_nodes['execution_type'] == "injection"][["injection_point", "logical_error"]].groupby(["injection_point"], as_index=False).median()
        max_average_node_logical_error = df_inj_point["logical_error"].max()
        if max_average_node_logical_error > vmax:
            vmax = max_average_node_logical_error
        coupling_map = df_all_nodes["coupling_map"].iloc[0]
        G = nx.Graph(coupling_map)
        for node_index in list(G):
            G.nodes[node_index]["logical_error"] = 0
        for index, row in df_inj_point.iterrows():
            node_logical_error = row["logical_error"] if not row.empty else 0
            node_index = row["injection_point"][0]
            G.nodes[node_index]["logical_error"] = node_logical_error
        p2v_map = golden_counts["p2v_map"].iloc[0]
        graphs_with_data.append((df_all_nodes["device_backend_name"].iloc[0], G, p2v_map))

    nc = int(np.ceil(np.sqrt(len(graphs_with_data))))
    nr = int(np.ceil(len(graphs_with_data)/nc))
    fig = plt.figure(figsize=(12*nc, 10*nr))
    plt.clf()
    plt.axis('off')
    fig, ax = plt.subplots(nr, nc, num=1)

    # drawing nodes and edges separately so we can capture collection for colobar
    for i, (name, G, p2v_map) in enumerate(graphs_with_data):
        ix = np.unravel_index(i, ax.shape)
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato", args="-Gepsilon=.00000000001 -Glen=200 -Gmaxiter=100 -Goverlap_scaling=20 -Goverlap='false' -Gsep=+50") if name!="linear" else nx.spiral_layout(G, equidistant=True)
        ec = nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax[ix])
        colors = nx.get_node_attributes(G, "logical_error")
        
        # nc = nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=250, cmap=plt.cm.Spectral_r, ax=ax[ix], vmin=0.0, vmax=vmax)
        layout = {k:v.replace('t_', '').replace('tq_', '').replace('ancilla', 'a').replace('data', 'd') for k,v in p2v_map.items()}
        ancilla_nodelist = [k for k,v in layout.items() if "a" in v]
        measure_nodelist = [k for k,v in layout.items() if "m" in v]
        data_nodelist = [k for k,v in layout.items() if "d" in v]
        nc = nx.draw_networkx_nodes(G, pos, node_color=[colors[n] for n in ancilla_nodelist], node_size=1000, cmap=plt.cm.Spectral_r, nodelist=ancilla_nodelist, node_shape='p', edgecolors="black", ax=ax[ix], vmin=0.0, vmax=vmax)
        nc = nx.draw_networkx_nodes(G, pos, node_color=[colors[n] for n in measure_nodelist], node_size=1000, cmap=plt.cm.Spectral_r, nodelist=measure_nodelist, node_shape='s', edgecolors="black", ax=ax[ix], vmin=0.0, vmax=vmax)
        nc = nx.draw_networkx_nodes(G, pos, node_color=[colors[n] for n in data_nodelist], node_size=1000, cmap=plt.cm.Spectral_r, nodelist=data_nodelist, node_shape='o', edgecolors="black", ax=ax[ix], vmin=0.0, vmax=vmax)

        nodes = list(G)
        light_text = [n for n, name in layout.items() if (G.nodes[n]['logical_error'] < 0.25*vmax or G.nodes[n]['logical_error'] > 0.75*vmax)]
        nx.draw_networkx_labels(G, pos, labels={n:name for n, name in layout.items() if n not in light_text}, font_color="black", ax=ax[ix])
        nx.draw_networkx_labels(G, pos, labels={n:name for n, name in layout.items() if n in light_text}, font_color="white", ax=ax[ix])
        ax[ix].set_title(name, fontsize=25)

    for i in range(i+1, len(ax.flatten())):
        ix = np.unravel_index(i, ax.shape)
        ax[ix].set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r, norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist())
    cbar.ax.tick_params(labelsize=25) 
    
    circuit_name = re.sub("[^a-zA-Z]", "", result_df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/topology_injection_point_{compare_function.__name__}_{circuit_name} on {result_df["device_backend_name"].iloc[0]}'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename)
    plt.close()

def plot_minimum_inj_qubits(result_df, compare_function_generator, threshold_min=0.0, threshold_catasptrophic=0.5):
    """Plot the results of a <circuit_name> minimum_inj_qubits results file. The results are saved under the ./plots directory."""

    def get_len(row):
        return len(row["injection_point"])

    # code distance analysis
    df_device_circuit_name = [x for _, x in result_df.groupby(["circuit_name"])]
    merged_df_list = []
    for df_circuit_name in df_device_circuit_name:
        golden_counts = df_circuit_name.loc[df_circuit_name['execution_type'] == "golden"].head(1)
        d = make_tuple(re.sub("[^0-9(),]", "", golden_counts["circuit_name"].values[0]))
        compare_function = compare_function_generator(d=d)
        # Add compare fn
        df_circuit_name["logical_error"] = df_circuit_name["counts"].apply(lambda row: compare_function(golden_counts, row))
        golden_counts = df_circuit_name.loc[df_circuit_name['execution_type'] == "golden"].head(1)
        df_circuit_name = df_circuit_name.loc[df_circuit_name['execution_type'] == "injection"]
        df_circuit_name["total_injected_qubits"] = df_circuit_name.apply(get_len, axis=1)
        df_circuit_name.reset_index(drop=True, inplace=True)
        df_circuit_name = df_circuit_name[df_circuit_name.columns.intersection(["circuit_name", "device_backend_name", "total_injected_qubits", "logical_error"])]
        df_circuit_name = df_circuit_name.groupby(["circuit_name", "device_backend_name", "total_injected_qubits"], as_index=False).agg({'logical_error': ['mean', 'median', 'std']}).reset_index()
        df_circuit_name.reset_index(drop=True, inplace=True)
        df_circuit_name.columns = ["_".join(col_name).rstrip('_') for col_name in df_circuit_name.columns]
        
        df_circuit_name.sort_values(by=['total_injected_qubits'])
        df_circuit_name = df_circuit_name.loc[df_circuit_name["logical_error_median"] > threshold_min]
        df_circuit_name["d"] = d[0]
        df_circuit_name["label"] = "Minimum observable error"
        merged_df_list.append(df_circuit_name.head(1))
        df_circuit_name = df_circuit_name.loc[df_circuit_name["logical_error_median"] > threshold_catasptrophic]
        df_circuit_name["label"] = "Unrecoverable error"
        merged_df_list.append(df_circuit_name.head(1))

    merged_df = pd.concat(merged_df_list, ignore_index=True)
    merged_df.sort_values(by=["d"], inplace=True)
    sns.set_theme(style="whitegrid", palette="deep", rc={'figure.figsize':(20,10)})
    sns.barplot(merged_df,  y="circuit_name", x="total_injected_qubits", hue="label", orient="h")

    circuit_name = re.sub("[^a-zA-Z]", "", merged_df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/min_injected_qubits_{compare_function.__name__}_{circuit_name} on {merged_df["device_backend_name"].iloc[0]}'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename)
    plt.close()

def log_tick_formatter(val, pos=None):
    return f'$10^{{{val:.0f}}}$'

def percentage_tick_formatter_no_decimal(val, pos=None):
    return f'${{{val*100:.0f}}} \%$'

def percentage_tick_formatter(val, pos=None):
    return f'     ${{{val*100:.1f}}} \%$'

# %%
