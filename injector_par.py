#%%
from copy import deepcopy
from inspect import getsource
from functools import partial
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent import futures
from itertools import repeat
from more_itertools import chunked
from time import time, sleep
from datetime import datetime
from objprint import op
from math import exp, pi
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import isdir, dirname
from os import mkdir, scandir, system
from subprocess import Popen
import dill, gzip 
import tqdm
from mycolorpy import colorlist as mcp
from qiskit import transpile, QuantumCircuit
from qiskit import Aer
from qiskit.providers.fake_provider import FakeJakarta, FakeMumbai
from qiskit_aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit_aer.noise.device.parameters import thermal_relaxation_values, readout_error_values
from qiskit.providers.fake_provider import ConfigurableFakeBackend
from qiskit.providers.aer.noise import reset_error, mixed_unitary_error, pauli_error
from qiskit.circuit.library import RYGate, IGate
from qiskit.quantum_info.analysis import hellinger_distance, hellinger_fidelity

file_logging = False
logging_filename = "./asuqa.log"
console_logging = True

def FakeSycamore25():
    n_qubits = 25
    cmap_sycamore_25 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 9], [9, 14], [14, 19], [19, 24], [24, 23], [23, 18], [18, 13], [13, 8], [8, 7], [7, 6], [6, 5], [5, 10], [10, 15], [15, 20], [20, 21], [21, 16], [11, 16], [0, 5], [1, 6], [6, 11], [11, 12], [12, 7], [7, 2], [3, 8], [8, 9], [14, 13], [13, 12], [12, 17], [18, 17], [16, 17], [17, 22], [23, 22], [22, 21], [10, 11], [16, 15], [18, 19]]
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

def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def log(content):
    """Logging wrapper, can redirect both to stdout and a file"""
    if file_logging:
        fp = open(logging_filename, "a")
        fp.write(content+'\n')
        fp.flush()
        fp.close()
    if console_logging:
        print(content)

def f_wrapped(arg):
    return run_injection_campaing(*arg)  # Unpacks args

def save_results(results, filename="./results.p.gz"):
    """Save a single/double circuits results object"""
    # Temporary fix for pickle.dump
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    dill.dump(results, gzip.open(filename, 'w'))
    log(f"Files saved to {filename}")

def read_file(filename):
    """Read a partial result file"""
    with open(filename, 'rb') as pickle_file:
        data = dill.load(pickle_file)
        return data

def read_results_directory(dir):
    """Process double fault injection results directory and return all data"""
    filenames = []
    for filename in scandir(dir):
        if filename.is_file():
            filenames.append(filename.path)

    filenames.sort()

    pool = Pool(cpu_count())
    injection_results = pool.map(read_file, filenames)
    pool.close()
    pool.join()
    
    return injection_results

def plot_coupling_map(device_backend):
    G = nx.Graph(device_backend.configuration().coupling_map)
    nx.draw(G, with_labels=True)
    plt.show()

# Define a custom transient error, parametrised by the probability p of being applied.
# Errors can be composed or tensored together to achieve more complex behaviours:
# - Composition: E(ρ)=E2(E1(ρ)) -> error = error1.compose(error2)
# - Tensor: E(ρ)=(E1⊗E2)(ρ) -> error error1.tensor(error2)
def reset_to_zero(p):
    return reset_error(p, 0.0)

def reset_to_one(p):
    return reset_error(0.0, p)

def reset_to_zero_and_ry(p):
    ry_error_matrix = RYGate(theta=pi/2).to_matrix()
    identity_matrix = IGate().to_matrix()
    return reset_error(p, 0.0).compose(mixed_unitary_error([(ry_error_matrix, p), (identity_matrix, 1 - p)]))

# Define a custom damping function parametrised by the distance from the root impact qubit
# Multiplicative damping function
def geometric_damping(depth=0):
    return 0.9**depth

def square_damping(depth=0):
    n = 1
    return n**2/((depth+n)**2)

def exponential_damping(depth=0):
    return exp(-(1/10)*depth**2)

def error_probability_decay(input, transient_duration_ns):
    # factor = 1/(4*1.0e6*25.0e6)
    factor = 4.5
    x = input/transient_duration_ns
    return exp(-(factor)*x**2)

def valid_inj_qubits(sssp_dict_of_qubits):
    inj_points = []
    for paths in sssp_dict_of_qubits.values():
        for endpoint in paths:
            inj_points.append(endpoint)
        
    return list(set(inj_points))

# Define a dummy inj_gate to be used as an anchor for the NoiseModel
def get_inj_gate(inj_duration=0.0):
    # gate_qc = QuantumCircuit(1, name='id')
    # gate_qc.id(0)
    # inj_gate = gate_qc.to_gate()
    inj_gate = IGate()
    inj_gate.label = "inj_gate"
    inj_gate.duration = inj_duration

    return inj_gate

def spread_transient_error(noise_model, sssp_dict_of_qubit, root_inj_probability, spread_depth, transient_error_function, damping_function):
    if spread_depth > 0:
        noise_model = spread_transient_error(noise_model, sssp_dict_of_qubit, root_inj_probability, spread_depth-1, transient_error_function, damping_function)
    
    for path in sssp_dict_of_qubit.values():
        if len(path)-1 == spread_depth:
            transient_error = transient_error_function(root_inj_probability*damping_function(depth=spread_depth))
            # Add error to specific gate on specific qubit
            # noise_model.add_quantum_error(transient_error, instructions=["inj_gate"], qubits=[path[-1]])

            # Add error to all gates on specific qubit
            target_instructions = list(noise_model._1qubit_instructions.intersection(noise_model.basis_gates))
            target_instructions.append("inj_gate")
            noise_model.add_quantum_error(transient_error, instructions=target_instructions, qubits=[path[-1]], warnings=False)

            if len(path) >= 2:
                target_instructions_2q = list(noise_model._2qubit_instructions.intersection(noise_model.basis_gates))
                two_qubit_targets = (path[-2], path[-1])
                noise_model.add_quantum_error(transient_error.tensor(transient_error), instructions=target_instructions_2q, qubits=two_qubit_targets, warnings=False)

    return noise_model

def run_injection_campaing(circuit, injection_points=[0], transient_error_function=reset_to_zero, root_inj_probability=1.0, time_step=0, spread_depth=0, damping_function=None, device_backend=None, noise_model=None, shots=1024, execution_type="fault"):
    results = []

    if noise_model is not None:
        noise_model_simulator = noise_model
    elif device_backend is not None:
        noise_model_simulator = NoiseModel.from_backend(device_backend)
    else:
        noise_model_simulator = NoiseModel()
        
    if noise_model.is_ideal():
        noise_model_simulator.add_basis_gates(list(NoiseModel._1qubit_instructions))
    
    # basis_gates = noise_model_simulator.basis_gates
    # basis_gates.append("id")
    noise_model_simulator.basis_gates.append("id")

    # log(f"Injection will be performed on the following circuit:\n{t_circ.draw()}")

    # The noisy AerSimulator doesn't recognise the inj_gate.
    # Possible solution: map the error to an Identity Gate, wihtout using a custom gate
    # Create a column of id gates as injection points, then append the rest of the transpiled circuit afterwards
    # Use deepcopy() to preserve the QuantumRegister structure of the original circuit and allow composition
    inj_circuit = deepcopy(circuit)
    inj_circuit.clear()
    for i in range(len(circuit.qubits)):
        # inj_circuit.id([i])
        inj_circuit.append(get_inj_gate(), [i])
    inj_circuit.compose(circuit, inplace=True)

    # Dictionary of sssp indexed by starting point
    sssp_dict = {}
    G = nx.Graph(device_backend.configuration().coupling_map)
    for inj_index in range(len(circuit.qubits)):
        # Populate the dictionary of single source shortest paths among nodes in the device's topology up to spread_depth distance
        sssp_dict[inj_index] = nx.single_source_shortest_path(G, inj_index, cutoff=spread_depth)

    # Inject in each physical qubit that could propagate its errors to one of the qubits actually used in the circuit, according to spread_depth
    # Add the sssp of the neighbouring unused qubits that may corrupt the output qubits
    inj_qubits = valid_inj_qubits(sssp_dict)
    keys = sssp_dict.keys()
    for endpoint in inj_qubits:
        if endpoint not in keys:
            sssp_dict[endpoint] = nx.single_source_shortest_path(G, endpoint, cutoff=spread_depth)

    if execution_type == "golden":
        sim = AerSimulator(noise_model=deepcopy(noise_model_simulator), basis_gates=noise_model_simulator.basis_gates)
        try:
            sim.set_options(device='GPU')
            counts = sim.run(inj_circuit, shots=shots).result().get_counts()
        except RuntimeError:
            sim.set_options(device='CPU')
            counts = sim.run(inj_circuit, shots=shots).result().get_counts()
        results.append( {"root_injection_point":None, "shots":shots, "transient_fault_prob":root_inj_probability, "counts":counts} )
    else:
        for inj_index in injection_points:
            noise_model = deepcopy(noise_model_simulator)
            noise_model.add_basis_gates(["inj_gate"])
            noise_model = spread_transient_error(noise_model, sssp_dict[inj_index], root_inj_probability, spread_depth, transient_error_function, damping_function)

            sim = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
            try:
                sim.set_options(device='GPU')
                counts = sim.run(inj_circuit, shots=shots).result().get_counts()
            except RuntimeError:
                sim.set_options(device='CPU')
                counts = sim.run(inj_circuit, shots=shots).result().get_counts()
            results.append( {"root_injection_point":inj_index, "shots":shots, "transient_fault_prob":root_inj_probability, "counts":counts} )

    return results

def get_shot_execution_time_ns(circuit):
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

def run_transient_injection(circuit, device_backend=None, noise_model=None, injection_points=[0], transient_error_function = reset_to_zero, spread_depth = 0, damping_function = exponential_damping, transient_error_duration_ns = 25000000, n_quantised_steps = 4, processes=1):
    data_wrapper = {"circuit":circuit,
                    "device_backend":device_backend,
                    "noise_model":noise_model,
                    "transient_error_function":transient_error_function,
                    "transient_error_duration_ns":transient_error_duration_ns,
                    "spatial_damping_function":damping_function,
                    "spread_depth":spread_depth,
                    "compare_function":None,
                    "golden_df":None,
                    "injected_df":None,
                    }
    
    shot_time_per_circuit = get_shot_execution_time_ns(circuit)

    total_shots = transient_error_duration_ns // shot_time_per_circuit
    shots_per_time_batch = total_shots // n_quantised_steps
    
    probability_per_batch = []
    for batch in range(n_quantised_steps+1):
        time_step = batch*shots_per_time_batch*shot_time_per_circuit
        probability = error_probability_decay(time_step, transient_error_duration_ns)
        probability_per_batch.append(probability)
    
    golden_results = run_injection_campaing(circuit,
                                            injection_points,
                                            transient_error_function,
                                            root_inj_probability=0.0,
                                            spread_depth=spread_depth,
                                            damping_function=damping_function, 
                                            device_backend=device_backend, 
                                            noise_model=noise_model,
                                            shots=int(shots_per_time_batch*n_quantised_steps),
                                            execution_type="golden")
    data_wrapper["golden_df"] = pd.DataFrame(golden_results)

    child_process_args = {"circuit":circuit,
                          "injection_points":injection_points,
                          "transient_error_function":transient_error_function,
                          "probability_per_batch":probability_per_batch,
                          "spread_depth":spread_depth,
                          "damping_function":damping_function,
                          "device_backend":device_backend,
                          "noise_model":noise_model,
                          "shots_per_time_batch":shots_per_time_batch,
                          "n_quantised_steps":n_quantised_steps,
                          }
    dill.settings['recurse'] = True

    child_process_args_name = f'tmp/data_{circuit.name}.pkl'
    if not isdir(dirname(child_process_args_name)):
        mkdir(dirname(child_process_args_name))
    with open(child_process_args_name, 'wb') as handle:
        dill.dump(child_process_args, handle, protocol=dill.HIGHEST_PROTOCOL)

    proc_list = []
    probs_circ = enumerate(probability_per_batch)
    queue = [ ['python3', 'wrapper.py', f'{iteration}', f'{child_process_args_name}'] for iteration, p in probs_circ ]
    log(f"Injecting {circuit.name}:")

    with tqdm.tqdm(total=len(probability_per_batch)) as pbar:
        for process in queue:
            p = Popen(process)
            proc_list.append(p)
            if len(proc_list) == processes:
                wait = True
                while wait:
                    done, num = check_for_done(proc_list)
                    if done:
                        proc_list.pop(num)
                        wait = False
                    else:
                        sleep(1)
                pbar.update()
        while len(proc_list) > 0:
            done, num = check_for_done(proc_list)
            if done:
                proc_list.pop(num)
                pbar.update()

    res_dirname = f"results/{circuit.name}_{n_quantised_steps}/"
    data_wrapper["injected_df"] = pd.DataFrame(list(flatten(read_results_directory(res_dirname))))

    system(f"rm -rf '{res_dirname}'")
    data_wrapper_filename = f"results/campaign_{datetime.now()}"
    with open(data_wrapper_filename, 'wb') as handle:
        dill.dump(data_wrapper, handle, protocol=dill.HIGHEST_PROTOCOL)
    system("rm -rf ./tmp")

    return data_wrapper

def relative_error(golden_counts, inj_counts):
    # Basic str compare
    golden_bitstring = max(golden_counts, key=golden_counts.get)
    return 1.0 - (golden_counts[golden_bitstring] / sum(golden_counts.values()))

def percentage_collapses_error(golden_counts, inj_counts):
    # Google's test
    zeros_measured = 0
    total_measurements = 0
    for bitstring, count in inj_counts.items():
        zeros_in_bitstring = bitstring.count("0")
        zeros_measured += zeros_in_bitstring*count
        total_measurements += len(bitstring)*count
    return (zeros_measured / total_measurements)

def count_collapses_error(golden_counts, inj_counts):
    # Google's test
    zeros_measured = 0
    total_measurements = 0
    for bitstring, count in inj_counts.items():
        zeros_in_bitstring = bitstring.count("0")
        zeros_measured += zeros_in_bitstring*count
        total_measurements += len(bitstring)*count
    return (zeros_measured / total_measurements) * len(bitstring)

def filter_entry(circuit_data, qubits):
    shot_dict = {}
    for inj_type, res_tuple_list in circuit_data[1].items():
        shot_dict[inj_type] = []
        for inj_index, res in [item for item in res_tuple_list]:
            if inj_index in qubits:
                shot_dict[inj_type].append((inj_index, res))

    return (circuit_data[0], shot_dict)

def filter_dict(result_dict, qubits):
    items = result_dict["injection_results"].items()
    for circuit_name, circuit_data in items:
        pool = Pool(cpu_count())
        filtered_exp = pool.map(partial(filter_entry, qubits=qubits), circuit_data)
        pool.close()
        pool.join()
        result_dict["injection_results"][circuit_name] = filtered_exp

    return result_dict

def plot_transient(data, compare_function):
    markers = {"ideal":"o", "noisy":"s"}
    linestyle = {"ideal":"--", "noisy":"-"}
    
    sns.set()
    sns.set_palette("deep")

    golden_executions = data['golden_df']
    inj_executions = data['injected_df']
    inj_executions.sort_values('transient_fault_prob')
    time_steps = [index*row["shots"] for index, row in inj_executions.iterrows()]
    plt.figure(figsize=(20,10))

    target = "ideal" if data["noise_model"].is_ideal() else "noisy"

    # Golden
    golden_counts = golden_executions["counts"].iloc[0]
    golden_bitstring = max(golden_counts, key=golden_counts.get)
    golden_y = []
    for time_step in time_steps:
        golden_y.append(compare_function(golden_counts, golden_counts))
    plt.plot(time_steps, golden_y, label=f"golden {target}", color="gold", marker=markers[target], linestyle=linestyle[target]) 

    injected_qubits = [qubit for qubit in inj_executions["root_injection_point"].unique()]
    colours = mcp.gen_color(cmap="cool", n=len(injected_qubits))
    for index, inj_qubit in enumerate(injected_qubits):
        injected_y = []
        for row, p in enumerate(sorted(inj_executions["transient_fault_prob"])):
            inj_counts = inj_executions["counts"].iloc[row]
            if golden_bitstring not in inj_counts.keys():
                inj_counts[golden_bitstring] = 0
            injected_y.append(compare_function(golden_counts, inj_counts))
        plt.plot(time_steps, injected_y, label=f"transient {target} qubit {inj_qubit}", 
                marker=markers[target], color=colours[index], linestyle=linestyle[target]) 
    plt.xlabel(f'discretised time (shots), transient duration: {data["transient_error_duration_ns"]/1e6} ms')
    plt.ylabel(f'{compare_function.__name__}')
    plt.title(f'{data["circuit"].name} on {data["device_backend"].backend_name}, error function: {data["transient_error_function"].__name__}, spread depth: {data["spread_depth"]}, spatial damping function: {data["spatial_damping_function"].__name__}')
    plt.legend()
    # plt.show()

    filename = f'plots/{data["circuit"].name}_res_{len(golden_y)}_errorfn_{compare_function.__name__}_backend_{data["device_backend"].backend_name}_terrorfn_{data["transient_error_function"].__name__}_sd_{data["spread_depth"]}_dampfn_{data["spatial_damping_function"].__name__}'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename)
