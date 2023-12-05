#%%
from copy import deepcopy
from inspect import getsource
from functools import partial
from multiprocessing import Pool, cpu_count
from concurrent import futures
from itertools import repeat
from more_itertools import chunked
from time import time
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
from qiskit.providers.fake_provider import FakeJakarta, FakeMumbai
from qiskit_aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import ConfigurableFakeBackend
from qiskit.providers.aer.noise import reset_error, mixed_unitary_error
from qiskit.circuit.library import RYGate, IGate
from qiskit.quantum_info.analysis import hellinger_distance, hellinger_fidelity

file_logging = False
logging_filename = "./qufi.log"
console_logging = True

def FakeSycamore25():
    n_qubits = 25

    cmap_sycamore_25 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 9], [9, 14], [14, 19], [19, 24], [24, 23], [23, 18], [18, 13], [13, 8], [8, 7], [7, 6], [6, 5], [5, 10], [10, 15], [15, 20], [20, 21], [21, 16], [11, 16], [0, 5], [1, 6], [6, 11], [11, 12], [12, 7], [7, 2], [3, 8], [8, 9], [14, 13], [13, 12], [12, 17], [18, 17], [16, 17], [17, 22], [23, 22], [22, 21], [10, 11], [16, 15], [18, 19]]
    cmap_sycamore_53 = [[0, 5], [5, 1], [1, 6], [6, 2], [2, 7], [7, 14], [14, 8], [8, 3], [3, 9], [9, 4], [4, 10], [10, 16], [16, 9], [9, 15], [15, 8], [5, 11], [11, 17], [17, 23], [23, 29], [29, 35], [35, 41], [41, 47], [41, 48], [48, 42], [42, 49], [49, 43], [43, 50], [50, 44], [44, 51], [51, 45], [45, 52], [52, 46], [46, 40], [40, 45], [45, 39], [39, 44], [44, 38], [38, 43], [43, 37], [37, 42], [42, 36], [36, 41], [36, 29], [29, 24], [24, 17], [17, 12], [12, 5], [12, 6], [6, 13], [13, 7], [12, 18], [18, 13], [13, 19], [19, 14], [14, 20], [20, 15], [15, 21], [21, 16], [16, 22], [22, 28], [28, 21], [21, 27], [27, 20], [20, 26], [26, 19], [19, 25], [25, 18], [18, 24], [24, 30], [30, 25], [25, 31], [31, 26], [26, 32], [32, 27], [27, 33], [33, 28], [28, 34], [34, 40], [40, 33], [33, 39], [39, 32], [32, 38], [38, 31], [31, 37], [37, 30], [30, 36]]

    ibm_device_backend = FakeMumbai()

    qubit_properties_backend = ibm_device_backend.properties()._qubits.items()
    single_qubit_gates = set(ibm_device_backend.configuration().basis_gates).intersection(NoiseModel()._1qubit_instructions)
    single_qubit_gates.add("reset")
    single_qubit_gates.add("measure")

    qubit_t1 = [item[1]["T1"][0] for item in qubit_properties_backend]
    qubit_t2 = [item[1]["T2"][0] for item in qubit_properties_backend]
    qubit_frequency = [item[1]["frequency"][0] for item in qubit_properties_backend]
    qubit_anharmonicity = [item[1]["anharmonicity"][0] for item in qubit_properties_backend]
    qubit_readout_error = [item[1]["readout_error"][0] for item in qubit_properties_backend]
    qubit_readout_length = [item[1]["readout_length"] for item in qubit_properties_backend]
    basis_gates = ibm_device_backend.configuration().basis_gates

    device_backend = ConfigurableFakeBackend(name="FakeSycamore", n_qubits=n_qubits, version=1, 
                                             coupling_map=cmap_sycamore_25, basis_gates=basis_gates, 
                                             qubit_t1=qubit_t1, qubit_t2=qubit_t2, qubit_frequency=qubit_frequency, 
                                             qubit_readout_error=qubit_readout_error, single_qubit_gates=single_qubit_gates, 
                                             dt=None)

    for q, props in device_backend._properties._qubits.items():
        if "readout_length" not in props and q in set(range(n_qubits)):
            props["readout_length"] = qubit_readout_length[q]

    return device_backend

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

# An output bistring is considered correct if it starts with 1
def percentage_correct_outputs(counts):
    corrected = 0
    total = 0
    for basis, count in counts.items():
        if basis.startswith("1"):
            corrected += count
        total += count
    
    return corrected / total

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
    return 1/((depth+1)**2)

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


def print_results_dict(results_dict):
    log(f"\n############ FAULT MODEL USED ############")
    log(f"Transient error function:\n{results_dict['fault_model']['transient_error_function']}")
    log(f"Spread depth: {results_dict['fault_model']['spread_depth']}")
    log(f"Damping function:\n{results_dict['fault_model']['damping_function']}")
    log(f"Backend noise model used: {results_dict['fault_model']['device_backend'].backend_name if results_dict['fault_model']['device_backend'] is not None else 'None'}")
    
    for circuit_name, results in results_dict["jobs"].items():
        res_keys = list(results.keys())
        log(f"\n############ RESULTS {circuit_name} ############")
        if "ideal" in res_keys:
            log(f"Noiseless golden execution of {circuit_name}:\n{results['ideal'].get_counts()}")
        if "ideal_with_transient" in res_keys:
            for inj_qubit, job in results["ideal_with_transient"]:
                log(f"Noiseless fault execution of {circuit_name} with injection root qubit {inj_qubit}:\n{job.get_counts()}")
        log(f"")
        if "noisy" in res_keys:
            log(f"Noisy golden execution of {circuit_name}:\n{results['noisy'].get_counts()}")
        if "noisy_with_transient" in res_keys:
            for inj_qubit, job in results["noisy_with_transient"]:
                log(f"Noisy fault execution of {circuit_name} with injection root qubit {inj_qubit}:\n{job.get_counts()}")
                log(f"Time taken: {job.time_taken}")

# Define a dummy inj_gate to be used as an anchor for the NoiseModel
def get_inj_gate(inj_duration=0.0):
    # gate_qc = QuantumCircuit(1, name='id')
    # gate_qc.id(0)
    # inj_gate = gate_qc.to_gate()
    inj_gate = IGate()
    inj_gate.label = "inj_gate"
    # inj_gate.name = "inj_gate"
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
        
    return noise_model

def run_injection_campaing(circuits, injection_points=[0], transient_error_function=reset_to_zero, root_inj_probability=1.0, time_step=0, spread_depth=0, damping_function=None, device_backend=None, apply_transpiler=True, noiseless=False, shots=1024, execution_type="both"):
    # Select a simulator
    qasm_sim = AerSimulator()
    result = {"fault_model":{}, "jobs":{}}

    # Save injection information
    result["fault_model"]["transient_error_function"] = transient_error_function
    result["fault_model"]["spread_depth"] = spread_depth
    result["fault_model"]["damping_function"] = damping_function
    result["fault_model"]["device_backend"] = device_backend

    if device_backend is not None:
        noise_model_backend = NoiseModel.from_backend(device_backend)
    else:
        noise_model_backend = NoiseModel()
    
    basis_gates = noise_model_backend.basis_gates
    coupling_map = device_backend.configuration().coupling_map

    # Add the dummy inj_gate and the id to the basis gates of the device
    # basis_gates.append("inj_gate")
    basis_gates.append("id")

    for iteration, circuit in enumerate(circuits):
        if circuit.name in result['jobs'].keys():
            circuit.name = circuit.name + str(iteration)
        result['jobs'][circuit.name] = {}

        # Get the transpiled version of the circuit, if needed
        if apply_transpiler:
            t_circ = transpile(
                circuit,
                device_backend,
                scheduling_method='asap',
                initial_layout=list(range(len(circuit.qubits))),
                seed_transpiler=42,
                basis_gates=basis_gates
            )
        else:
            t_circ = circuit

        # log(f"Injection will be performed on the following circuit:\n{t_circ.draw()}")

        # The noisy AerSimulator doesn't recognise the inj_gate.
        # Possible solution: map the error to an Identity Gate, wihtout using a custom gate
        # Create a column of id gates as injection points, then append the rest of the transpiled circuit afterwards
        # Use deepcopy() to preserve the QuantumRegister structure of the original circuit and allow composition
        new_t_circ = deepcopy(t_circ)
        new_t_circ.clear()
        for i in range(len(circuit.qubits)):
            # new_t_circ.id([i])
            new_t_circ.append(get_inj_gate(), [i])
        new_t_circ.compose(t_circ, inplace=True)
        t_circ = new_t_circ

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

        targets = []
        if noiseless:
            ideal_noise_model = NoiseModel()
            ideal_noise_model.add_basis_gates(list(ideal_noise_model._1qubit_instructions))
            targets.append(("ideal", ideal_noise_model))
        if device_backend is not None:
            targets.append(("noisy", NoiseModel.from_backend(device_backend)))

        for target in targets:
            if execution_type == "golden" or execution_type != "injection":
                # probs = execute(t_circ, qasm_sim, noise_model=target[1], coupling_map=coupling_map, shots=shots)
                # result["jobs"][circuit.name][target[0]] = probs.result()

                qasm_sim = AerSimulator(noise_model=target[1], basis_gates=basis_gates)
                try:
                    qasm_sim.set_options(device='GPU')
                except:
                    pass
                probs = qasm_sim.run(t_circ, shots=shots).result()
                result["jobs"][circuit.name][target[0]] = probs

            # Execute the circuit with/without noise (depending on target), but with the transient_fault, with respect to each qubit
            if execution_type == "injection" or execution_type != "golden":
                for inj_index in injection_points:
                    noise_model = deepcopy(target[1])
                    noise_model.add_basis_gates(["inj_gate"])
                    noise_model = spread_transient_error(noise_model, sssp_dict[inj_index], root_inj_probability, spread_depth, transient_error_function, damping_function)

                    qasm_sim = AerSimulator(noise_model=noise_model, basis_gates=noise_model.basis_gates)
                    try:
                        qasm_sim.set_options(device='GPU')
                    except:
                        pass    
                    probs_with_transient = qasm_sim.run(t_circ, shots=shots).result()

                    if target[0] + "_with_transient" not in result["jobs"][circuit.name].keys():
                        result["jobs"][circuit.name][target[0] + "_with_transient"] = []
                    result["jobs"][circuit.name][target[0] + "_with_transient"].append((inj_index, probs_with_transient))

    return (time_step, result)

def get_shot_execution_time_ns(circuits, device_backend=None, apply_transpiler=True):
    shot_time_per_circuit = {}

    if device_backend is not None:
        noise_model_backend = NoiseModel.from_backend(device_backend)
    else:
        noise_model_backend = NoiseModel()
    
    basis_gates = noise_model_backend.basis_gates

    for iteration, circuit in enumerate(circuits):
        if circuit.name in shot_time_per_circuit.keys():
            circuit.name = circuit.name + str(iteration)

        # Get the transpiled version of the circuit, if needed
        if apply_transpiler:
            t_circ = transpile(
                circuit,
                device_backend,
                scheduling_method='asap',
                initial_layout=list(range(len(circuit.qubits))),
                seed_transpiler=42,
                basis_gates=basis_gates
            )
        else:
            t_circ = circuit

        min_time = []
        max_time = []
        for qubit in t_circ.qubits:
            min_time.append(t_circ.qubit_start_time(qubit))
            max_time.append(t_circ.qubit_stop_time(qubit))

        shot_time_per_circuit[circuit.name] = max(max_time) - min(min_time)

    return shot_time_per_circuit

def run_transient_injection(circuits, device_backend=None, injection_points=[0], transient_error_function = reset_to_zero, spread_depth = 0, damping_function = exponential_damping, apply_transpiler = False, noiseless = True, transient_error_duration_ns = 25000000, n_quantised_steps = 4, processes=1):
    data_wrapper = {}
    data_wrapper["injection_results"] = {}
    
    shot_time_per_circuit = get_shot_execution_time_ns(circuits, device_backend=device_backend, apply_transpiler=apply_transpiler)

    shots_per_time_batch_per_circuits = {}
    for circuit_name, execution_time in shot_time_per_circuit.items():
        total_shots = transient_error_duration_ns // execution_time
        time_steps = total_shots // n_quantised_steps
        shots_per_time_batch_per_circuits[circuit_name] = time_steps
    
    probability_per_batch_per_circuits = {}
    for circuit_name, execution_time in shot_time_per_circuit.items():
        probability_per_batch_per_circuits[circuit_name] = []
        for batch in range(n_quantised_steps+1):
            time_step = batch*shots_per_time_batch_per_circuits[circuit_name]*shot_time_per_circuit[circuit_name]
            probability = error_probability_decay(time_step, transient_error_duration_ns)
            probability_per_batch_per_circuits[circuit_name].append(probability)
    
    injection_results = {}
    golden_results = {}
    fault_model = {}
    for circuit in circuits:
        injection_results[circuit.name] = []
        golden_result = run_injection_campaing([circuit],
                                                injection_points,
                                                transient_error_function,
                                                spread_depth=spread_depth,
                                                time_step=0,
                                                damping_function=damping_function, 
                                                device_backend=device_backend, 
                                                apply_transpiler=apply_transpiler,
                                                noiseless=noiseless,
                                                shots=int(shots_per_time_batch_per_circuits[circuit.name]*n_quantised_steps),
                                                execution_type="golden")
        golden_results[circuit.name] = golden_result[1]
        if 'damping_function' not in fault_model.keys():
            fault_model = golden_results[circuit.name]['fault_model']
            fault_model["transient_error_duration_ns"] = transient_error_duration_ns

        data = {}
        data["args"] = {"circuit":circuit, "injection_points":injection_points, "transient_error_function":transient_error_function, "spread_depth":spread_depth, "damping_function":damping_function, "device_backend":device_backend, "apply_transpiler":apply_transpiler, "noiseless":noiseless, "n_quantised_steps":n_quantised_steps}
        data["probability_per_batch_per_circuits"] = probability_per_batch_per_circuits
        data["shots_per_time_batch_per_circuits"] = shots_per_time_batch_per_circuits
        data["circuit_name"] = circuit.name

        dill.settings['recurse'] = True

        data_name = f'tmp/data_{circuit.name}.pkl'
        if not isdir(dirname(data_name)):
            mkdir(dirname(data_name))
        with open(data_name, 'wb') as handle:
            dill.dump(data, handle, protocol=dill.HIGHEST_PROTOCOL)

        proc_list = []
        probs_circ = enumerate(probability_per_batch_per_circuits[circuit.name])
        log(f"Injecting {circuit.name}:")
        with tqdm.tqdm(total=len(probability_per_batch_per_circuits[circuit.name])) as pbar:
            for batch in chunked(probs_circ, processes):
                for iteration, p in batch:
                    # system(f"python3 wrapper.py {iteration} '{data_name}'")
                    proc = Popen(['python3', 'wrapper.py', f'{iteration}', f'{data_name}'])
                    proc_list.append(proc)
                for proc in proc_list:
                    proc.wait()
                    pbar.update()

    for circuit in circuits:
        res_dirname = f"results/{circuit.name}_{n_quantised_steps}/"
        data_wrapper["injection_results"][circuit.name] = read_results_directory(res_dirname)
        system(f"rm -rf '{res_dirname}'")
    data_wrapper['golden_results'] = golden_results
    data_wrapper['fault_model'] = fault_model

    data_wrapper_filename = f"results/campaign_{datetime.now()}"
    with open(data_wrapper_filename, 'wb') as handle:
        dill.dump(data_wrapper, handle, protocol=dill.HIGHEST_PROTOCOL)
    system("rm -rf ./tmp")

    return data_wrapper

def relative_error(golden_counts, inj_counts):
    # Basic str compare
    golden_bitstring = max(golden_counts, key=golden_counts.get)
    return 1.0 - (golden_counts[golden_bitstring] / sum(golden_counts.values()))

def count_collapses_error(golden_counts, inj_counts):
    # Google's test
    zeros_measured = 0
    total_measurements = 0
    for bitstring, count in inj_counts.items():
        zeros_in_bitstring = bitstring.count("0")
        zeros_measured += zeros_in_bitstring*count
        total_measurements += len(bitstring)*count
    return (zeros_measured / total_measurements)

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

def plot_data(data, compare_function):
    markers = {"ideal":"o", "noisy":"s"}
    linestyle = {"ideal":"--", "noisy":"-"}

    sns.set()
    sns.set_palette("deep")

    for circuit_name in data['golden_results'].keys():
        golden_executions = data['golden_results'][circuit_name]['jobs'][circuit_name]
        inj_executions = data['injection_results'][circuit_name]
        time_steps = [time_step for time_step, dict in data['injection_results'][circuit_name]]
        plt.figure(figsize=(20,10))
        for target in golden_executions.keys():
            # Golden
            golden_counts = golden_executions[target].get_counts()
            golden_bitstring = max(golden_counts, key=golden_counts.get)
            golden_y = []
            for time_step in time_steps:
                golden_y.append(compare_function(golden_counts, golden_counts))
            plt.plot(time_steps, golden_y, label=f"golden {target}", color="gold", marker=markers[target], linestyle=linestyle[target]) 

            injected_qubits = [qubit for qubit, result in inj_executions[0][1][target+"_with_transient"]]
            colours = mcp.gen_color(cmap="cool", n=len(injected_qubits))
            for index, inj_qubit in enumerate(injected_qubits):
                injected_y = []
                for p in inj_executions:
                    inj_counts = [item[1].get_counts() for item in p[1][target+"_with_transient"] if item[0] == inj_qubit]
                    inj_counts = inj_counts[0]
                    if golden_bitstring not in inj_counts.keys():
                        inj_counts[golden_bitstring] = 0
                    injected_y.append(compare_function(golden_counts, inj_counts))
                plt.plot(time_steps, injected_y, label=f"transient {target} qubit {inj_qubit}", 
                        marker=markers[target], color=colours[index], linestyle=linestyle[target]) 
        plt.xlabel(f'discretised time (shots), transient duration: {data["fault_model"]["transient_error_duration_ns"]/1e6} ms')
        plt.ylabel(f'{compare_function.__name__}')
        plt.title(f'{circuit_name} on {data["fault_model"]["device_backend"].backend_name}, error function: {data["fault_model"]["transient_error_function"].__name__}, spread depth: {data["fault_model"]["spread_depth"]}, spatial damping function: {data["fault_model"]["damping_function"].__name__}')
        plt.legend()
        # plt.show()

        filename = f'plots/{circuit_name}_res_{len(golden_y)}_errorfn_{compare_function.__name__}_backend_{data["fault_model"]["device_backend"].backend_name}_terrorfn_{data["fault_model"]["transient_error_function"].__name__}_sd_{data["fault_model"]["spread_depth"]}_dampfn_{data["fault_model"]["damping_function"].__name__}'
        if not isdir(dirname(filename)):
            mkdir(dirname(filename))
        plt.savefig(filename)
