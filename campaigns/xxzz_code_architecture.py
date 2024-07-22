#%%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asuqa import *
from utils import *
from visualisation import *

def main():
    ts = time()
    log(f"Job started at {datetime.fromtimestamp(ts)}.")
    log(f"Architecture campaign on XXZZ Qubit")
    read_from_file = True

    ##################################################################### Transient error controls #####################################################################
    physical_error = 0.01
    transient_error_function = reset_to_zero
    damping_function = square_damping
    transient_error_duration_ns = 100_000_000
    spread_depth = 10
    n_quantised_steps = 10

    ################################################################# Simulation performance controls ##################################################################
    n_nodes = 1
    max_qubits = 30
    max_cores = 12 * n_nodes
    processes = 1
    max_gpu_memory = 64 * n_nodes #GB
    gpu_limit = int( (max_gpu_memory*1024**3)/((2**max_qubits)*16) )
    use_gpu = True
    if use_gpu:
        processes = min(max_cores, gpu_limit)
  
    ################################################################### Hardware architecture analysis ##################################################################
    ts = time()
    optimization_level = 1
    qt_object = xxzz_qubit(d=(3,3))
    circuit = qt_object.circ
    compare_error_function = get_decoded_logical_error_xxzz(d=(3,3))
    min_size = circuit.num_qubits
    injection_points = [q for q in range(min_size)]
    available_qubits = range(min_size) # Use range(30) if you want to double transpile, accounting for the possibility of extra "routing qubits" used in the simulation
    topologies = get_coupling_maps(min_size=min_size)

    def check_cm_isomorphism(list_cm, target_cm):
        for tested_topology in list_cm:
            if nx.is_isomorphic(nx.Graph(tested_topology), nx.Graph(target_cm)):
                return True
        return False

    tested_topolgies = []
    args_dict_of_lists = {"circuits":[], "device_backends":[], "noise_models":[]}
    for topology_name, topology in topologies.items():
        try: # If forced available qubits is not connected, skip it
            device_backend = CustomBackend(active_qubits=available_qubits, coupling_map=topology, backend_name=topology_name)
        except Exception as e:
            continue
        t_circuit = transpile(circuit, device_backend, optimization_level=optimization_level, scheduling_method='asap', seed_transpiler=42)
        active_qubits = get_active_qubits(t_circuit)
        reduced_topology = filter_coupling_map(topology, active_qubits)
        if check_cm_isomorphism(tested_topolgies, reduced_topology):
            continue
        # Create a reduced CustomBackend and retranspile the circuit only if the active qubits in the first CustomBackend are more than the circuits's min_size
        if len(available_qubits) != min_size:
            device_backend = CustomBackend(active_qubits=active_qubits, coupling_map=reduced_topology, backend_name=topology_name)
            t_circuit = transpile(circuit, device_backend, optimization_level=optimization_level, scheduling_method='asap', seed_transpiler=42)
        noise_model = bitphase_flip_noise_model(physical_error, t_circuit.num_qubits)
        args_dict_of_lists["circuits"].append(t_circuit)
        args_dict_of_lists["device_backends"].append(device_backend)
        args_dict_of_lists["noise_models"].append(noise_model)
        tested_topolgies.append(reduced_topology)

    if not read_from_file:
        log(f"Generating dataset")
        ts = time()
        result_df = injection_campaign(circuits=args_dict_of_lists["circuits"], 
                                       device_backends=args_dict_of_lists["device_backends"],
                                       noise_models=args_dict_of_lists["noise_models"],
                                       injection_points=injection_points,
                                       transient_error_functions = transient_error_function,
                                       spread_depths = spread_depth,
                                       damping_functions = damping_function,
                                       transient_error_duration_ns = transient_error_duration_ns,
                                       n_quantised_steps = n_quantised_steps,
                                       processes=processes)

        log(f"Topological analysis done in {timedelta(seconds=time() - ts)}")
        with bz2.BZ2File(f"./results/{circuit.name} architecture_analysis", 'wb') as handle:
            pickle.dump(result_df, handle)
    else:
        log(f"Reading stored dataset")
        with bz2.BZ2File(f"./results/{circuit.name} architecture_analysis", 'rb') as handle:
            result_df = pickle.load(handle)
    plot_architecture_analysis(result_df, compare_error_function)

    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()

# %%
