# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asuqa import *
from utils import *
from visualisation import *

def main():
    ts = time()
    log(f"Job started at {datetime.fromtimestamp(ts)}.")
    log(f"Running spreading fault vs. erasure fault on Repetition Qubit")
    read_from_file = True

    ##################################################################### Transient error controls #####################################################################
    transient_error_function = reset_to_zero
    damping_function = square_damping
    transient_error_duration_ns = 100_000_000
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

    ############################################################ Spreading fault vs. erasure fault analysis ############################################################
    ts = time()
    physical_error = 0.01
    qt_object = repetition_qubit(d=(15,1))
    circuit = qt_object.circ
    compare_error_function = get_decoded_logical_error_repetition(d=(15,1))
    injection_point = 2
    spread_depths = [0, 10]
    device_backends = [CustomBackend(active_qubits=range(circuit.num_qubits), coupling_map=mesh_edge_list, backend_name="Mesh")]
    
    if not read_from_file:
        spread_depth_list_dict = []
        for device_backend in device_backends:
            t_circuit = transpile(circuit, device_backend, scheduling_method='asap', seed_transpiler=42)
            noise_model = bitphase_flip_noise_model(physical_error, t_circuit.num_qubits)
            bfs_ordered_inj_list = [injection_point] + [e for (s, e) in nx.algorithms.bfs_tree(nx.Graph(device_backend.coupling_map), injection_point).edges()]
            injection_points = [ bfs_ordered_inj_list[0:limit] for limit in range(1, len(bfs_ordered_inj_list) + 1) ]

            result_df = injection_campaign(circuits=t_circuit,
                                           device_backends=device_backend,
                                           noise_models=noise_model,
                                           injection_points=injection_points,
                                           transient_error_functions = transient_error_function,
                                           spread_depths = spread_depths,
                                           damping_functions = damping_function,
                                           transient_error_duration_ns = transient_error_duration_ns,
                                           n_quantised_steps = 1,
                                           processes=processes)
            spread_depth_list_dict.append(result_df)
        concatenated_df = pd.concat(spread_depth_list_dict, ignore_index=True)
        log(f"Affected qubits analysis done in {timedelta(seconds=time() - ts)}")
        with bz2.BZ2File(f"./results/{circuit.name} spatial_spread_analysis", 'wb') as handle:
            pickle.dump(concatenated_df, handle)
    else:
        with bz2.BZ2File(f"./results/{circuit.name} spatial_spread_analysis", 'rb') as handle:
            concatenated_df = pickle.load(handle)
    plot_spatial_spread_analysis(concatenated_df, compare_error_function, subgroup_sizes=[1, 10, 11, 15, 16])
    
    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()

# %%
