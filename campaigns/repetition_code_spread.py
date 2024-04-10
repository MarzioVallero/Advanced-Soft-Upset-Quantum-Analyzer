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

    ##################################################################### Transient error controls #####################################################################
    transient_error_function = reset_to_zero
    damping_function = square_damping
    transient_error_duration_ns = 100_000_000
    n_quantised_steps = 10

    ################################################################# Simulation performance controls ##################################################################
    n_nodes = 1
    max_qubits = 30
    max_cores = 12 * n_nodes
    max_gpu_memory = 64 * n_nodes #GB
    gpu_limit = int( (max_gpu_memory*1024**3)/((2**max_qubits)*16) )
    use_gpu = True
    processes = min(max_cores, gpu_limit, n_quantised_steps)
    if not use_gpu and (processes > n_quantised_steps or processes > max_cores):
        processes = min(n_quantised_steps+1, max_cores)

    ############################################################ Spreading fault vs. erasure fault analysis ############################################################
    ts = time()
    physical_error = 0.01
    aq_object = repetition_qubit(d=(15,1))
    aq_circuit = aq_object.circ
    compare_error_function = get_decoded_logical_error_repetition(d=(15,1))
    aq_injection_point = 2
    aq_spread_depths = [0, 10]
    aq_device_backends = [CustomBackend(active_qubits=range(aq_circuit.num_qubits), coupling_map=mesh_edge_list, backend_name="Mesh")]
    
    read_from_file = True
    if not read_from_file:
        spread_depth_list_dict = []
        for aq_backend in aq_device_backends:
            aq_transpiled_circuit = transpile(aq_circuit, aq_backend, scheduling_method='asap', seed_transpiler=42)
            noise_model = bitphase_flip_noise_model(physical_error, aq_transpiled_circuit.num_qubits)
            bfs_ordered_inj_list = [aq_injection_point] + [e for (s, e) in nx.algorithms.bfs_tree(nx.Graph(aq_backend.coupling_map), aq_injection_point).edges()]
            injection_points = [ bfs_ordered_inj_list[0:limit] for limit in range(1, len(bfs_ordered_inj_list) + 1) ]

            result_df = injection_campaign(circuits=aq_transpiled_circuit,
                                                    device_backends=aq_backend,
                                                    noise_models=noise_model,
                                                    injection_points=injection_points,
                                                    transient_error_functions = transient_error_function,
                                                    spread_depths = aq_spread_depths,
                                                    damping_functions = damping_function,
                                                    transient_error_duration_ns = transient_error_duration_ns,
                                                    n_quantised_steps = 1,
                                                    processes=processes,
                                                            )
            spread_depth_list_dict.append(result_df)
        concatenated_df = pd.concat(spread_depth_list_dict, ignore_index=True)
        log(f"Affected qubits analysis done in {timedelta(seconds=time() - ts)}")
        with bz2.BZ2File(f"./results/{aq_circuit.name} histogram_affected_qubits", 'wb') as handle:
            pickle.dump(concatenated_df, handle)
    else:
        with bz2.BZ2File(f"./results/{aq_circuit.name} histogram_affected_qubits", 'rb') as handle:
            concatenated_df = pickle.load(handle)
    plot_histogram_error(concatenated_df, compare_error_function, subgroup_sizes=[1, 10, 11, 15, 16])
    
    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()
