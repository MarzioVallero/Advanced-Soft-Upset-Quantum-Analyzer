# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asuqa import *
from utils import *
from visualisation import *

def main():
    ts = time()
    log(f"Job started at {datetime.fromtimestamp(ts)}.")
    log(f"Code distance campaign on Repetition Qubit")
    read_from_file = True

    ##################################################################### Transient error controls #####################################################################
    transient_error_function = reset_to_zero
    damping_function = square_damping
    transient_error_duration_ns = 100_000_000
    n_quantised_steps = 1

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
    
    ###################################################################### Code distance analysis ######################################################################
    ts = time()
    spread_depth = 0
    physical_error = 0.01
    lattice_sizes = [(n,1) for n in range(3, 16, 2)]
    circuit = repetition_qubit(d=lattice_sizes[-1]).circ

    if not read_from_file:
        log(f"Generating dataset")
        list_result_df = []
        for d in lattice_sizes:
            qt_object = repetition_qubit(d=d)
            circuit = qt_object.circ
            device_backend = CustomBackend(active_qubits=range(circuit.num_qubits))
            t_circuit = transpile(circuit, device_backend, scheduling_method='asap', seed_transpiler=42)
            noise_model = bitphase_flip_noise_model(physical_error, t_circuit.num_qubits)
            
            n_qubits = t_circuit.num_qubits
            coupling_map = device_backend.configuration().coupling_map

            for spread_depth, group_size in [(0,1), (10,1), (0,int(n_qubits/2))]:
                injection_points = get_some_connected_subgraphs(nx.Graph(coupling_map), group_size)

                result_df = injection_campaign(circuits=t_circuit,
                                               device_backends=device_backend,
                                               noise_models=noise_model,
                                               injection_points=injection_points,
                                               transient_error_functions = transient_error_function,
                                               spread_depths = spread_depth,
                                               damping_functions = damping_function,
                                               transient_error_duration_ns = transient_error_duration_ns,
                                               n_quantised_steps = n_quantised_steps,
                                               processes=processes)
                list_result_df.append(result_df)
        concatenated_df = pd.concat(list_result_df, ignore_index=True)
        log(f"Code distance analysis simulation done in {timedelta(seconds=time() - ts)}")
        with bz2.BZ2File(f"./results/{circuit.name} sd{spread_depth} code_distance_analysis", 'wb') as handle:
            pickle.dump(concatenated_df, handle)
    else:
        log(f"Reading stored dataset")
        with bz2.BZ2File(f"./results/{circuit.name} sd{spread_depth} code_distance_analysis", 'rb') as handle:
            concatenated_df = pickle.load(handle)
    plot_code_distance_analysis(concatenated_df, get_decoded_logical_error_repetition)

    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()
# %%
