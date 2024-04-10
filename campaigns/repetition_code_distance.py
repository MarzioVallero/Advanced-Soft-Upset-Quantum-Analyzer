# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asuqa import *
from utils import *
from visualisation import *

def main():
    ts = time()
    log(f"Job started at {datetime.fromtimestamp(ts)}.")
    log(f"Running code distance campaign on Repetition Qubit")

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
    
    ###################################################################### Code distance analysis ######################################################################
    ts = time()
    physical_error = 0.01
    lattice_sizes = [(n,1) for n in range(3, 16, 2)]
    mq_circuit = repetition_qubit(d=lattice_sizes[-1]).circ
    mq_spread_depth = 0 # Only inject root qubits

    read_from_file = True
    if not read_from_file:
        list_result_df = []
        for d in lattice_sizes:
            mq_object = repetition_qubit(d=d)
            mq_circuit = mq_object.circ
            mq_device_backend = CustomBackend(active_qubits=range(mq_circuit.num_qubits))
            mq_transpiled_circuit = transpile(mq_circuit, mq_device_backend, scheduling_method='asap', seed_transpiler=42)
            noise_model = bitphase_flip_noise_model(physical_error, mq_transpiled_circuit.num_qubits)
            
            n_qubits = mq_transpiled_circuit.num_qubits
            coupling_map = mq_device_backend.configuration().coupling_map

            injection_points = get_some_connected_subgraphs(nx.Graph(coupling_map), n_qubits)
            injection_points = [list(el) for el in injection_points]
            injection_points.sort(key=lambda t: len(t), reverse=False)

            result_df = injection_campaign(circuits=mq_transpiled_circuit,
                                           device_backends=mq_device_backend,
                                           noise_models=noise_model,
                                           injection_points=injection_points,
                                           transient_error_functions = transient_error_function,
                                           spread_depths = mq_spread_depth,
                                           damping_functions = damping_function,
                                           transient_error_duration_ns = transient_error_duration_ns,
                                           n_quantised_steps = 1,
                                           processes=processes)
            list_result_df.append(result_df)
        concatenated_df = pd.concat(list_result_df, ignore_index=True)
        log(f"Code distance analysis simulation done in {timedelta(seconds=time() - ts)}")
        with bz2.BZ2File(f"./results/{mq_circuit.name} minimum_inj_qubits", 'wb') as handle:
            pickle.dump(concatenated_df, handle)
    else:
        with bz2.BZ2File(f"./results/{mq_circuit.name} minimum_inj_qubits", 'rb') as handle:
            concatenated_df = pickle.load(handle)
    plot_minimum_inj_qubits(concatenated_df, get_decoded_logical_error_repetition, threshold_min=0.01)

    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()
# %%

# %%
