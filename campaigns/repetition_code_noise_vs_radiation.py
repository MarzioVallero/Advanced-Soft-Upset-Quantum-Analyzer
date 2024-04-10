# %%
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asuqa import *
from utils import *
from visualisation import *

def main():
    ts = time()
    log(f"Job started at {datetime.fromtimestamp(ts)}.")
    log(f"Running noise vs. radiation-induced faults campaign on Repetition Qubit")

    ##################################################################### Transient error controls #####################################################################
    transient_error_function = reset_to_zero
    spread_depth = 10
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
    
    ####################################################### Noise vs. radiation-induced faults analysis analysis ########################################################
    ts = time()
    pe_qtcodes_circ = repetition_qubit(d=(5,1))
    pe_target_circuit = pe_qtcodes_circ.circ
    pe_device_backend = CustomBackend(active_qubits=range(pe_target_circuit.num_qubits))
    compare_error_function = get_decoded_logical_error_repetition(d=(5,1))
    pe_transpiled_circuit = transpile(pe_target_circuit, pe_device_backend, scheduling_method='asap', seed_transpiler=42)
    pe_injection_point = 2

    args_dict_of_lists = {"circuits":[], "device_backends":[], "noise_models":[]}
    pe_physical_error_list = [10**val for val in np.arange(-8, -0.4, 0.5)]
    for pe_physical_error in pe_physical_error_list:
        noise_model = bitphase_flip_noise_model(pe_physical_error, pe_transpiled_circuit.num_qubits)
        args_dict_of_lists["circuits"].append(pe_transpiled_circuit)
        args_dict_of_lists["device_backends"].append(pe_device_backend)
        args_dict_of_lists["noise_models"].append(noise_model)

    read_from_file = True
    if not read_from_file:
        ts = time()
        result_df = injection_campaign(circuits=args_dict_of_lists["circuits"],
                                            device_backends=args_dict_of_lists["device_backends"],
                                            noise_models=args_dict_of_lists["noise_models"],
                                            injection_points=pe_injection_point,
                                            transient_error_functions = transient_error_function,
                                            spread_depths = spread_depth,
                                            damping_functions = damping_function,
                                            transient_error_duration_ns = transient_error_duration_ns,
                                            n_quantised_steps = len(pe_physical_error_list),
                                            processes=processes,
                                            )
        log(f"Physical error analysis done in {timedelta(seconds=time() - ts)}")
        with bz2.BZ2File(f"./results/{pe_target_circuit.name} surface", 'wb') as handle:
            pickle.dump(result_df, handle)
    else:
        with bz2.BZ2File(f"./results/{pe_target_circuit.name} surface", 'rb') as handle:
            result_df = pickle.load(handle)
    plot_3d_surface(result_df, compare_error_function, ip=10)

    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()
# %%

# %%
