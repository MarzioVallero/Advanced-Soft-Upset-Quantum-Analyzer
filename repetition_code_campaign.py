#%%
from injector import *
from qtcodes import RepetitionQubit, RepetitionDecoder

def repetition_qubit(d=3, T=1):
    T = 1
    repetition_q = RepetitionQubit({"d":d},"t")
    repetition_q.reset_z()
    repetition_q.stabilize()
    repetition_q.x()
    repetition_q.stabilize()
    repetition_q.readout_z()
    repetition_q.circ.name = "Repetition Qubit"

    return repetition_q

def qtcodes_decoded_logical_readout_error(decoder, readout_type, golden_logical_bistring, golden_counts, inj_counts):
    wrong_logical_bitstring_count = 0
    total_measurements = 0
    for bitstring, count in inj_counts.items():
        logical_bitstring = decoder.correct_readout(bitstring, readout_type)
        if logical_bitstring != golden_logical_bistring:
            wrong_logical_bitstring_count += count
        total_measurements += count
        
    return (wrong_logical_bitstring_count / total_measurements)

def get_decoded_logical_error(d=3, T=1):
    decoder = RepetitionDecoder({"d":d, "T":T})
    readout_type = "Z"
    expected_logical_output = 1

    logical_readout_qtcodes = partial(qtcodes_decoded_logical_readout_error, decoder, readout_type, expected_logical_output)
    logical_readout_qtcodes.__name__ = f"decoder_{readout_type}_{d}"

    return logical_readout_qtcodes

def main():
    ts = time()
    log(f"Job started at {datetime.fromtimestamp(ts)}.")

    ##################################################################### Transient error controls #####################################################################
    injection_point = 2
    transient_error_function = reset_to_zero
    spread_depth = 10
    damping_function = square_damping
    transient_error_duration_ns = 25000000
    n_quantised_steps = 10

    ################################################################# Simulation performance controls ##################################################################
    max_qubits = 30
    max_cores = 8
    max_gpu_memory = 64 #GB
    gpu_limit = int( (max_gpu_memory*1024**3)/((2**max_qubits)*8) )
    use_gpu = True
    processes = min(max_cores, gpu_limit, n_quantised_steps)
    if not use_gpu and (processes > n_quantised_steps or processes > max_cores):
        processes = min(n_quantised_steps+1, max_cores)

    # To make a shot-level simulation, use this with CAUTION!
    # shot_time_per_circuit = get_shot_execution_time_ns(circuit)
    # n_quantised_steps = transient_error_duration_ns // int(shot_time_per_circuit))
        
    ##################################################################### Code distance analysis #######################################################################
    cd_result_dict_list = []
    physical_error = 0.01
    read_from_file = False
    if not read_from_file:
        lattice_sizes = [(n, 1) for n in range(3, 16, 2)]
        for d in lattice_sizes:
            cd_object = repetition_qubit(d=d)
            cd_circuit = cd_object.circ
            device_backend = CustomBackend(active_qubits=range(cd_circuit.num_qubits))
            cd_transpiled_circuit = transpile(cd_circuit, device_backend, scheduling_method='asap', seed_transpiler=42)
            noise_model = bitphase_flip_noise_model(physical_error, cd_transpiled_circuit.num_qubits)
            result_dict = run_transient_injection(cd_transpiled_circuit, 
                                                    device_backend=device_backend,
                                                    noise_model=noise_model,
                                                    injection_point=injection_point,
                                                    transient_error_function = transient_error_function,
                                                    spread_depth = spread_depth,
                                                    damping_function = damping_function,
                                                    transient_error_duration_ns = transient_error_duration_ns,
                                                    n_quantised_steps = n_quantised_steps,
                                                    processes=processes,
                                                    save=False
                                                    )
            cd_compare_function = get_decoded_logical_error(d=d)
            result_dict["runtime_compare_function"] = cd_compare_function
            cd_result_dict_list.append(result_dict)
        log(f"Code distance analysis simulation done in {timedelta(seconds=time() - ts)}")
        with open(f"./results/{cd_circuit.name}_code_distance_analysis", 'wb') as handle:
            dill.dump(cd_result_dict_list, handle, protocol=dill.HIGHEST_PROTOCOL)
    else:
        cd_object = repetition_qubit(d=3)
        cd_circuit = cd_object.circ
        with open(f"./results/{cd_circuit.name}_code_distance_analysis", 'rb') as pickle_file:
            cd_result_dict_list = dill.load(pickle_file)
    plot_injection_logical_error(cd_result_dict_list, get_decoded_logical_error())

    ##################################################################### Affected qubits analysis #####################################################################
    qtcodes_circ = repetition_qubit(d=(15, 1))
    target_circuit = qtcodes_circ.circ
    device_backend = CustomBackend(active_qubits=range(target_circuit.num_qubits))
    compare_error_function = get_decoded_logical_error(d=(15, 1))
    aq_transpiled_circuit = transpile(target_circuit, device_backend, scheduling_method='asap', seed_transpiler=42)
    noise_model = bitphase_flip_noise_model(physical_error, aq_transpiled_circuit.num_qubits)
    aq_injection_point = 12
    bfs_ordered_inj_list = [aq_injection_point] + [e for (s, e) in nx.algorithms.bfs_tree(nx.Graph(device_backend.coupling_map), aq_injection_point).edges()]

    read_from_file = True
    if not read_from_file:
        spread_depth_list_dict = []
        for limit in range(1, len(bfs_ordered_inj_list) + 1):
            result_dict = run_transient_injection(aq_transpiled_circuit,
                                                    device_backend=device_backend,
                                                    noise_model=noise_model,
                                                    injection_point=bfs_ordered_inj_list[0:limit],
                                                    transient_error_function = transient_error_function,
                                                    spread_depth = 0,
                                                    damping_function = lambda depth: 1.0,
                                                    transient_error_duration_ns = transient_error_duration_ns,
                                                    n_quantised_steps = 10,
                                                    processes=processes,
                                                    save=False
                                                    )
            spread_depth_list_dict.append(result_dict)
        log(f"Spread depth analysis simulation done in {timedelta(seconds=time() - ts)}")
        with open(f"./results/{target_circuit.name}_histogram_affected_qubits", 'wb') as handle:
            dill.dump(spread_depth_list_dict, handle, protocol=dill.HIGHEST_PROTOCOL)
    else:
        with open(f"./results/{target_circuit.name}_histogram_affected_qubits", 'rb') as pickle_file:
            spread_depth_list_dict = dill.load(pickle_file)
    plot_histogram_error(spread_depth_list_dict, compare_error_function)

    ##################################################################### Physical error analysis ######################################################################
    qtcodes_circ = repetition_qubit()
    target_circuit = qtcodes_circ.circ
    device_backend = CustomBackend(active_qubits=range(target_circuit.num_qubits))
    compare_error_function = get_decoded_logical_error()
    transpiled_circuit = transpile(target_circuit, device_backend, scheduling_method='asap', seed_transpiler=42)

    read_from_file = False
    if not read_from_file:
        physical_error_list = [10**val for val in np.arange(-8, -1.5, 0.5)]
        surface_plot_result_dict = []
        for physical_error in physical_error_list:
            # Run transient injection simulation
            ts = time()
            noise_model = bitphase_flip_noise_model(physical_error, target_circuit.num_qubits)
            result_dict = run_transient_injection(transpiled_circuit, 
                                                device_backend=device_backend,
                                                noise_model=noise_model,
                                                injection_point=injection_point,
                                                transient_error_function = transient_error_function,
                                                spread_depth = spread_depth,
                                                damping_function = damping_function,
                                                transient_error_duration_ns = transient_error_duration_ns,
                                                n_quantised_steps = n_quantised_steps,
                                                processes=processes,
                                                save=False
                                                )
            surface_plot_result_dict.append(result_dict)
            log(f"Surface plot simulation done in {timedelta(seconds=time() - ts)}")
        with open(f"./results/{target_circuit.name}_surfaceplot", 'wb') as handle:
            dill.dump(surface_plot_result_dict, handle, protocol=dill.HIGHEST_PROTOCOL)
    else:
        with open(f"./results/{target_circuit.name}_surfaceplot", 'rb') as pickle_file:
            surface_plot_result_dict = dill.load(pickle_file)
    plot_3d_surface(surface_plot_result_dict, compare_error_function, ip=4)

    ####################################################################### Topological analysis #######################################################################
    read_from_file = False
    optimization_level = 1
    min_size = target_circuit.num_qubits
    available_qubits = range(30)
    topologies = get_coupling_maps(min_size=min_size)
    noise_model = bitphase_flip_noise_model(0.01, target_circuit.num_qubits)
    qubits_list = [q for q in range(target_circuit.num_qubits)]
    topologies_results = {}

    if not read_from_file:
        for topology_name, topology in topologies.items():
        # Graph plot: loop over root injection qubits
            ts = time()
            # Transpile a first time to find which qubits are active
            device_backend = CustomBackend(active_qubits=available_qubits, coupling_map=topology)
            transpiled_circuit = transpile(target_circuit, device_backend, optimization_level=optimization_level, scheduling_method='asap', seed_transpiler=42)
            active_qubits = get_active_qubits(transpiled_circuit)
            reduced_topology = filter_coupling_map(topology, active_qubits)
            # Create a reduced CustomBackend and retranspile the circuit
            device_backend = CustomBackend(active_qubits=active_qubits, coupling_map=reduced_topology)
            transpiled_circuit = transpile(target_circuit, device_backend, optimization_level=optimization_level, scheduling_method='asap', seed_transpiler=42)
            log(f"Transpilation on {topology_name} topology done in {timedelta(seconds=time() - ts)}")

            graph_plot_result_dict = {}
            list_injected_df = []
            for root_injection_point in qubits_list:
                # Run transient injection simulation
                result_dict = run_transient_injection(transpiled_circuit, 
                                                    device_backend=device_backend,
                                                    noise_model=noise_model,
                                                    injection_point=root_injection_point,
                                                    transient_error_function = transient_error_function,
                                                    spread_depth = spread_depth,
                                                    damping_function = damping_function,
                                                    transient_error_duration_ns = transient_error_duration_ns,
                                                    n_quantised_steps = n_quantised_steps,
                                                    processes=processes,
                                                    save=False
                                                    )
                list_injected_df.append(result_dict["injected_df"])
            log(f"Topology plot {topology_name} simulation done in {timedelta(seconds=time() - ts)}")
            graph_plot_result_dict = result_dict
            graph_plot_result_dict["injected_df"] = pd.concat(list_injected_df)
            topologies_results[topology_name] = graph_plot_result_dict
        with open(f"./results/{target_circuit.name}_topologies_analysis", 'wb') as handle:
            dill.dump(topologies_results, handle, protocol=dill.HIGHEST_PROTOCOL)
    else:
        with open(f"./results/{target_circuit.name}_topologies_analysis", 'rb') as pickle_file:
            topologies_results = dill.load(pickle_file)
    plot_topology_injection_point_error(topologies_results, compare_error_function, topology_name)

    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()

# %%
