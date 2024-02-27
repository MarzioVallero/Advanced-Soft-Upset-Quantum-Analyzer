# %%
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
    repetition_q.circ.name = f"Repetition Qubit {d}"

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
    max_cores = 24
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
    log(f"Code distance analysis started at {datetime.fromtimestamp(time())}")
    ts = time()
    physical_error = 0.01
    lattice_sizes = [n for n in range(3, 16, 2)]
    cd_spread_depth = 2 # This spread depth covers only the number of qubits in the repretition (3,1)

    args_dict_of_lists = {"circuits":[], "device_backends":[], "noise_models":[]}
    for d in lattice_sizes:
        cd_object = repetition_qubit(d=d)
        cd_circuit = cd_object.circ
        device_backend = CustomBackend(active_qubits=range(cd_circuit.num_qubits))
        cd_transpiled_circuit = transpile(cd_circuit, device_backend, scheduling_method='asap', initial_layout=list(range(cd_circuit.num_qubits)), seed_transpiler=42)
        noise_model = bitphase_flip_noise_model(physical_error, cd_transpiled_circuit.num_qubits)
        args_dict_of_lists["circuits"].append(cd_transpiled_circuit)
        args_dict_of_lists["device_backends"].append(device_backend)
        args_dict_of_lists["noise_models"].append(noise_model)

    read_from_file = True
    if not read_from_file:
        result_df = injection_campaign(circuits=args_dict_of_lists["circuits"], 
                                            device_backends=args_dict_of_lists["device_backends"],
                                            noise_models=args_dict_of_lists["noise_models"],
                                            injection_points=injection_point,
                                            transient_error_functions = transient_error_function,
                                            spread_depths = cd_spread_depth,
                                            damping_functions = damping_function,
                                            transient_error_duration_ns = transient_error_duration_ns,
                                            n_quantised_steps = n_quantised_steps,
                                            processes=processes,
                                            save=False
                                            )
        log(f"Code distance analysis simulation done in {timedelta(seconds=time() - ts)}")
        with open(f"./results/{cd_circuit.name}_code_distance_analysis", 'wb') as handle:
            dill.dump(result_df, handle, protocol=dill.HIGHEST_PROTOCOL)
    else:
        with open(f"./results/{cd_circuit.name}_code_distance_analysis", 'rb') as pickle_file:
            result_df = dill.load(pickle_file)
    # plot_injection_logical_error(result_df, get_decoded_logical_error)

    ##################################################################### Affected qubits analysis #####################################################################
    log(f"Affected qubits analysis started at {datetime.fromtimestamp(time())}")
    ts = time()
    aq_object = repetition_qubit(d=(15, 1))
    aq_circuit = aq_object.circ
    compare_error_function = get_decoded_logical_error(d=(15, 1))
    aq_injection_point = 2
    aq_spread_depths = [0, 10]
    aq_device_backends = [CustomBackend(active_qubits=range(aq_circuit.num_qubits), coupling_map=mesh_edge_list, backend_name="Mesh"),
                          CustomBackend(active_qubits=range(aq_circuit.num_qubits), coupling_map=line_edge_list, backend_name="Linear"),
                          CustomBackend(active_qubits=range(aq_circuit.num_qubits), coupling_map=complete_edge_list, backend_name="Complete")]
    
    read_from_file = True
    if not read_from_file:
        spread_depth_list_dict = []
        for aq_backend in aq_device_backends:
            aq_transpiled_circuit = transpile(aq_circuit, aq_backend, scheduling_method='asap', initial_layout=list(range(aq_circuit.num_qubits)), seed_transpiler=42)
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
                                                    save=False
                                                    )
            spread_depth_list_dict.append(result_df)
        concatenated_df = pd.concat(spread_depth_list_dict, ignore_index=True)
        log(f"Spread depth analysis simulation done in {timedelta(seconds=time() - ts)}")
        with open(f"./results/{aq_circuit.name}_histogram_affected_qubits", 'wb') as handle:
            dill.dump(concatenated_df, handle, protocol=dill.HIGHEST_PROTOCOL)
    else:
        with open(f"./results/{aq_circuit.name}_histogram_affected_qubits", 'rb') as pickle_file:
            concatenated_df = dill.load(pickle_file)
    # plot_histogram_error(concatenated_df, compare_error_function)
    
    ##################################################################### Physical error analysis ######################################################################
    log(f"Physical error analysis started at {datetime.fromtimestamp(time())}")
    ts = time()
    pe_qtcodes_circ = repetition_qubit(d=5)
    pe_target_circuit = pe_qtcodes_circ.circ
    pe_device_backend = CustomBackend(active_qubits=range(pe_target_circuit.num_qubits))
    compare_error_function = get_decoded_logical_error(d=5)
    pe_transpiled_circuit = transpile(pe_target_circuit, pe_device_backend, scheduling_method='asap', initial_layout=list(range(pe_target_circuit.num_qubits)), seed_transpiler=42)
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
                                            save=False
                                            )
        log(f"Surface plot simulation done in {timedelta(seconds=time() - ts)}")
        with open(f"./results/{pe_target_circuit.name}_surfaceplot", 'wb') as handle:
            dill.dump(result_df, handle, protocol=dill.HIGHEST_PROTOCOL)
    else:
        with open(f"./results/{pe_target_circuit.name}_surfaceplot", 'rb') as pickle_file:
            result_df = dill.load(pickle_file)
    # plot_3d_surface(result_df, compare_error_function, ip=4)

    ####################################################################### Topological analysis #######################################################################
    log(f"Topological analysis started at {datetime.fromtimestamp(time())}")
    ts = time()
    optimization_level = 1
    ta_qtcodes_circ = repetition_qubit(d=15)
    ta_target_circuit = ta_qtcodes_circ.circ
    compare_error_function = get_decoded_logical_error(d=15)
    min_size = ta_target_circuit.num_qubits
    ta_injection_points = [q for q in range(min_size)]
    available_qubits = range(30)
    topologies = get_coupling_maps(min_size=min_size)

    def check_cm_isomorphism(list_cm, target_cm):
        for tested_topology in list_cm:
            if nx.is_isomorphic(nx.Graph(tested_topology), nx.Graph(target_cm)):
                return True
        return False

    tested_topolgies = []
    args_dict_of_lists = {"circuits":[], "device_backends":[], "noise_models":[]}
    for topology_name, topology in topologies.items():
        device_backend = CustomBackend(active_qubits=available_qubits, coupling_map=topology, backend_name=topology_name)
        transpiled_circuit = transpile(ta_target_circuit, device_backend, optimization_level=optimization_level, scheduling_method='asap', initial_layout=list(range(ta_target_circuit.num_qubits)), seed_transpiler=42)
        active_qubits = get_active_qubits(transpiled_circuit)
        reduced_topology = filter_coupling_map(topology, active_qubits)
        if check_cm_isomorphism(tested_topolgies, reduced_topology):
            continue
        # Create a reduced CustomBackend and retranspile the circuit
        device_backend = CustomBackend(active_qubits=active_qubits, coupling_map=reduced_topology, backend_name=topology_name)
        transpiled_circuit = transpile(ta_target_circuit, device_backend, optimization_level=optimization_level, scheduling_method='asap', initial_layout=list(range(ta_target_circuit.num_qubits)), seed_transpiler=42)
        noise_model = bitphase_flip_noise_model(physical_error, transpiled_circuit.num_qubits)
        args_dict_of_lists["circuits"].append(transpiled_circuit)
        args_dict_of_lists["device_backends"].append(device_backend)
        args_dict_of_lists["noise_models"].append(noise_model)
        tested_topolgies.append(reduced_topology)

    read_from_file = False
    if not read_from_file:
        # Graph plot: loop over root injection qubits
        ts = time()
        result_df = injection_campaign(circuits=args_dict_of_lists["circuits"], 
                                            device_backends=args_dict_of_lists["device_backends"],
                                            noise_models=args_dict_of_lists["noise_models"],
                                            injection_points=ta_injection_points,
                                            transient_error_functions = transient_error_function,
                                            spread_depths = spread_depth,
                                            damping_functions = damping_function,
                                            transient_error_duration_ns = transient_error_duration_ns,
                                            n_quantised_steps = n_quantised_steps,
                                            processes=processes,
                                            save=False
                                            )

        log(f"Topology plot simulation done in {timedelta(seconds=time() - ts)}")
        with open(f"./results/{ta_target_circuit.name}_topologies_analysis", 'wb') as handle:
            dill.dump(result_df, handle, protocol=dill.HIGHEST_PROTOCOL)
    else:
        with open(f"./results/{ta_target_circuit.name}_topologies_analysis", 'rb') as pickle_file:
            result_df = dill.load(pickle_file)
    # plot_topology_injection_point_error(result_df, compare_error_function)

    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()

# %%
