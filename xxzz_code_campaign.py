#%%
from injector import *
from qtcodes import XXZZQubit, RotatedDecoder

def xxzz_qubit(d=3, T=1):
    xxzzd3 = XXZZQubit({"d":d})
    xxzzd3.reset_z()
    xxzzd3.stabilize()
    xxzzd3.x()
    xxzzd3.stabilize()
    xxzzd3.readout_z()
    xxzzd3.circ.name = f"XXZZ Qubit {d}"

    return xxzzd3

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
    decoder = RotatedDecoder({"d":d, "T":T})
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
    lattice_sizes = [(1,3), (3,1), (3,3), (3,5), (5,3)]
    cd_spread_depth = 2 # This spread depth covers only the number of qubits in the repretition (1,3)

    args_dict_of_lists = {"circuits":[], "device_backends":[], "noise_models":[]}
    for d in lattice_sizes:
        cd_object = xxzz_qubit(d=d)
        cd_circuit = cd_object.circ
        device_backend = CustomBackend(active_qubits=range(cd_circuit.num_qubits))
        cd_transpiled_circuit = transpile(cd_circuit, device_backend, scheduling_method='asap', initial_layout=list(range(cd_circuit.num_qubits)), seed_transpiler=42)
        noise_model = bitphase_flip_noise_model(physical_error, cd_transpiled_circuit.num_qubits)
        args_dict_of_lists["circuits"].append(cd_transpiled_circuit)
        args_dict_of_lists["device_backends"].append(device_backend)
        args_dict_of_lists["noise_models"].append(noise_model)

    read_from_file = False
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
                                            )
        log(f"Code distance analysis simulation done in {timedelta(seconds=time() - ts)}")
        with bz2.BZ2File(f"./results/{cd_circuit.name} code_distance_analysis", 'wb') as handle:
            pickle.dump(result_df, handle)
    else:
        with bz2.BZ2File(f"./results/{cd_circuit.name} code_distance_analysis", 'rb') as handle:
            result_df = pickle.load(handle)
    plot_injection_logical_error(result_df, get_decoded_logical_error)

    ##################################################################### Affected qubits analysis #####################################################################
    log(f"Affected qubits analysis started at {datetime.fromtimestamp(time())}")
    ts = time()
    aq_object = xxzz_qubit(d=(3,3))
    aq_circuit = aq_object.circ
    compare_error_function = get_decoded_logical_error(d=(3,3))
    aq_injection_point = 2
    aq_spread_depths = [0, 10]
    aq_device_backends = [CustomBackend(active_qubits=range(aq_circuit.num_qubits), coupling_map=mesh_edge_list, backend_name="Mesh"),
                          CustomBackend(active_qubits=range(aq_circuit.num_qubits), coupling_map=line_edge_list, backend_name="Linear"),
                          CustomBackend(active_qubits=range(aq_circuit.num_qubits), coupling_map=complete_edge_list, backend_name="Complete")]
    
    read_from_file = False
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
                                                            )
            spread_depth_list_dict.append(result_df)
        concatenated_df = pd.concat(spread_depth_list_dict, ignore_index=True)
        log(f"Affected qubits analysis done in {timedelta(seconds=time() - ts)}")
        with bz2.BZ2File(f"./results/{aq_circuit.name} histogram_affected_qubits", 'wb') as handle:
            pickle.dump(concatenated_df, handle)
    else:
        with bz2.BZ2File(f"./results/{aq_circuit.name} histogram_affected_qubits", 'rb') as handle:
            concatenated_df = pickle.load(handle)
    plot_histogram_error(concatenated_df, compare_error_function)
    
    ##################################################################### Physical error analysis ######################################################################
    log(f"Physical error analysis started at {datetime.fromtimestamp(time())}")
    ts = time()
    pe_qtcodes_circ = xxzz_qubit(d=(3,3))
    pe_target_circuit = pe_qtcodes_circ.circ
    pe_device_backend = CustomBackend(active_qubits=range(pe_target_circuit.num_qubits))
    compare_error_function = get_decoded_logical_error(d=(3,3))
    pe_transpiled_circuit = transpile(pe_target_circuit, pe_device_backend, scheduling_method='asap', initial_layout=list(range(pe_target_circuit.num_qubits)), seed_transpiler=42)
    pe_injection_point = 2

    args_dict_of_lists = {"circuits":[], "device_backends":[], "noise_models":[]}
    pe_physical_error_list = [10**val for val in np.arange(-8, -0.4, 0.5)]
    for pe_physical_error in pe_physical_error_list:
        noise_model = bitphase_flip_noise_model(pe_physical_error, pe_transpiled_circuit.num_qubits)
        args_dict_of_lists["circuits"].append(pe_transpiled_circuit)
        args_dict_of_lists["device_backends"].append(pe_device_backend)
        args_dict_of_lists["noise_models"].append(noise_model)

    read_from_file = False
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

    ####################################################################### Topological analysis #######################################################################
    log(f"Topological analysis started at {datetime.fromtimestamp(time())}")
    ts = time()
    optimization_level = 1
    ta_qtcodes_circ = xxzz_qubit(d=(3,3))
    ta_target_circuit = ta_qtcodes_circ.circ
    compare_error_function = get_decoded_logical_error(d=(3,3))
    min_size = ta_target_circuit.num_qubits
    ta_injection_points = [q for q in range(min_size)]
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
        transpiled_circuit = transpile(ta_target_circuit, device_backend, optimization_level=optimization_level, scheduling_method='asap', initial_layout=list(range(ta_target_circuit.num_qubits)), seed_transpiler=42)
        active_qubits = get_active_qubits(transpiled_circuit)
        reduced_topology = filter_coupling_map(topology, active_qubits)
        if check_cm_isomorphism(tested_topolgies, reduced_topology):
            continue
        # Create a reduced CustomBackend and retranspile the circuit only if the active qubits in the first CustomBackend are more than the circuits's min_size
        if len(available_qubits) != min_size:
            device_backend = CustomBackend(active_qubits=active_qubits, coupling_map=reduced_topology, backend_name=topology_name)
            transpiled_circuit = transpile(ta_target_circuit, device_backend, optimization_level=optimization_level, scheduling_method='asap', initial_layout=list(range(ta_target_circuit.num_qubits)), seed_transpiler=42)
        noise_model = bitphase_flip_noise_model(physical_error, transpiled_circuit.num_qubits)
        args_dict_of_lists["circuits"].append(transpiled_circuit)
        args_dict_of_lists["device_backends"].append(device_backend)
        args_dict_of_lists["noise_models"].append(noise_model)
        tested_topolgies.append(reduced_topology)

    read_from_file = False
    if not read_from_file:
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
                                            )

        log(f"Topological analysis done in {timedelta(seconds=time() - ts)}")
        with bz2.BZ2File(f"./results/{ta_target_circuit.name} topologies_analysis", 'wb') as handle:
            pickle.dump(result_df, handle)
    else:
        with bz2.BZ2File(f"./results/{ta_target_circuit.name} topologies_analysis", 'rb') as handle:
            result_df = pickle.load(handle)
    plot_topology_injection_point_error(result_df, compare_error_function)

    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()

# %%
