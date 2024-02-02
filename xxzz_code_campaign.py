#%%
from injector import *
from qtcodes import XXZZQubit, RotatedDecoder

def main():
    ts = time()
    log(f"Job started at {datetime.fromtimestamp(ts)}.")

    d=3
    T=1
    # Surface code 3,3 XXZZ
    xxzzd3 = XXZZQubit({'d':d})
    xxzzd3.stabilize()
    xxzzd3.stabilize()
    xxzzd3.x()
    xxzzd3.stabilize()
    xxzzd3.readout_z()
    xxzzd3.circ.name = "XXZZ d3 Qubit"

    # Backend and circuit selection
    target_circuit = xxzzd3.circ
    qtcodes_circ = xxzzd3
    device_backend = CustomBackend(active_qubits=range(target_circuit.num_qubits))

    # Readout and error correction
    decoder = RotatedDecoder({"d":d,"T":T})
    readout_type = "Z"
    expected_logical_output = 1

    # Raw logical error ratio compare_function
    def qtcodes_logical_readout_error(qtcodes_obj, golden_logical_bistring, golden_counts, inj_counts):
        wrong_logical_bitstring_count = 0
        total_measurements = 0
        for bitstring, count in inj_counts.items():
            logical_bitstring = qtcodes_obj.parse_readout(bitstring, "Z")[0]
            if logical_bitstring != golden_logical_bistring:
                wrong_logical_bitstring_count += count
            total_measurements += count
            
        return (wrong_logical_bitstring_count / total_measurements)

    logical_readout_qtcodes = partial(qtcodes_logical_readout_error, qtcodes_circ, expected_logical_output)
    logical_readout_qtcodes.__name__ = "logical_readout_error"

    # Decoded logical error ratio compare_function
    def qtcodes_decoded_logical_readout_error(decoder, readout_type, golden_logical_bistring, golden_counts, inj_counts):
        wrong_logical_bitstring_count = 0
        total_measurements = 0
        for bitstring, count in inj_counts.items():
            logical_bitstring = decoder.correct_readout(bitstring, readout_type)
            if logical_bitstring != golden_logical_bistring:
                wrong_logical_bitstring_count += count
            total_measurements += count
            
        return (wrong_logical_bitstring_count / total_measurements)

    decoded_logical_readout_qtcodes = partial(qtcodes_decoded_logical_readout_error, decoder, readout_type, expected_logical_output)
    decoded_logical_readout_qtcodes.__name__ = "decoder_logical_readout_error"

    transpiled_circuit = transpile(target_circuit, device_backend, scheduling_method='asap', seed_transpiler=42)
    log(f"Initialisation and transpilation done in {timedelta(seconds=time() - ts)}")

    ##################################################################### Transient error controls #####################################################################
    injection_point = 2
    transient_error_function = reset_to_zero
    spread_depth = 10
    damping_function = square_damping
    transient_error_duration_ns = 25000000
    n_quantised_steps = 10

    ################################################################# Simulation performance controls ##################################################################
    max_qubits = 25
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

    ##################################################################### Physical error analysis ######################################################################
    read_from_file = False
    if not read_from_file:
        physical_error_list = [10**val for val in np.arange(-8, -1.5, 0.5)]
        surface_plot_result_dict = []
        for physical_error in physical_error_list:
            # Run transient injection simulation
            ts = time()
            noise_model = bitphase_flip_noise_model(0.01, target_circuit.num_qubits)
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
    plot_3d_surface(surface_plot_result_dict, decoded_logical_readout_qtcodes, ip=4)

    ####################################################################### Topological analysis #######################################################################
    read_from_file = False
    optimization_level = 1
    min_size = target_circuit.num_qubits
    available_qubits = range(30)
    topologies = get_coupling_maps(min_size=min_size)
    noise_model = bitphase_flip_noise_model(0.01, target_circuit.num_qubits)
    qubits_list = [q for q in range(target_circuit.num_qubits)]

    for topology_name, topology in topologies.items():
        # Graph plot: loop over root injection qubits
        if not read_from_file:
            ts = time()
            # Transpile a first time to find which qubits are active
            device_backend = CustomBackend(active_qubits=available_qubits, coupling_map=topology)
            transpiled_circuit = transpile(target_circuit, device_backend, optimization_level=optimization_level, scheduling_method='asap', seed_transpiler=42)
            active_qubits = get_active_qubits(transpiled_circuit)
            reduced_topology = filter_coupling_map(topology, active_qubits)
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
            with open(f"./results/{target_circuit.name}_{topology_name}_graphplot", 'wb') as handle:
                dill.dump(graph_plot_result_dict, handle, protocol=dill.HIGHEST_PROTOCOL)
        else:
            with open(f"./results/{target_circuit.name}_{topology_name}_graphplot", 'rb') as pickle_file:
                graph_plot_result_dict = dill.load(pickle_file)
        plot_topology_injection_point_error(graph_plot_result_dict, decoded_logical_readout_qtcodes, topology_name)

    log(f"Campaign finished at {datetime.fromtimestamp(time())}")

if __name__ == "__main__":
    main()

# %%
