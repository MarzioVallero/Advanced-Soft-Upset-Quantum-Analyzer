#%%
from injector_par import *
from qtcodes import RepetitionQubit, RepetitionDecoder

def main():
    ts = time()
    log(f"Job started at {datetime.fromtimestamp(ts)}.")

    # Repetition qubit surface code
    d = 5
    T = 1
    repetition_q = RepetitionQubit({"d":d},"t")
    repetition_q.reset_z()
    repetition_q.stabilize()
    repetition_q.x()
    repetition_q.stabilize()
    repetition_q.readout_z()
    repetition_q.circ.name = "Repetition Qubit"

    # Backend and circuit selection
    target_circuit = repetition_q.circ
    qtcodes_circ = repetition_q
    device_backend = CustomBackend(n_qubits=target_circuit.num_qubits)

    # Readout and error correction
    decoder = RepetitionDecoder({"d":d,"T":T})
    readout_type = "Z"
    expected_logical_output = 1

    # Logical error ratio
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

    # After decoding logical error rate
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

    transpiled_circuit = transpile(target_circuit, device_backend, scheduling_method='asap',
                                    # initial_layout=list(range(len(target_circuit.qubits))),
                                    seed_transpiler=42)
    log(f"Initialisation and transpilation done in {timedelta(seconds=time() - ts)}")

    # Transient error controls
    injection_point = 2
    transient_error_function = reset_to_zero
    spread_depth = 10
    damping_function = square_damping
    transient_error_duration_ns = 25000000
    n_quantised_steps = 10

    # Simulation performance controls
    max_qubits = 25
    max_cores = 8
    max_gpu_memory = 64 #GB
    gpu_limit = int( (max_gpu_memory*1024**3)/((2**max_qubits)*8) )
    use_gpu = True
    processes = min(max_cores, gpu_limit, n_quantised_steps)
    if not use_gpu and (processes > n_quantised_steps or processes > max_cores):
        processes = min(n_quantised_steps+1, max_cores)

    # To make a shot-level simulation, use this only with a single circuit at a time!
    # shot_time_per_circuit = get_shot_execution_time_ns(circuit)
    # n_quantised_steps = transient_error_duration_ns // int(shot_time_per_circuit))

    # Surface plot: loop over physical error list
    read_from_file = True
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

    # Topological analysis
    read_from_file = False
    line_edge_list = [[i, i+1] for i in range(25)]
    complete_edge_list = [[i, j] for i in range(25) for j in range(i) if i != j]
    mesh_edge_list = [[(i*5)+j, (i*5)+j+1] for i in range(5) for j in range(4)] +  [[((i)*5)+j, ((i+1)*5)+j] for i in range(4) for j in range(5)]
    topologies = {"linear":line_edge_list, "complete":complete_edge_list, "square_mesh":mesh_edge_list}
    noise_model = bitphase_flip_noise_model(0.01, target_circuit.num_qubits)
    qubits_list = [q for q in range(target_circuit.num_qubits)]

    for topology_name, topology in topologies.items():
        # Graph plot: loop over root injection qubits
        if not read_from_file:
            ts = time()
            device_backend = CustomBackend(n_qubits=target_circuit.num_qubits, coupling_map=topology)
            transpiled_circuit = transpile(target_circuit, device_backend, scheduling_method='asap',
                                            # initial_layout=list(range(len(target_circuit.qubits))),
                                            seed_transpiler=42)
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
