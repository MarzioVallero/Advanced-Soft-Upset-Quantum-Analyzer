#%%
from injector_par import *
from qtcodes import XXZZQubit, XZZXQubit, RepetitionQubit

def main():
    circ = QuantumCircuit(25, 25)
    circ.x(range(25))
    circ.measure(range(25), range(25))
    circ.name = "Google Experiment"

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

    # Surface code 3,3 XXZZ
    xxzzd3 = XXZZQubit({'d':3})
    xxzzd3.stabilize()
    xxzzd3.stabilize()
    xxzzd3.x()
    xxzzd3.stabilize()
    xxzzd3.readout_z()
    xxzzd3.circ.name = "XXZZ d3 Qubit"

    xzzxd3 = XZZXQubit({'d':3})
    xzzxd3.stabilize()
    xzzxd3.stabilize()
    xzzxd3.x()
    xzzxd3.stabilize()
    xzzxd3.readout_z()
    xzzxd3.circ.name = "XZZX d3 Qubit"

    device_backend = CustomBackend()
    target_circuit = repetition_q.circ
    qtcodes_circ = repetition_q

    def bitphase_flip_noise_model(p_error, n_qubits):
        bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
        phase_flip = pauli_error([('Z', p_error), ('I', 1 - p_error)])
        bitphase_flip = bit_flip.compose(phase_flip)
        noise_model = NoiseModel()
        for q_index in range(n_qubits):
            noise_model.add_quantum_error(bitphase_flip, instructions=list(NoiseModel()._1qubit_instructions), qubits=[q_index])

        return noise_model

    ts = time()
    log(f"Job started at {datetime.fromtimestamp(ts)}.")

    # Transpile the circuits at the start to not reapeat transpilation at every injection campaing
    # t_circ = transpile(target_circuit, device_backend, scheduling_method='asap',
    #         initial_layout=list(range(len(target_circuit.qubits))), seed_transpiler=42)
    # log(f"Transpilation done ({time() - ts} elapsed)")

    # Transient simulation controls
    noise_model = bitphase_flip_noise_model(0.0, device_backend.configuration().n_qubits)
    injection_point = 2
    transient_error_function = reset_to_zero
    spread_depth = 10
    damping_function = square_damping
    transient_error_duration_ns = 25000000
    n_quantised_steps = 10

    max_qubits = 25
    max_cores = 8
    max_gpu_memory = 64 #GB
    gpu_limit = int( (max_gpu_memory*1024**3)/((2**max_qubits)*8) )
    use_gpu = True

    processes = min(max_cores, gpu_limit, n_quantised_steps)
    if not use_gpu and (processes > n_quantised_steps or processes > max_cores):
        processes = min(n_quantised_steps+1, max_cores)

    # If you want to make a shot-level simulation, use this only with a single circuit at a time!
    # shot_time_per_circuit = get_shot_execution_time_ns(circuits, device_backend=device_backend, apply_transpiler=apply_transpiler)
    # n_quantised_steps = transient_error_duration_ns // int(shot_time_per_circuit[circuits[0].name])

    # Run transient injection simulation
    debug = False
    if not debug:
        result_dict = run_transient_injection(target_circuit, 
                                          device_backend=device_backend,
                                          noise_model=noise_model,
                                          injection_point=injection_point,
                                          transient_error_function = transient_error_function,
                                          spread_depth = spread_depth,
                                          damping_function = damping_function,
                                          transient_error_duration_ns = transient_error_duration_ns,
                                          n_quantised_steps = n_quantised_steps,
                                          processes=processes
                                         )
    else:
        result_dict_name = f"results/campaign_X"
        with open(result_dict_name, 'rb') as pickle_file:
            result_dict = dill.load(pickle_file)
    
    log(f"Simulation done ({time() - ts} elapsed)")

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
    
    logical_readout_qtcodes = partial(qtcodes_logical_readout_error, qtcodes_circ, 1)
    logical_readout_qtcodes.__name__ = "logical_readout_error"

    # Decoder
    from qtcodes import RepetitionDecoder, RotatedDecoder
    decoder = RepetitionDecoder({"d":d,"T":T})
    readout_type = "Z"

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

    decoded_logical_readout_qtcodes = partial(qtcodes_decoded_logical_readout_error, decoder, readout_type, 1)
    decoded_logical_readout_qtcodes.__name__ = "decoder_logical_readout_error"

    # plot_transient(result_dict, logical_readout_qtcodes)
    # plot_transient(result_dict, decoded_logical_readout_qtcodes)
    log(f"Data processed ({time() - ts} elapsed)")

if __name__ == "__main__":
    main()

# %%
