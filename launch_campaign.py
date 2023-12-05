#%%
from injector_par import *

def main():
    circuits = []

    # Minimal working example circuit
    circ = QuantumCircuit(7, 7)
    circ.x(range(7))
    circ.measure(range(7), range(7))
    circ.name = "Google Experiment"
    circuits.append(circ)

    # Repetition qubit surface code
    from qtcodes import RepetitionQubit
    qubit = RepetitionQubit({"d":3},"t")
    qubit.reset_z()
    qubit.stabilize()
    qubit.x()
    qubit.stabilize()
    qubit.readout_z()
    qubit.circ.name = "Repetition Qubit"
    circuits.append(qubit.circ)

    # Logical level Grover's Algorithm
    from qiskit_algorithms import AmplificationProblem, Grover
    from qiskit.primitives import Sampler
    # the state we desire to find is '11'
    good_state = ['11']
    # specify the oracle that marks the state '11' as a good solution
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)
    # define Grover's algorithm
    problem = AmplificationProblem(oracle, is_good_state=good_state)
    grover_operator = Grover(sampler=Sampler())
    grover_circuit = grover_operator.construct_circuit(problem, power=grover_operator.optimal_num_iterations(1, 2), measurement=True)
    grover_circuit.name = "Grover Search 2 qubits"

    circuits.append(grover_circuit)

    # Surface code 3,3 XXZZ
    from qtcodes import XXZZQubit, XZZXQubit
    xxzzd3 = XXZZQubit({'d':3})
    xxzzd3.stabilize()
    xxzzd3.stabilize()
    xxzzd3.x()
    xxzzd3.stabilize()
    xxzzd3.readout_z()
    xxzzd3.circ.name = "XXZZ d3 Qubit"
    circuits.append(xxzzd3.circ)

    # Surface code 5,5 XXZZ
    xxzzd5 = XXZZQubit({'d':5})
    xxzzd5.stabilize()
    xxzzd5.stabilize()
    xxzzd5.x()
    xxzzd5.stabilize()
    xxzzd5.readout_z()
    xxzzd5.circ.name = "XXZZ d5 Qubit"
    circuits.append(xxzzd5.circ)

    xzzxd3 = XZZXQubit({'d':3})
    xzzxd3.stabilize()
    xzzxd3.stabilize()
    xzzxd3.x()
    xzzxd3.stabilize()
    xzzxd3.readout_z()
    xzzxd3.circ.name = "XXZZ d3 Qubit"
    circuits.append(xzzxd3.circ)

    xzzxd5 = XZZXQubit({'d':5})
    xzzxd5.stabilize()
    xzzxd5.stabilize()
    xzzxd5.x()
    xzzxd5.stabilize()
    xzzxd5.readout_z()
    xzzxd5.circ.name = "XXZZ d5 Qubit"
    circuits.append(xzzxd5.circ)

    # Overwrite to test only circ
    circuits = [circ]

    # Define a device backend
    device_backend = FakeJakarta()
    # device_backend = FakeSycamore25()

    # circ = QuantumCircuit(25, 25)
    # circ.x(range(25))
    # circ.measure(range(25), range(25))
    # circ.name = "Google Experiment 25 qubits"
    # circuits.append(circ)
    circuits = [circ]

    ts = time()
    log(f"Job started at {ts}.")

    # Transpile the circuits at the start to not reapeat transpilation at every injection campaing
    t_circuits = []
    for circuit in circuits:
        t_circ = transpile(circuit, device_backend, scheduling_method='asap',
                initial_layout=list(range(len(circuit.qubits))), seed_transpiler=42)
        t_circuits.append(t_circ)
    circuits = t_circuits
    log(f"Transpilation done ({time() - ts} elapsed)")

    # Transient simulation controls
    injection_points = [0]
    transient_error_function = reset_to_zero
    spread_depth = 6
    damping_function = exponential_damping
    apply_transpiler = False
    noiseless = False
    transient_error_duration_ns = 25000000
    n_quantised_steps = 100

    max_qubits = 27
    max_cores = 8
    gpu_limit = int( (64*1024**3)/((2**max_qubits)*8) )
    use_gpu = True

    processes = min(2*max_cores, gpu_limit)
    if not use_gpu and (processes > n_quantised_steps or processes > max_cores):
        processes = min(n_quantised_steps+1, max_cores)

    # If you want to make a shot-level simulation, use this only with a single circuit at a time!
    # shot_time_per_circuit = get_shot_execution_time_ns(circuits, device_backend=device_backend, apply_transpiler=apply_transpiler)
    # n_quantised_steps = transient_error_duration_ns // int(shot_time_per_circuit[circuits[0].name])

    # Run transient injection simulation
    result_dict = run_transient_injection(circuits, 
                                          device_backend=device_backend,
                                          injection_points=injection_points,
                                          transient_error_function = transient_error_function,
                                          spread_depth = spread_depth,
                                          damping_function = damping_function,
                                          apply_transpiler = apply_transpiler,
                                          noiseless = noiseless,
                                          transient_error_duration_ns = transient_error_duration_ns,
                                          n_quantised_steps = n_quantised_steps,
                                          processes=processes
                                         )
    
    log(f"Simulation done ({time() - ts} elapsed)")

    plot_data(result_dict, count_collapses_error)
    log(f"Data processed ({time() - ts} elapsed)")

if __name__ == "__main__":
    main()
