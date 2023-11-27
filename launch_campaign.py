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

    # TODO: Sycamore backend
    from qiskit.providers.fake_provider import ConfigurableFakeBackend
    # sycamore_t1 = 15 # microseconds
    # sycamore_t2 = 0 # micorseconds (MISSING DATA)
    # qubit_frequency = 5 # GHz (TODO: Add freqeuncy list)
    # qubit_readout_error = 7 # % of preparing |1> and measuring |0>
    # basis_gates = device_backend.configuration().basis_gates # TODO: Add actual instruction list
    # nm = NoiseModel.from_backend(device_backend)
    cmap_sycamore_53 = [[0, 5], [5, 1], [1, 6], [6, 2], [2, 7], [7, 14], [14, 8], [8, 3], [3, 9], [9, 4], [4, 10], [10, 16], [16, 9], [9, 15], [15, 8], [5, 11], [11, 17], [17, 23], [23, 29], [29, 35], [35, 41], [41, 47], [41, 48], [48, 42], [42, 49], [49, 43], [43, 50], [50, 44], [44, 51], [51, 45], [45, 52], [52, 46], [46, 40], [40, 45], [45, 39], [39, 44], [44, 38], [38, 43], [43, 37], [37, 42], [42, 36], [36, 41], [36, 29], [29, 24], [24, 17], [17, 12], [12, 5], [12, 6], [6, 13], [13, 7], [12, 18], [18, 13], [13, 19], [19, 14], [14, 20], [20, 15], [15, 21], [21, 16], [16, 22], [22, 28], [28, 21], [21, 27], [27, 20], [20, 26], [26, 19], [19, 25], [25, 18], [18, 24], [24, 30], [30, 25], [25, 31], [31, 26], [26, 32], [32, 27], [27, 33], [33, 28], [28, 34], [34, 40], [40, 33], [33, 39], [39, 32], [32, 38], [38, 31], [31, 37], [37, 30], [30, 36]]
    # device_backend = ConfigurableFakeBackend(name="FakeSycamore", n_qubits=53, version=1, coupling_map=cmap, basis_gates=basis_gates, qubit_t1=sycamore_t1, single_qubit_gates=basis_gates, qubit_readout_error=qubit_readout_error, dt=device_backend.configuration().dt)

    from qiskit.providers.fake_provider import FakeMumbai
    device_backend = FakeMumbai()

    cmap_sycamore_25 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 9], [9, 14], [14, 19], [19, 24], [24, 23], [23, 18], [18, 13], [13, 8], [8, 7], [7, 6], [6, 5], [5, 10], [10, 15], [15, 20], [20, 21], [21, 16], [11, 16], [0, 5], [1, 6], [6, 11], [11, 12], [12, 7], [7, 2], [3, 8], [8, 9], [14, 13], [13, 12], [12, 17], [18, 17], [16, 17], [17, 22], [23, 22], [22, 21], [10, 11], [16, 15], [18, 19]]

    device_backend._configuration.coupling_map = cmap_sycamore_25
    device_backend._configuration.n_qubits = 25

    circ = QuantumCircuit(25, 25)
    circ.x(range(25))
    circ.measure(range(25), range(25))
    circ.name = "Google Experiment 25 qubits"
    circuits.append(circ)
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
    transient_error_function = reset_to_zero
    spread_depth = 6
    damping_function = exponential_damping
    apply_transpiler = False
    noiseless = False
    transient_error_duration_ns = 25000000
    n_quantised_steps = 10
    processes = 1

    # If you want to make a shot-level simulation, use this only with a single circuit at a time!
    # shot_time_per_circuit = get_shot_execution_time_ns(circuits, device_backend=device_backend, apply_transpiler=apply_transpiler)
    # n_quantised_steps = transient_error_duration_ns // int(shot_time_per_circuit[circuits[0].name])

    # Run transient injection simulation
    result_dict = run_transient_injection(circuits, 
                                          device_backend=device_backend,
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

    # TODO: Sycamore backend
    # TODO: Test surface codes:
        # - Surface code 3,3 XZZX
        # - Surface code 5,5 XZZX
        # - Surface code 3,3 XXZZ
        # - Surface code 5,5 XXZZ

if __name__ == "__main__":
    main()

#%%
# import dill
# from injector_par import *
# result_dict_name = f"results/campaign_2023-11-24 14:28:38.054834"
# with open(result_dict_name, 'rb') as pickle_file:
#     result_dict = dill.load(pickle_file)

# # filter_dict(result_dict, [3, 5])
# plot_data(result_dict, count_collapses_error)
# %%
