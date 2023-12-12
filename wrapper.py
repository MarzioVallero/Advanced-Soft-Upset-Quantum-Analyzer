from sys import argv
from injector_par import *
from dill import load
from os import makedirs

index = int(argv[1])
data_name = argv[2]
splits = cpu_count()

dill.settings['recurse'] = True

with open(data_name, 'rb') as handle:
    data = load(handle)

original_args = data["args"]
probability_per_batch_per_circuits = data["probability_per_batch_per_circuits"]
shots_per_time_batch_per_circuits = data["shots_per_time_batch_per_circuits"]
circuit_name = data["circuit_name"]

for iteration, p in enumerate(probability_per_batch_per_circuits[circuit_name]):
    if iteration == index:
        inj_result = run_injection_campaing([original_args["circuit"]],
                                        injection_points=original_args["injection_points"],
                                        transient_error_function=original_args["transient_error_function"],
                                        root_inj_probability=p,
                                        time_step=int(shots_per_time_batch_per_circuits[circuit_name])*iteration,
                                        spread_depth=original_args["spread_depth"],
                                        damping_function=original_args["damping_function"], 
                                        device_backend=original_args["device_backend"], 
                                        apply_transpiler=original_args["apply_transpiler"],
                                        noiseless=original_args["noiseless"],
                                        shots=int(shots_per_time_batch_per_circuits[circuit_name]),
                                        execution_type="injection")
        res_filename = f"results/{circuit_name}_{original_args['n_quantised_steps']}/{circuit_name}_{str(iteration).zfill(10)}.pkl"
        if not isdir(dirname(res_filename)):
            makedirs(dirname(res_filename), exist_ok=True)
        with open(res_filename, 'wb') as handle:
            dill.dump((inj_result[0], inj_result[1]['jobs'][circuit_name]), handle, protocol=dill.HIGHEST_PROTOCOL)

exit(0)