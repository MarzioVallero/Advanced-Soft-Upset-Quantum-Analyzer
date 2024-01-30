from sys import argv
from injector_par import *
from dill import load
from os import makedirs

index = int(argv[1])
data_name = argv[2]

dill.settings['recurse'] = True

with open(data_name, 'rb') as handle:
    original_args = load(handle)

probability_per_batch = original_args["probability_per_batch"]
shots_per_time_batch = original_args["shots_per_time_batch"]
circuit_name = original_args["circuit"].name

for iteration, p in enumerate(probability_per_batch):
    if iteration == index:
        result = run_injection_campaing(original_args["circuit"],
                                        injection_point=original_args["injection_point"],
                                        transient_error_function=original_args["transient_error_function"],
                                        root_inj_probability=p,
                                        time_step=int(shots_per_time_batch)*iteration,
                                        spread_depth=original_args["spread_depth"],
                                        damping_function=original_args["damping_function"], 
                                        device_backend=original_args["device_backend"], 
                                        noise_model=original_args["noise_model"],
                                        shots=int(shots_per_time_batch),
                                        execution_type="injection")
        res_filename = f"results/{circuit_name}_{original_args['n_quantised_steps']}/{circuit_name}_{str(iteration).zfill(10)}.pkl"
        if not isdir(dirname(res_filename)):
            makedirs(dirname(res_filename), exist_ok=True)
        with open(res_filename, 'wb') as handle:
            dill.dump(result, handle, protocol=dill.HIGHEST_PROTOCOL)

exit(0)