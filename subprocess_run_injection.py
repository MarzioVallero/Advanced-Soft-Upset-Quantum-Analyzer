from sys import stdin, stdout
from injector import run_injection
import pickle

if __name__ == "__main__":
    raw_input = stdin.buffer.read()
    original_args = pickle.loads(raw_input)

    result = run_injection(original_args["circuit"],
                                    injection_point=original_args["injection_point"],
                                    transient_error_function=original_args["transient_error_function"],
                                    root_inj_probability=original_args["root_inj_probability"],
                                    time_step=original_args["time_step"],
                                    spread_depth=original_args["spread_depth"],
                                    damping_function=original_args["damping_function"], 
                                    device_backend=original_args["device_backend"], 
                                    noise_model=original_args["noise_model"],
                                    shots=original_args["shots"],
                                    execution_type=original_args["execution_type"])

    stdout.buffer.write(pickle.dumps(result[0]["counts"]))

exit(0)