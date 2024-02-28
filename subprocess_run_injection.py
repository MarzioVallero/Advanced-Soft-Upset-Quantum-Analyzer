from sys import stdin, stdout
from injector import run_injection
import pickle, bz2

if __name__ == "__main__":
    raw_input = stdin.buffer.read()
    args = pickle.loads(bz2.decompress(raw_input))

    counts = run_injection(circuit=args["circuit"],
                           injection_point=args["injection_point"],
                           transient_error_function=args["transient_error_function"],
                           root_inj_probability=args["root_inj_probability"],
                           time_step=args["time_step"],
                           spread_depth=args["spread_depth"],
                           damping_function=args["damping_function"], 
                           device_backend=args["device_backend"], 
                           noise_model=args["noise_model"],
                           shots=args["shots"],
                           execution_type=args["execution_type"])

    stdout.buffer.write(bz2.compress(pickle.dumps(counts)))

exit(0)