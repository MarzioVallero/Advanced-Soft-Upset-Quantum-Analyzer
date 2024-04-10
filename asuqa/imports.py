from copy import deepcopy
from functools import partial
from itertools import combinations
import pandas as pd
import pickle, bz2
from time import time, sleep
from datetime import datetime, timedelta
from math import exp, inf
import networkx as nx
from subprocess import Popen, PIPE, STDOUT
import tqdm
from qiskit import transpile, QuantumCircuit
from qiskit import Aer
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import FakeBrooklyn, FakeProvider
from qiskit_aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit_aer.noise.device.parameters import thermal_relaxation_values, readout_error_values
from qiskit.providers.fake_provider import ConfigurableFakeBackend
from qiskit.providers.aer.noise import reset_error, pauli_error
from qiskit.circuit.library import IGate
from qiskit.quantum_info.analysis import hellinger_distance, hellinger_fidelity
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

file_logging = False
logging_filename = "./asuqa.log"
console_logging = True
line_edge_list = [[i, i+1] for i in range(30)]
complete_edge_list = [[i, j] for i in range(30) for j in range(i) if i != j]
mesh_edge_list = [[(i*5)+j, (i*5)+j+1] for i in range(6) for j in range(4)] +  [[((i)*5)+j, ((i+1)*5)+j] for i in range(5) for j in range(5)]