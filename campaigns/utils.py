from qtcodes import RepetitionQubit, RepetitionDecoder, XXZZQubit, RotatedDecoder
from functools import partial
from os.path import isdir, dirname
from os import mkdir
import pandas as pd
import networkx as nx
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from mycolorpy import colorlist as mcp
from scipy.interpolate import griddata
from ast import literal_eval as make_tuple
import mapply

def repetition_qubit(d=3, T=1):
    T = 1
    repetition_q = RepetitionQubit({"d":d},"t")
    repetition_q.reset_z()
    repetition_q.stabilize()
    repetition_q.x()
    repetition_q.stabilize()
    repetition_q.readout_z()
    repetition_q.circ.name = f"Repetition Qubit {d}"

    return repetition_q

def xxzz_qubit(d=3, T=1):
    xxzzd3 = XXZZQubit({"d":d})
    xxzzd3.reset_z()
    xxzzd3.stabilize()
    xxzzd3.x()
    xxzzd3.stabilize()
    xxzzd3.readout_z()
    xxzzd3.circ.name = f"XXZZ Qubit {d}"

    return xxzzd3

def qtcodes_decoded_logical_readout_error(decoder, readout_type, golden_logical_bistring, golden_counts, inj_counts):
    wrong_logical_bitstring_count = 0
    total_measurements = 0
    for bitstring, count in inj_counts.items():
        logical_bitstring = decoder.correct_readout(bitstring, readout_type)
        if logical_bitstring != golden_logical_bistring:
            wrong_logical_bitstring_count += count
        total_measurements += count
        
    return (wrong_logical_bitstring_count / total_measurements)

def get_decoded_logical_error_repetition(d=3, T=1):
    decoder = RepetitionDecoder({"d":d, "T":T})
    readout_type = "Z"
    expected_logical_output = 1

    logical_readout_qtcodes = partial(qtcodes_decoded_logical_readout_error, decoder, readout_type, expected_logical_output)
    logical_readout_qtcodes.__name__ = f"decoder_{readout_type}_{d}"

    return logical_readout_qtcodes

def get_decoded_logical_error_xxzz(d=3, T=1):
    decoder = RotatedDecoder({"d":d, "T":T})
    readout_type = "Z"
    expected_logical_output = 1

    logical_readout_qtcodes = partial(qtcodes_decoded_logical_readout_error, decoder, readout_type, expected_logical_output)
    logical_readout_qtcodes.__name__ = f"decoder_{readout_type}_{d}"

    return logical_readout_qtcodes