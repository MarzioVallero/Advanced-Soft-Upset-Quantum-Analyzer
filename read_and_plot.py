import dill
from injector_par import *
result_dict_name = f"results/campaign_2023-11-24 14:28:38.054834"
with open(result_dict_name, 'rb') as pickle_file:
    result_dict = dill.load(pickle_file)

# filter_dict(result_dict, [3, 5])
plot_data(result_dict, count_collapses_error)