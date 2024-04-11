from utils import *

mapply.init(
    n_workers=50,
    chunk_size=2,
    max_chunks_per_worker=1,
    progressbar=True,
)

def plot_transient(result_df, compare_function):
    """Plot the results of an injection campaign according to the supplied compare_function(golden_counts, inj_counts) over time. The results are saved under the ./plots directory."""
    sns.set_theme(font_scale=1.7)
    sns.set_style("whitegrid", )
    golden_counts = result_df.loc[result_df['execution_type'] == "golden"]
    result_df["logical_error"] = result_df["counts"].mapply(lambda row: compare_function(golden_counts, row))
    result_df.sort_values(by=['time_step'])
    df_golden = result_df.loc[result_df['execution_type'] == "golden"]
    df_injection = result_df.loc[result_df['execution_type'] == "injection"]

    sns.set_theme(style="whitegrid", palette="deep")    
    plt.figure(figsize=(20,10))

    # Golden
    golden_counts = df_golden["counts"].iloc[0]
    golden_bitstring = max(golden_counts, key=golden_counts.get)
    golden_y = []
    time_steps = df_injection["time_step"].drop_duplicates()
    golden_logical_error = compare_function(golden_counts, golden_counts)
    for time_step in time_steps:
        golden_y.append(golden_logical_error)
    plt.plot(time_steps, golden_y, label=f"Golden", color="gold")

    df_injection_point = [x for _, x in df_injection.groupby(["injection_point"])]
    for df in df_injection_point:
        injected_qubit = df["injection_point"].iloc[0][0]
        plt.plot(df["time_step"], df["logical_error"], label=f"Transient on qubit {injected_qubit}") 
    plt.xlabel(f'discretised time (shots)')
    plt.ylabel(f'{compare_function.__name__}')
    plt.title(f'Transient logical error evolution over time')
    plt.legend()
    
    circuit_name = re.sub("[^a-zA-Z]", "", df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/transient_error_over_time_{compare_function.__name__}_{circuit_name} on {df["device_backend_name"].iloc[0]}'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_spatial_spread_analysis(result_df, compare_function, subgroup_sizes):
    """Plot the results of a <circuit_name>_histogram_affected_qubits results file. The results are saved under the ./plots directory."""

    def get_label(row):
        return f"{row['device_backend_name']} {'non-isolated' if row['spread_depth'] != 0 else 'isolated'}"

    def get_len(row):
        return len(row["injection_point"])

    # code distance analysis
    df_device_backend_name = [x for _, x in result_df.groupby(["device_backend_name"])]
    merged_df = pd.DataFrame()
    for df_backend in df_device_backend_name:
        golden_counts = df_backend.loc[df_backend['execution_type'] == "golden"].head(1)
        df_backend["logical_error"] = df_backend["counts"].mapply(lambda row: compare_function(golden_counts, row))
        df_backend.sort_values(by=['time_step'])
        golden_counts = df_backend.loc[df_backend['execution_type'] == "golden"].head(1) #["logical_error"]
        df_backend = df_backend.loc[df_backend['execution_type'] == "injection"] #["logical_error"]
        df_backend["label"] = df_backend.mapply(get_label, axis=1)
        df_backend["total_injected_qubits"] = df_backend.mapply(get_len, axis=1)
        df_refline_y = df_backend.loc[df_backend['spread_depth'] != 0]
        df_refline_y = df_refline_y.loc[df_refline_y['total_injected_qubits'] == 1]
        refline_y = df_refline_y["logical_error"].min()
        df_backend = df_backend.loc[df_backend['spread_depth'] == 0]
        merged_df = pd.concat([merged_df, df_backend], ignore_index=True)

    merged_df = merged_df[merged_df['total_injected_qubits'].isin(subgroup_sizes)]

    df = pd.pivot(merged_df, index="label", columns="total_injected_qubits", values="logical_error")
    df = df.loc[:, ~(df.eq(1) | df.isna()).all()]
    df = df.unstack().reset_index(name="logical_error")
    sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.7)
    g = sns.catplot(x="total_injected_qubits", y="logical_error", hue="total_injected_qubits", data=df, kind="bar", legend=False, height=5, aspect=10/5)
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(percentage_tick_formatter_no_decimal)
    g.refline(y=refline_y, color='red')
    g.set_xlabels('number of corrupted qubits')
    g.set_ylabels('logical error')
    g.set(ylim=(0, 1))

    circuit_name = re.sub("[^a-zA-Z]", "", merged_df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/histogram_spread_depth_error_{compare_function.__name__}_{circuit_name} on {merged_df["device_backend_name"].iloc[0]}.pdf'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_noise_radiation_analysis(result_df, compare_function, ip=1):
    """Plot the results of a <circuit_name>_surfaceplot results file. The results are saved under the ./plots directory."""
    sns.set_theme(font_scale=1.7)
    sns.set_style("whitegrid", )
    df_data_list = []
    df_noise_model = [x for _, x in result_df.groupby(["noise_model"])]
    for df in df_noise_model:
        golden_counts = df.loc[df['execution_type'] == "golden"]
        df["logical_error"] = df["counts"].mapply(lambda row: compare_function(golden_counts, row))
        df.sort_values(by=['time_step'])
        inj_logical_error = df.loc[df['execution_type'] == "injection"][["noise_model", "root_inj_probability", "logical_error"]]
        inj_logical_error["physical_error"] = inj_logical_error.mapply(lambda row: float(re.findall("-?[\d.]+(?:e-?\d+)?", row["noise_model"])[0]), axis=1)
        df_data_list.append(inj_logical_error)

    df = pd.concat(df_data_list, ignore_index=True)
    sns.set_style("whitegrid", {'axes.grid' : False})
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.invert_xaxis()

    # ax.yaxis.set_major_formatter(mticker.FuncFormatter(percentage_tick_formatter_no_decimal))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(percentage_tick_formatter_no_decimal))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    x1 = 10**(np.linspace(np.log10(df['physical_error'].min()), np.log10(df['physical_error'].max()), ip*len(df['physical_error'].drop_duplicates()) ))
    # y1 = np.linspace(df['root_inj_probability'].min(), df['root_inj_probability'].max(), ip*len(df['root_inj_probability'].drop_duplicates()))
    y1 = 10**(np.linspace(np.log10(df['root_inj_probability'].min()), np.log10(df['root_inj_probability'].max()), ip*len(df['root_inj_probability'].drop_duplicates()) ))
    X, Y = np.meshgrid(x1, y1)
    Z = griddata(((df['physical_error']), df['root_inj_probability']), df['logical_error'], (X, Y), method="linear")
    g = ax.plot_surface(np.log10(X), np.log10(Y), Z, rstride=1, cstride=1, cmap="Spectral_r", linewidth=0.0, antialiased=True)

    # g = ax.plot_trisurf(np.log10(df.physical_error), df.root_inj_probability, df.logical_error, cmap="Spectral_r", linewidth=0.2, antialiased=True)
    ax.set_xlabel('\nPhysical error rate', labelpad=10)
    ax.set_ylabel('\nInjection probability', labelpad=10)
    ax.set_zlabel('\nLogical error', labelpad=15)
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)

    circuit_name = re.sub("[^a-zA-Z]", "", result_df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/3d_surfaceplot_{compare_function.__name__}_{circuit_name} on {result_df["device_backend_name"].iloc[0]}.pdf'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename)
    plt.close()
    plt.clf()

def plot_architecture_analysis(result_df, compare_function):
    """Plot the results of a <circuit_name>_topologies_analysis results file. The results are saved under the ./plots directory."""
    sns.set_theme(font_scale=1.7)
    sns.set_style("whitegrid", {'axes.grid' : False})
    vmin, vmax = (1.0, 0.0)
    graphs_with_data = []
    df_device_backend_name = [x for _, x in result_df.groupby(["device_backend_name"])]
    for df_all_nodes in df_device_backend_name:
        golden_counts = df_all_nodes.loc[df_all_nodes['execution_type'] == "golden"]
        df_all_nodes["logical_error"] = df_all_nodes["counts"].mapply(lambda row: compare_function(golden_counts, row))
        df_all_nodes.sort_values(by=['time_step'])
        df_inj_point = df_all_nodes.loc[df_all_nodes['execution_type'] == "injection"][["injection_point", "logical_error"]].groupby(["injection_point"], as_index=False).median()
        max_average_node_logical_error = df_inj_point["logical_error"].max()
        if max_average_node_logical_error > vmax:
            vmax = max_average_node_logical_error
        min_average_node_logical_error = df_inj_point["logical_error"].min()
        if min_average_node_logical_error < vmin:
            vmin = min_average_node_logical_error
        coupling_map = df_all_nodes["coupling_map"].iloc[0]
        G = nx.Graph(coupling_map)
        for node_index in list(G):
            G.nodes[node_index]["logical_error"] = 0
        for index, row in df_inj_point.iterrows():
            node_logical_error = row["logical_error"] if not row.empty else 0
            node_index = row["injection_point"][0]
            G.nodes[node_index]["logical_error"] = node_logical_error
        p2v_map = golden_counts["p2v_map"].iloc[0]
        graphs_with_data.append((df_all_nodes["device_backend_name"].iloc[0], G, p2v_map))

    nc = int(np.ceil(np.sqrt(len(graphs_with_data))))
    nr = int(np.ceil(len(graphs_with_data)/nc))
    fig = plt.figure(figsize=(12*nc, 10*nr))
    plt.clf()
    plt.axis('off')
    fig, ax = plt.subplots(nr, nc, num=1)

    # drawing nodes and edges separately so we can capture collection for colobar
    for i, (name, G, p2v_map) in enumerate(graphs_with_data):
        ix = np.unravel_index(i, ax.shape)
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato", args="-Gepsilon=.00000000001 -Glen=200 -Gmaxiter=100 -Goverlap_scaling=20 -Goverlap='false' -Gsep=+50") if name!="linear" else nx.spiral_layout(G, equidistant=True)
        ec = nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax[ix])
        colors = nx.get_node_attributes(G, "logical_error")

        print(colors)
        print(f"{name} avg(logical_error)={sum(colors.values()) / len(colors) }")
        
        # nc = nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=250, cmap=plt.cm.Spectral_r, ax=ax[ix], vmin=0.0, vmax=vmax)
        layout = {k:v.replace('t_', '').replace('tq_', '').replace('ancilla', 'a').replace('data', 'd') for k,v in p2v_map.items()}
        ancilla_nodelist = [k for k,v in layout.items() if "a" in v]
        measure_nodelist = [k for k,v in layout.items() if "m" in v]
        data_nodelist = [k for k,v in layout.items() if "d" in v]
        nc = nx.draw_networkx_nodes(G, pos, node_color=[colors[n] for n in ancilla_nodelist], node_size=1000, cmap=plt.cm.Spectral_r, nodelist=ancilla_nodelist, node_shape='p', edgecolors="black", ax=ax[ix], vmin=vmin, vmax=vmax)
        nc = nx.draw_networkx_nodes(G, pos, node_color=[colors[n] for n in measure_nodelist], node_size=1000, cmap=plt.cm.Spectral_r, nodelist=measure_nodelist, node_shape='s', edgecolors="black", ax=ax[ix], vmin=vmin, vmax=vmax)
        nc = nx.draw_networkx_nodes(G, pos, node_color=[colors[n] for n in data_nodelist], node_size=1000, cmap=plt.cm.Spectral_r, nodelist=data_nodelist, node_shape='o', edgecolors="black", ax=ax[ix], vmin=vmin, vmax=vmax)

        nodes = list(G)
        light_text = [n for n, name in layout.items() if (G.nodes[n]['logical_error'] < 0.25*(vmax-vmin) or G.nodes[n]['logical_error'] > 0.75*(vmax-vmin))]
        nx.draw_networkx_labels(G, pos, labels={n:name for n, name in layout.items() if n not in light_text}, font_color="black", ax=ax[ix])
        nx.draw_networkx_labels(G, pos, labels={n:name for n, name in layout.items() if n in light_text}, font_color="white", ax=ax[ix])
        ax[ix].set_title(str(name).split("_")[-1], fontsize=30)

    for i in range(i+1, len(ax.flatten())):
        ix = np.unravel_index(i, ax.shape)
        ax[ix].set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), format=percentage_tick_formatter_no_decimal)
    cbar.ax.tick_params(labelsize=40) 
    
    circuit_name = re.sub("[^a-zA-Z]", "", result_df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/topology_injection_point_{compare_function.__name__}_{circuit_name} on {result_df["device_backend_name"].iloc[0]}.pdf'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_code_distance_analysis(result_df, compare_function_generator):
    """Plot the results of a <circuit_name> minimum_inj_qubits results file. The results are saved under the ./plots directory."""

    def get_len(row):
        return len(row["injection_point"])

    # code distance analysis
    df_device_circuit_name = [x for _, x in result_df.groupby(["circuit_name"])]
    merged_df_list = []
    for df_circuit_name in df_device_circuit_name:
        golden_counts = df_circuit_name.loc[df_circuit_name['execution_type'] == "golden"].head(1)
        d = make_tuple(re.sub("[^0-9(),]", "", golden_counts["circuit_name"].values[0]))
        compare_function = compare_function_generator(d=d)
        # Add compare fn
        df_circuit_name["logical_error"] = df_circuit_name["counts"].mapply(lambda row: compare_function(golden_counts, row))
        golden_counts = df_circuit_name.loc[df_circuit_name['execution_type'] == "golden"].head(1)
        df_circuit_name = df_circuit_name.loc[df_circuit_name['execution_type'] == "injection"]
        df_circuit_name["total_injected_qubits"] = df_circuit_name.mapply(get_len, axis=1)
        df_circuit_name.reset_index(drop=True, inplace=True)
        df_circuit_name = df_circuit_name[df_circuit_name.columns.intersection(["circuit_name", "device_backend_name", "total_injected_qubits", "logical_error"])]
        df_circuit_name = df_circuit_name.groupby(["circuit_name", "device_backend_name", "total_injected_qubits"], as_index=False).agg({'logical_error': ['mean', 'median', 'std']}).reset_index()
        df_circuit_name.reset_index(drop=True, inplace=True)
        df_circuit_name.columns = ["_".join(col_name).rstrip('_') for col_name in df_circuit_name.columns]
        
        df_circuit_name.sort_values(by=['total_injected_qubits'])
        # df_circuit_name = df_circuit_name.loc[df_circuit_name["logical_error_median"] > threshold_min]
        # df_circuit_name["d"] = d[0]
        # df_circuit_name["d_str"] = str(d)
        # df_circuit_name["label"] = f"${{{threshold_min*100:.0f}}} \%$ logic error"
        # merged_df_list.append(df_circuit_name.head(1))
        # df_circuit_name = df_circuit_name.loc[df_circuit_name["logical_error_median"] > threshold_catasptrophic]
        # df_circuit_name["label"] = f'${{{threshold_catasptrophic*100:.0f}}} \%$ logic error'
        # merged_df_list.append(df_circuit_name.head(1))
        
        value = int(int(d[0])*int(d[1])*2)
        print(value)
        df_circuit_name["circuit size"] = value
        df_circuit_name["d_str"] = str(d)
        df_circuit_name = df_circuit_name.loc[df_circuit_name["total_injected_qubits"] == 1]
        merged_df_list.append(df_circuit_name.head(1))

    merged_df = pd.concat(merged_df_list, ignore_index=True)
    merged_df.sort_values('circuit size', inplace=True)
    sns.set_theme(style="whitegrid", palette="husl", rc={'figure.figsize':(12,6)}, font_scale=1.7)
    g = sns.barplot(merged_df,  y="d_str", x="logical_error_median", hue="circuit size", orient="h", legend="full")
    g.set(xlabel='median logic error', ylabel='surface code distance')
    g.xaxis.set_major_formatter(mticker.FuncFormatter(percentage_tick_formatter_no_decimal))
    g.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    circuit_name = re.sub("[^a-zA-Z]", "", merged_df["circuit_name"].iloc[0])
    filename = f'plots/{circuit_name}/min_injected_qubits_{compare_function.__name__}_{circuit_name} on {merged_df["device_backend_name"].iloc[0]}.pdf'
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    plt.clf()

def log_tick_formatter(val, pos=None):
    return f'$\quad10^{{{val:.0f}}}$'

def percentage_tick_formatter_no_decimal(val, pos=None):
    return f'$\quad{{{val*100:.0f}}} \%$'

def percentage_tick_formatter(val, pos=None):
    return f'$\quad{{{val*100:.2f}}} \%$'
