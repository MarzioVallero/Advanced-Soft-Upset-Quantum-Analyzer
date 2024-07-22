# %% Plot topologies from IBM devices
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asuqa import *
from utils import *

# Filter out all "small" backends (less than size qubits)
size = 18
backends = {backend.name():backend for backend in FakeProvider().backends() if backend.configuration().n_qubits > size}
qubit_range = set(range(30))
graphs = {}
for name, backend in backends.items():
    isomorphic = False
    coupling_map = backend.configuration().coupling_map
    for edge in deepcopy(coupling_map):
        if (edge[0] not in qubit_range) or (edge[1] not in qubit_range):
            coupling_map.remove(edge)
    G = nx.Graph(coupling_map)

    for graph in graphs.values():
        if nx.is_isomorphic(G, graph[1]):
            isomorphic = True
            break
    if isomorphic or not nx.is_connected(G):
        continue

    graphs[name] = (coupling_map, G)

nr = int(np.ceil(np.sqrt(len(graphs.values()))))
fig = plt.figure(figsize=(12*nr, 12*nr)); plt.clf()
fig, ax = plt.subplots(nr, nr, num=1)

for i, (name, (cm, graph)) in enumerate(graphs.items()):
    ix = np.unravel_index(i, ax.shape)
    pos = nx.spring_layout(graph, weight=0.0001, iterations=1000, threshold=0.00001)
    nx.draw_networkx(graph, pos=pos, with_labels=True, ax=ax[ix])
    ax[ix].set_title(name, fontsize=30)
filename = f"plots/extra/architectures.pdf"
Path(dirname(filename)).mkdir(parents=True, exist_ok=True)
plt.savefig(filename)
plt.close()

# %% Get total number of shots in the ./results folder
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asuqa import *
from os import listdir
from os.path import isfile, join
from utils import *

results_path = "results"
file_paths = [join(results_path, f) for f in listdir(results_path) if isfile(join(results_path, f))]

tot_shots = 0
tot_simulations = 0
for file_path in file_paths:
    with bz2.BZ2File(file_path, 'rb') as handle:
        result_df = pickle.load(handle)
    df_tot_shots = result_df['shots'].sum()
    tot_shots += df_tot_shots
    tot_simulations += len(result_df)

print(f"Total number of shots in ./results: {tot_shots}.\nTotal number of simulations in  ./results: {tot_simulations}")

# %% Plot repetition circuit and DAG
from utils import *
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers

q2 = xxzz_qubit(d=(5,1))
circ = q2.circ
filename='./plots/extra/repetition_5.pdf'
Path(dirname(filename)).mkdir(parents=True, exist_ok=True)
circ.draw(output="mpl", fold=170, filename=filename)

circ = RemoveBarriers()(circ)
dag = circuit_to_dag(circ)
dag.draw(filename='./plots/extra/repetition_5_DAG.pdf')

# %% Plot XXZZ circuit and DAG
from utils import *
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers

q2 = xxzz_qubit(d=(3,3))
circ = q2.circ
filename='./plots/extra/xxzz_3.pdf'
Path(dirname(filename)).mkdir(parents=True, exist_ok=True)
circ.draw(output="mpl", fold=170, filename=filename)

circ = RemoveBarriers()(circ)
dag = circuit_to_dag(circ)
dag.draw(filename='./plots/extra/xxzz_3_DAG.pdf')

# %% Plot spatial distribution of error
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asuqa import *
from visualisation import *
from utils import *

sns.set_theme(font_scale=1.7)
sns.set_style("whitegrid", {'axes.grid' : False})

# Generate data for a 3D surface plot
res = 5
x = np.linspace(-10, 10, 21*res)
y = np.linspace(-10, 10, 21*res)
X, Y = np.meshgrid(x, y)
Z = np.log10(error_probability_decay(0, 10)*square_damping(np.sqrt( X**2 + Y**2)))

# Create a 3D surface plot with Seaborn
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="magma", linewidth=0.0, antialiased=True)
ax.set_xlabel('distance [a.u.]', labelpad=20)
ax.set_ylabel('distance [a.u.]', labelpad=20)
ax.set_zlabel('injection probability', labelpad=-40)
plt.tight_layout()
filename = f"plots/extra/spatial_square_damping.pdf"
Path(dirname(filename)).mkdir(parents=True, exist_ok=True)
plt.savefig(filename)
plt.close()

# %% Plot the temporal decay function
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asuqa import *
from utils import *

sns.set_theme(font_scale=1.7)
sns.set_style("whitegrid", )

res = 5
x1 = np.linspace(0, 1, 10*res)
x2 = np.linspace(0, 1, 11)
y1 = [ error_probability_decay(x, 1) for x in x1]
y2 = [ error_probability_decay(x, 1) for x in x2]

fig = plt.figure(figsize=(10, 4))
ax=fig.add_subplot(111)
sns.lineplot(x=x1, y=y1, ax=ax, label="$T(t)$")
sns.lineplot(x=x2, y=y2, drawstyle='steps-post', label="$\hat{T}(t)$")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel('time [a.u.]', labelpad=0)
ax.set_ylabel('injection probability', labelpad=0)
plt.legend(title='', loc='upper right')

plt.tight_layout()
filename = f"plots/extra/temporal_decay.pdf"
Path(dirname(filename)).mkdir(parents=True, exist_ok=True)
plt.savefig(filename)
plt.close()

# %% Get time required to execute a surface code shot
from utils import *
from qiskit.providers.fake_provider import FakeBrooklyn

print("Real-time duration of different circuits.")
data = {"rq5":repetition_qubit(d=5), "rq25":repetition_qubit(d=25), "xxzz3":xxzz_qubit(d=3), "xxzz5":xxzz_qubit(d=5)}
backend = FakeBrooklyn()

for name, circ in data.items():
    t_circ = transpile(circ.circ, backend, scheduling_method='asap', seed_transpiler=42)
    shot_time = get_shot_execution_time_ns(t_circ)
    print(f"{name}: {shot_time/10e6} ms")
# %%
