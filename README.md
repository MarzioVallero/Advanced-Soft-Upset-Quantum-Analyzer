

# ASUQA

<!-- ![Logo](doc/logo.svg) -->

- [What is ASUQA?](#what-is-ASUQA)
- [How do I use ASUQA?](#how-use-ASUQA)
- [How does ASUQA work?](#how-ASUQA-works)
- [How do I cite ASUQA?](#how-cite-ASUQA)

# <a name="what-is-ASUQA"></a>What is ASUQA?

ASUQA is a python-based tool for simulating the injection of transient errors in superconducting quantum computers.  
The main idea is to provide an easy to use interface for setting up injection campaign parameters and effortlessly getting significant results.  
Typically ASUQA is used as a python package (`pip install asuqa`), and imported as an external module (`import asuqa`).  

ASUQA's key features:  

1. **_GPU accelerated_ fault injection simulation**.  
Thanks to `qiskit`'s `AerSimulator` backend, plus a custom external parallelisation wrapper provided by ASUQA, performing hundreds of statevector simulations has never been easier.  
ASUQA auto-detects the presence of a GPU with a compatible `qiskit-aer` installation, and silently switches to CPU if none is found.  

2. **Flexible hands on design**.  
The module has been developed with adaptability in mind.  
Injection campaigns, noise models and fault parameters are fully customisable, according to your needs.  
All of `qiskit`'s gate operations are supported.  

3. **Compatibility with Qiskit**.  
Effortlessly test all your `qiskit` quantum programs and circuits, no need to rewrite anything.

ASUQA's main limitations are:

1. Simulation of quantum circuits is limited to 30 qubits, following `qiskit-aer`'s AerSimulator limitation.
2. Time and memory requirements for simulating very large circuits exceed the cabailities of most consumer grade laptops.

ASUQA's design philosophy:

- **Get the most out of HPC machines.**  
ASUQA's code is intended to be run on server-grade machines, so expect to get full utilisation of all the cores and GPUs in your HPC system.  
The more resources you can throw at it, the faster it will go.  

- **Highly descriptive.**  
ASUQA exposes plenty of parameters to give you fine grained control over the noise models and the fault injection parameters let you define exactly the properties of the fault event you are simulating.

- **Cross compatibility.**  
ASUQA is always updated according to the last major release of `qiskit` and `qiskit-aer`.

# <a name="how-use-ASUQA"></a>How do I use ASUQA?

Refer to the example campaign files in the "campaigns" directory.

# <a name="how-ASUQA-works"></a>How does ASUQA work?

Refer to the original paper by the authors, which is currently under review.

# <a name="how-cite-ASUQA"></a>How do I cite ASUQA?

When using ASUQA for research, [please cite](tbd):

```
@article{tbd,
  doi = {},
  url = {},
  title = {},
  author = {},
  journal = {},
  issn = {},
  publisher = {}
}
```
