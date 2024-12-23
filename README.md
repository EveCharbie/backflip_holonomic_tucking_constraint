# Introduction
This repository contains data and code from optimal control problems (OCP) of a planar digital twin performing a backward tuck somersault. 
The simulations explore the impact of limb-on-limb contact during the "tuck" phase, 
specifically how allowing the digital twin to pull its shanks with its hands (using holonomic constraints) affects 
movement compared to not having any contact or just having contact without force (kinematic constraints).

![kinograms](docs/kinograms.png)

# Cite this work
```bibtex
@article{Farr2025,
title = {Including limb-on-limb holonomic constraints in predictive simulation allows replicating athlete’s backflip technique},
journal = {submitted to Multibody System Dynamics},
volume = {},
pages = {},
year = {2025},
issn = {},
doi = {},
url = {},
author = {A. Farr, E. Charbonneau, M. Begon and P. Puchaud},
keywords = {Holonomic constraint, Constraint dynamics, Predictive simulation, Optimal control, Biomechanics, Gymnastics, Closed-loop}
```

# Status
| Type | Status |
|---|---|
| Zenodo  | [![DOI](https://zenodo.org/badge/589623975.svg)](https://doi.org/10.5281/zenodo.14546204) |

# How to install dependencies
In order to run the code, you need to install the following packages from [pyomeca]( https://github.com/pyomeca):
```bash
conda install -c conda-forge biorbd=1.11.1 python-graphviz matplotlib numpy scipy pyorerun bioviz=2.3.2 plotly
```

# Installing Bioptim from source
1. Clone the repository:
   ```bash
   git clone https://github.com/Ipuch/bioptim.git
   cd bioptim
   git checkout ReleasePaperBackTuck
   ``` 
2. Install the library:
   ```bash
   python setup.py install
   ```
# Setting Linear Solver for Ipopt
To enhance optimization, configure the linear solver (e.g., `ma57`) in your script:
```python
from bioptim import Solver
solver = Solver.IPOPT()
solver.set_linear_solver("ma57")
```
Otherwise, use mumps, but you won't replicate the results.

# Exemples
In the folder `examples\`, you will find the following examples (NTC, KTC, HTC):
- `somersault_htc.py`: This script generates the optimal control problem with the holonomic constraint.
- `somersault_htc_taudot.py`: This script generates the optimal control problem with the holonomic constraint and torque derivative driven
- `somersault_ktc_taudot.py`: This script generates the optimal control problem with the kinematic constraint and torque derivative driven
- `somersault.py`: This script generates the optimal control problem without the holonomic constraint.
- `somersault_taudot.py`: This script generates the optimal control problem without the holonomic constraint and torque derivative driven 

