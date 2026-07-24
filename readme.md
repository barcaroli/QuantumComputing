Optimal stratification aggregates a large number of atomic strata into a small number of final strata so as to minimise the total sample size required to meet 
target precision constraints, an objective that is combinatorial and, once reformulated as a within-cluster dispersion surrogate, expressible as a quadratic unconstrained 
binary optimisation (QUBO) problem. This repository allows a controlled comparison of four solvers for that surrogate on an identical twenty-stratum frame drawn from the 
swissmunicipalities dataset: a quantum annealer (D-Wave), a gate-based quantum processor (IBM Quantum, running the Quantum Approximate Optimisation Algorithm), 
a photonic entropy-quantum-computing device (QCi Dirac-3), and a classical GPU-based Ising machine used as a control (Fixstars Amplify AE). 

Against a classical genetic-algorithm (implemented in the R package SamplingStrata) benchmark of 129 sample units, the best-known solution of the QUBO surrogate, 
established by the free classical solver, reaches 160, a measured, 
roughly twenty-four percent cost attributable to the surrogate objective itself, not to any hardware. Both quantum devices fall further short of even that value, 
at 262 (IBM) and 286 (Dirac-3), for two distinct reasons: the gate-based circuit is degraded by SWAP-routing depth on a sparse qubit lattice, while the photonic device, 
though it does optimise genuinely, settles well short of the best-known value that a free graphics processor reaches in seconds. 

