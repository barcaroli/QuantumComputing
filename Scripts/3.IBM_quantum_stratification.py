#!/usr/bin/env python3
"""
================================================================================
 QUANTUM OPTIMAL STRATIFICATION — IBM Quantum (Qiskit QAOA)
================================================================================
 Solves the optimal stratification problem using QAOA on real IBM hardware.
 
 PREREQUISITES:
   1. IBM Quantum account (free): https://quantum.ibm.com
      Open plan: 10 min/month on real 100+ qubit processors
   2. pip install qiskit qiskit-ibm-runtime qiskit-algorithms
      pip install pandas numpy scipy
   3. Save the IBM token (available in the IBM Quantum dashboard):
      In the IBM dashboard click your user icon in the top-right corner,
      then "Manage account" -> "API token" -> copy the token.
 
 REAL-HARDWARE STRATEGY:
   The results reported in the accompanying paper use the reduced,
   twenty-atomic-stratum frame (swissmunicipalities, region REG=1,
   discretised into 5x5 = 20 non-empty atomic strata). At that scale no
   classical pre-grouping is needed: each atomic stratum is encoded
   directly with a binary-label Hamiltonian, 1 qubit/stratum for K=2 or
   2 qubits/stratum for K=4 -> 40 qubits for the 20-stratum, K=4 case
   actually run on ibm_kingston. This is the DEFAULT behaviour below.
   With direct encoding, the only classical stages are the Hamiltonian
   construction, transpilation, the COBYLA parameter loop, decoding and
   the final Bethel-Chromy evaluation; the state preparation/measurement
   is genuinely quantum. The binding constraint at this scale is circuit
   depth from SWAP-routing on the sparse heavy-hex lattice, NOT qubit
   count (see the paper, Sections 4.6 and 5.1).

   An earlier, exploratory attempt used a much larger frame (164 atomic
   strata, one-hot QUBO would need 656 qubits at K=4) which is far
   beyond what NISQ hardware or any of the free tiers tested could
   accept; that frame was in fact what got the Dirac-3 free-tier
   submission rejected, and was abandoned in favour of the 20-stratum
   frame before any real hardware run was performed (paper, Section
   5.1). For that scale, a two-level strategy is still available here
   as an OPT-IN, legacy path (--force-pregroup):
   1. Classical pre-grouping: N atomic strata -> --n-macro macro-strata
      (k-means on standardised stratum features)
   2. Quantum QAOA: partition the macro-strata into K clusters
      (1 qubit/macro-stratum for K=2, 2 qubits/macro-stratum for K=4)
   This legacy path is only ~10% quantum: the pre-grouping step is
   classical and coarsens the problem before QAOA ever runs. It is kept
   here for experimentation with frames too large for direct encoding,
   but it is NOT the pipeline behind the results in the paper.

 MODES:
   --mode simulator   Local Qiskit simulator (no token required)
   --mode hardware    Real IBM QPU via cloud (token required)

 ENCODING:
   Default            Direct: 1 qubit/atomic stratum (K=2) or 2 qubits/
                       atomic stratum (K=4). Matches the paper exactly
                       when run on the 20-stratum reduced frame.
   --force-pregroup    Legacy: classical k-means pre-grouping into
                       --n-macro macro-strata before QAOA (for frames
                       too large to encode directly, e.g. the original
                       164-stratum exploratory frame). Not used for the
                       results reported in the paper.

 USAGE:
   # Local (test), direct encoding on the 20-stratum reduced frame
   python quantum_stratification_ibm.py --mode simulator --csv swissmunicipalities.csv

   # Real IBM hardware, direct encoding (as in the paper)
   python quantum_stratification_ibm.py --mode hardware --csv swissmunicipalities.csv \
       --token "YOUR_IBM_TOKEN"

   # Legacy pre-grouped path, for a much larger frame
   python quantum_stratification_ibm.py --mode simulator --csv large_frame.csv \
       --force-pregroup --n-macro 12
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from itertools import combinations
from math import ceil, sqrt
import time
import json
import argparse

# ==============================================================================
# DATA PREPARATION (same as the D-Wave version)
# ==============================================================================

def var_bin(x, n_bins):
    x_arr = x.values.astype(float).reshape(-1, 1)
    np.random.seed(42)
    centroids, _ = kmeans(x_arr, n_bins)
    labels, _ = vq(x_arr, np.sort(centroids, axis=0))
    return labels + 1

def prepare_frame(csv_path):
    df = pd.read_csv(csv_path)
    if 'POPTOT' in df.columns:
        df['X1'] = var_bin(df['POPTOT'], 15)
    else:
        df['X1'] = df['POPTOT.cat'].astype(int)
    if 'HApoly' in df.columns:
        df['X2'] = var_bin(df['HApoly'], 15)
    else:
        df['X2'] = df['HApoly.cat'].astype(int)
    df['stratum_key'] = df['X1'].astype(str) + '*' + df['X2'].astype(str)
    strata = []
    for key, g in df.groupby('stratum_key'):
        N = len(g)
        strata.append({
            'STRATO': key, 'N': N,
            'M1': g['Surfacesbois'].mean(), 'M2': g['Airbat'].mean(),
            'S1': g['Surfacesbois'].std(ddof=1) if N > 1 else 0.0,
            'S2': g['Airbat'].std(ddof=1) if N > 1 else 0.0,
        })
    return pd.DataFrame(strata).reset_index(drop=True), df

def aggregate(atomic, assignment, K):
    at = atomic.copy()
    at['cluster'] = [assignment[i] for i in range(len(at))]
    rows = []
    for k in range(K):
        cl = at[at['cluster'] == k]
        if len(cl) == 0: continue
        Nt = cl['N'].sum()
        row = {'cluster': k, 'N': Nt, 'n_atomic': len(cl)}
        for mc, sc in [('M1','S1'), ('M2','S2')]:
            Mp = (cl['N']*cl[mc]).sum()/Nt
            Vb = (cl['N']*(cl[mc]-Mp)**2).sum()/Nt
            Vw = (cl['N']*cl[sc]**2).sum()/Nt
            row[mc] = Mp; row[sc] = sqrt(Vb+Vw)
        rows.append(row)
    return pd.DataFrame(rows)

def bethel(agg, cv_targets=[0.1,0.1], minnumstr=2):
    H = len(agg); N_h = agg['N'].values.astype(float)
    allocs = []
    for (mc,sc), cv in zip([('M1','S1'),('M2','S2')], cv_targets):
        M, S = agg[mc].values.astype(float), agg[sc].values.astype(float)
        Y = (N_h*M).sum()
        if abs(Y)<1e-10: allocs.append(np.zeros(H)); continue
        Vm=(cv*Y)**2; num=((N_h*S).sum())**2; den=Vm+(N_h*S**2).sum()
        ng=num/den if den>0 else N_h.sum()
        tNS=(N_h*S).sum()
        allocs.append(ng*(N_h*S)/tNS if tNS>0 else np.ones(H)*ng/H)
    n_tot = max(a.sum() for a in allocs)
    fa = np.zeros(H)
    for al in allocs:
        s=al.sum(); fa=np.maximum(fa, al*(n_tot/s) if s>0 else al)
    for h in range(H):
        fa[h]=max(fa[h],minnumstr) if N_h[h]>=minnumstr else N_h[h]
    fa = np.minimum(fa, N_h)
    cvs = []
    for mc,sc in [('M1','S1'),('M2','S2')]:
        M,S=agg[mc].values.astype(float),agg[sc].values.astype(float)
        Y=(N_h*M).sum()
        V=sum(N_h[h]**2*S[h]**2/fa[h]*(1-fa[h]/N_h[h]) for h in range(H) if fa[h]>0)
        cvs.append(sqrt(V)/abs(Y) if Y else 0)
    return ceil(fa.sum()), fa, cvs


# ==============================================================================
# STEP 1: MACRO-STRATA (direct encoding by default, legacy pre-grouping opt-in)
# ==============================================================================

def direct_group(atomic):
    """
    DEFAULT PATH - matches the paper.
    Wrap each atomic stratum as its own "macro-stratum" (identity
    mapping, member_indices = [idx]). No classical aggregation happens
    here: every atomic stratum gets its own qubit(s) downstream, exactly
    as in the direct binary-label encoding used for the real IBM run on
    the 20-stratum frame (paper, Section 4.6). This keeps the rest of
    the pipeline (build_ising_hamiltonian, decode_counts,
    macro_to_atomic_assignment) unchanged, since it already operates on
    a "macro_strata" dataframe - here that dataframe is just the atomic
    strata themselves.
    """
    macro = []
    for idx, row in atomic.reset_index(drop=True).iterrows():
        macro.append({
            'macro_id': idx, 'N': row['N'], 'n_atomic': 1,
            'member_indices': [idx],
            'M1': row['M1'], 'M2': row['M2'],
            'S1': row['S1'], 'S2': row['S2'],
        })
    labs = np.arange(len(atomic))
    return pd.DataFrame(macro), labs


def pre_group(atomic, n_groups=12):
    """
    LEGACY, OPT-IN PATH (--force-pregroup) - not used in the paper.
    Pre-group the atomic strata into n_groups macro-strata using
    k-means, for frames too large to encode one qubit (or two, for
    K=4) per atomic stratum directly - e.g. the original ~164-stratum
    exploratory frame mentioned in the paper (Section 5.1), which was
    abandoned before any real-hardware run in favour of the 20-stratum
    frame that direct_group() above handles without any aggregation.
    This step is entirely CLASSICAL, and it coarsens the problem before
    QAOA ever sees it.
    """
    feats = atomic[['M1','M2','S1','S2']].values.copy()
    for c in range(4):
        s = feats[:,c].std()
        feats[:,c] = (feats[:,c]-feats[:,c].mean())/(s if s>0 else 1)
    
    np.random.seed(42)
    cents, _ = kmeans(feats.astype(float), n_groups)
    labs, _ = vq(feats, cents)
    
    # Build macro-strata statistics
    macro = []
    for g in range(n_groups):
        mask = labs == g
        sub = atomic[mask]
        if len(sub) == 0: continue
        Nt = sub['N'].sum()
        row = {'macro_id': g, 'N': Nt, 'n_atomic': len(sub),
               'member_indices': list(sub.index)}
        for mc, sc in [('M1','S1'),('M2','S2')]:
            Mp = (sub['N']*sub[mc]).sum()/Nt
            Vb = (sub['N']*(sub[mc]-Mp)**2).sum()/Nt
            Vw = (sub['N']*sub[sc]**2).sum()/Nt
            row[mc] = Mp; row[sc] = sqrt(Vb+Vw)
        macro.append(row)
    
    return pd.DataFrame(macro), labs


# ==============================================================================
# STEP 2: QAOA - QUBO -> Ising Hamiltonian -> QAOA Circuit
# ==============================================================================

def build_ising_hamiltonian(macro_strata, K=2):
    """
    Build the Ising Hamiltonian for the partitioning problem.
    
    For K=2: each macro-stratum is one qubit. z_i = +1 (cluster 0) or -1 (cluster 1).
    Objective: minimize the intra-cluster distance.
    
    H = Σ_{i<j} J_ij · Z_i · Z_j + Σ_i h_i · Z_i
    
    where J_ij > 0 if i,j should be in the SAME cluster (attractive)
    and J_ij < 0 if they should be in DIFFERENT clusters.
    
    For K=4: we use 2 qubits per macro-stratum (binary encoding).
    
    Returns: list of SparsePauliOp terms for Qiskit
    """
    n = len(macro_strata)
    N_v = macro_strata['N'].values.astype(float)
    
    # Compute merge cost matrix
    merge_cost = np.zeros((n, n))
    for mc, sc in [('M1','S1'), ('M2','S2')]:
        M = macro_strata[mc].values.astype(float)
        S = macro_strata[sc].values.astype(float)
        Y = (N_v * M).sum()
        scale = 1.0/(Y**2) if abs(Y)>1e-10 else 1.0
        for i in range(n):
            for j in range(i+1, n):
                hN = N_v[i]*N_v[j]/(N_v[i]+N_v[j])
                cost = hN * (M[i]-M[j])**2 * scale
                merge_cost[i,j] += cost
                merge_cost[j,i] += cost
    
    if K == 2:
        # -- Direct encoding: 1 qubit per macro-stratum --
        # x_i ∈ {0,1} → Z_i = 1-2x_i → x_i = (1-Z_i)/2
        # Obj: Σ_{i<j} w_ij * x_i * x_j + Σ_{i<j} w_ij * (1-x_i)*(1-x_j)
        # In Ising: Σ_{i<j} w_ij/2 * (1 + Z_i*Z_j)
        # We want to MINIMIZE within-cluster distance -> minimize when
        # similar strata are together -> penalize DISSIMILAR pairs in the same cluster
        
        n_qubits = n
        pauli_list = []
        
        for i, j in combinations(range(n), 2):
            w = merge_cost[i,j]
            if w > 1e-12:
                # ZZ term: w/2 * Z_i Z_j
                label = ['I'] * n_qubits
                label[i] = 'Z'
                label[j] = 'Z'
                pauli_list.append((''.join(reversed(label)), w/2))
        
        return pauli_list, n_qubits, merge_cost
    
    elif K == 4:
        # -- Binary encoding: 2 qubits per macro-stratum --
        # cluster k ∈ {0,1,2,3} → (q_{2i}, q_{2i+1}) 
        # Same cluster = both qubit pairs are equal
        
        n_qubits = 2 * n
        pauli_list = []
        
        for i, j in combinations(range(n), 2):
            w = merge_cost[i,j]
            if w < 1e-12: continue
            
            # Two strata are in the same cluster when BOTH qubit pairs match
            # Match on bit 0: (1 + Z_{2i}*Z_{2j})/2
            # Match on bit 1: (1 + Z_{2i+1}*Z_{2j+1})/2
            # Same cluster penalty: w/4 * (1+Z_{2i}Z_{2j}) * (1+Z_{2i+1}Z_{2j+1})
            # = w/4 * (1 + Z_{2i}Z_{2j} + Z_{2i+1}Z_{2j+1} + Z_{2i}Z_{2j}Z_{2i+1}Z_{2j+1})
            
            # ZZ term for bit 0
            label = ['I'] * n_qubits
            label[2*i] = 'Z'; label[2*j] = 'Z'
            pauli_list.append((''.join(reversed(label)), w/4))
            
            # ZZ term for bit 1
            label = ['I'] * n_qubits
            label[2*i+1] = 'Z'; label[2*j+1] = 'Z'
            pauli_list.append((''.join(reversed(label)), w/4))
            
            # ZZZZ term (4-body)
            label = ['I'] * n_qubits
            label[2*i] = 'Z'; label[2*j] = 'Z'
            label[2*i+1] = 'Z'; label[2*j+1] = 'Z'
            pauli_list.append((''.join(reversed(label)), w/4))
        
        return pauli_list, n_qubits, merge_cost
    
    else:
        raise ValueError(f"K={K} not supported. Use K=2 or K=4.")


def solve_qaoa(pauli_list, n_qubits, mode='simulator', token=None, 
               p=2, maxiter=100, shots=4096):
    """
    ╔══════════════════════════════════════════════════════════════════╗
    |  THIS IS THE GENUINELY QUANTUM PART OF THE SCRIPT        |
    |  The QAOA circuit runs on a real IBM QPU (or simulator) |
    +---------------------------------------------------------+
    """
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from scipy.optimize import minimize

    def build_qaoa_ansatz(hamiltonian, reps):
        n = hamiltonian.num_qubits
        gamma = ParameterVector('γ', reps)
        beta  = ParameterVector('β', reps)
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for layer in range(reps):
            for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
                label = pauli.to_label()  # e.g. 'ZZII', MSB first
                active = [i for i, c in enumerate(reversed(label)) if c != 'I']
                if not active:
                    continue
                cr = float(coeff.real)
                if len(active) == 1:
                    qc.rz(2 * gamma[layer] * cr, active[0])
                else:
                    for k in range(len(active) - 1):
                        qc.cx(active[k], active[k + 1])
                    qc.rz(2 * gamma[layer] * cr, active[-1])
                    for k in range(len(active) - 2, -1, -1):
                        qc.cx(active[k], active[k + 1])
            qc.rx(2 * beta[layer], range(n))
        return qc

    # Build cost Hamiltonian
    hamiltonian = SparsePauliOp.from_list(pauli_list)

    # Build QAOA ansatz
    ansatz = build_qaoa_ansatz(hamiltonian, reps=p)
    print(f"    QAOA circuit: {ansatz.num_qubits} qubits, p={p}")
    print(f"    Variational parameters: {ansatz.num_parameters}")
    
    if mode == 'simulator':
        # -- LOCAL SIMULATOR --
        from qiskit.primitives import StatevectorEstimator
        estimator = StatevectorEstimator()
        
        # Initial parameters
        x0 = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
        
        objective_values = []
        
        def cost_fn(params):
            bound = ansatz.assign_parameters(dict(zip(ansatz.parameters, params)))
            pub = (bound, hamiltonian)
            result = estimator.run([pub]).result()
            cost = result[0].data.evs
            objective_values.append(float(cost))
            return float(cost)
        
        print(f"    COBYLA optimization (max {maxiter} iterations)...")
        t0 = time.time()
        result = minimize(cost_fn, x0, method='COBYLA',
                         options={'maxiter': maxiter, 'rhobeg': 0.5})
        t_opt = time.time() - t0
        
        # Sample the optimal circuit
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler()
        
        optimal_circuit = ansatz.assign_parameters(
            dict(zip(ansatz.parameters, result.x)))
        optimal_circuit.measure_all()
        
        pub = (optimal_circuit,)
        sample_result = sampler.run([pub], shots=shots).result()
        counts = sample_result[0].data.meas.get_counts()
        
        return counts, result.fun, t_opt, objective_values
    
    elif mode == 'hardware':
        # -- REAL IBM HARDWARE (open plan: single-job mode) --
        # Open plan does not allow Sessions. Strategy:
        #   1. Optimise parameters locally with StatevectorEstimator (fast, free)
        #   2. Submit one Sampler job to QPU with the optimal circuit
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit.primitives import StatevectorEstimator

        # -- Step 1: classical parameter optimization --
        print(f"    Optimizing parameters (local simulator)...")
        estimator = StatevectorEstimator()
        x0 = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
        objective_values = []

        def cost_fn(params):
            bound = ansatz.assign_parameters(dict(zip(ansatz.parameters, params)))
            result = estimator.run([(bound, hamiltonian)]).result()
            cost = float(result[0].data.evs)
            objective_values.append(cost)
            if len(objective_values) % 10 == 0:
                print(f"      Iteration {len(objective_values)}: cost={cost:.4f}")
            return cost

        t0 = time.time()
        opt_result = minimize(cost_fn, x0, method='COBYLA',
                              options={'maxiter': maxiter, 'rhobeg': 0.5})
        t_opt = time.time() - t0
        print(f"    Optimal parameters found (energy={opt_result.fun:.4f})")

        # ── Step 2: single QPU job for sampling ──
        print(f"    Connecting to IBM Quantum...")
        service = QiskitRuntimeService(
            channel='ibm_quantum_platform',
            token=token
        )

        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=n_qubits
        )
        print(f"    Backend: {backend.name} ({backend.num_qubits} qubits)")

        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        optimal_circuit = ansatz.assign_parameters(
            dict(zip(ansatz.parameters, opt_result.x)))
        optimal_circuit.measure_all()
        isa_circuit = pm.run(optimal_circuit)

        print(f"    Transpiled circuit: depth={isa_circuit.depth()}, "
              f"CX gates={isa_circuit.count_ops().get('cx', 0) + isa_circuit.count_ops().get('ecr', 0)}")
        print(f"    Submitting QPU job ({shots} shots)...")

        sampler = Sampler(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        print(f"    Job ID: {job.job_id()}  — waiting for results...")
        sample_result = job.result()
        counts = sample_result[0].data.meas.get_counts()

        return counts, opt_result.fun, t_opt, objective_values
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ==============================================================================
# STEP 3: DECODE QAOA RESULTS → CLUSTER ASSIGNMENT
# ==============================================================================

def decode_counts(counts, n_macro, K=2):
    """
    Decode QAOA bitstrings into cluster assignments.
    Return the bitstring with the highest count.
    """
    # Sort by frequency
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    
    best_bitstring = sorted_counts[0][0]
    
    if K == 2:
        # 1 qubit per macro-stratum
        assignment = {}
        for i in range(n_macro):
            if i < len(best_bitstring):
                assignment[i] = int(best_bitstring[i])
            else:
                assignment[i] = 0
    elif K == 4:
        # 2 qubits per macro-stratum
        assignment = {}
        for i in range(n_macro):
            b0 = int(best_bitstring[2*i]) if 2*i < len(best_bitstring) else 0
            b1 = int(best_bitstring[2*i+1]) if 2*i+1 < len(best_bitstring) else 0
            assignment[i] = b0 + 2*b1
    
    return assignment, sorted_counts[:10]


def macro_to_atomic_assignment(macro_assignment, macro_strata, n_atomic):
    """
    Map the macro-strata assignment to the atomic-strata assignment.
    """
    atomic_assignment = {}
    for macro_idx, cluster_id in macro_assignment.items():
        member_indices = macro_strata.iloc[macro_idx]['member_indices']
        for atom_idx in member_indices:
            atomic_assignment[atom_idx] = cluster_id
    return atomic_assignment


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Quantum Stratification — IBM Quantum (QAOA)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local test
  python quantum_stratification_ibm.py --mode simulator --csv swissmunicipalities.csv
  
  # Real IBM hardware
  python quantum_stratification_ibm.py --mode hardware --csv swissmunicipalities.csv \\
      --token "YOUR_IBM_QUANTUM_TOKEN"

To obtain the IBM token:
  1. Go to https://quantum.ibm.com
  2. Register (free, Open plan: 10 min/month)
  3. Dashboard -> user icon -> Manage account -> API token -> Copy
        """)
    parser.add_argument('--mode', choices=['simulator', 'hardware'],
                       default='simulator')
    parser.add_argument('--csv', default='swissmunicipalities.csv')
    parser.add_argument('--token', default=None,
                       help='IBM Quantum API token')
    parser.add_argument('--K', type=int, default=2, choices=[2, 4],
                       help='Number of clusters (2 or 4). With direct encoding (default): '
                            '1 qubit/atomic stratum for K=2, 2 qubits/atomic stratum for K=4.')
    parser.add_argument('--force-pregroup', action='store_true',
                       help='Legacy path, not used for the results in the paper: '
                            'classically groups (k-means) the atomic strata into '
                            '--n-macro macro-strata before QAOA. Only useful for frames '
                            'too large for direct encoding (e.g. the original '
                            'exploratory ~164-stratum frame, later abandoned).')
    parser.add_argument('--n-macro', type=int, default=12,
                       help='Number of macro-strata, used only with --force-pregroup (default: 12)')
    parser.add_argument('--p', type=int, default=2,
                       help='QAOA depth (default: 2)')
    parser.add_argument('--maxiter', type=int, default=80,
                       help='Max optimizer iterations (default: 80)')
    
    args = parser.parse_args()
    
    print("=" * 72)
    print(" QUANTUM STRATIFICATION — IBM Quantum (QAOA)")
    print(f" Mode: {args.mode.upper()}")
    if args.mode == 'hardware':
        print(" ⚛️  RUNNING ON A REAL IBM QUANTUM PROCESSOR")
    else:
        print(" 💻 LOCAL SIMULATOR (no token required)")
    print("=" * 72)
    
    # -- 1. DATA --
    print("\n[1] Preparing data...")
    atomic, df = prepare_frame(args.csv)
    n_atomic = len(atomic)
    K = args.K
    
    print(f"    Frame: {len(df)} units")
    print(f"    Atomic strata: {n_atomic}")
    print(f"    K = {K}")
    
    # -- 2. MACRO-STRATA: direct encoding (default) or legacy pre-grouping --
    if args.force_pregroup:
        print(f"\n[2] (legacy, --force-pregroup) Classical pre-grouping: "
              f"{n_atomic} -> {args.n_macro} macro-strata")
        print(f"    WARNING: this path is not the one used for the results in the paper; "
              f"it is only ~10% quantum (QAOA already sees a classically coarsened problem).")
        macro, pre_labels = pre_group(atomic, n_groups=args.n_macro)
    else:
        print(f"\n[2] Direct encoding (no classical grouping): "
              f"{n_atomic} atomic strata -> {n_atomic} units to encode")
        print(f"    This is the path used for the results in the paper on the "
              f"20-atomic-stratum frame (Section 4.6).")
        macro, pre_labels = direct_group(atomic)
    n_macro = len(macro)
    print(f"    Units to encode: {n_macro}")
    for _, row in macro.iterrows():
        print(f"      Unit {int(row['macro_id'])}: {int(row['n_atomic'])} atomic, "
              f"N={int(row['N'])}, M1={row['M1']:.0f}, M2={row['M2']:.1f}")
    
    # -- 3. ISING HAMILTONIAN --
    print(f"\n[3] Building the Ising Hamiltonian...")
    pauli_list, n_qubits, merge_cost = build_ising_hamiltonian(macro, K)
    print(f"    Pauli terms: {len(pauli_list)}")
    print(f"    Qubits required: {n_qubits}")
    
    # ── 4. QAOA ──
    print(f"\n[4] QAOA (p={args.p})...")
    print(f"    ╔════════════════════════════════════════════════╗")
    print(f"    ║  THIS IS THE GENUINELY QUANTUM COMPUTATION    ║")
    if args.mode == 'hardware':
        print(f"    ║  Executed on real IBM superconducting qubits  ║")
    else:
        print(f"    ║  Simulated locally (no quantum effect)        ║")
    print(f"    ╚════════════════════════════════════════════════╝")
    
    np.random.seed(42)
    counts, energy, t_opt, obj_values = solve_qaoa(
        pauli_list, n_qubits, 
        mode=args.mode, token=args.token,
        p=args.p, maxiter=args.maxiter
    )
    
    print(f"\n    Optimization time: {t_opt:.1f}s")
    print(f"    Final energy: {energy:.4f}")
    print(f"    Iterations: {len(obj_values)}")
    
    # -- 5. DECODING --
    print(f"\n[5] Decoding results...")
    macro_assign, top_bitstrings = decode_counts(counts, n_macro, K)
    
    print(f"    Top 5 bitstrings:")
    for bs, count in top_bitstrings[:5]:
        print(f"      {bs}: {count} counts")
    
    print(f"    Macro-strata assignment: {macro_assign}")
    
    # -- 6. MAP TO ATOMIC STRATA --
    print(f"\n[6] Mapping to atomic strata...")
    atomic_assign = macro_to_atomic_assignment(macro_assign, macro, n_atomic)
    
    # -- 7. EVALUATION --
    print(f"\n[7] Bethel evaluation...")
    agg = aggregate(atomic, atomic_assign, K)
    n_sample, alloc, cvs = bethel(agg)
    
    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║  QAOA RESULT ({args.mode.upper():>9})                  ║")
    print(f"  ║  Sample size:         {n_sample:>5}                ║")
    print(f"  ║  Aggregated strata:   {len(agg):>5}                ║")
    print(f"  ║  CV(Y1): {cvs[0]:>7.4f}                        ║")
    print(f"  ║  CV(Y2): {cvs[1]:>7.4f}                        ║")
    print(f"  ║  Qubits used:         {n_qubits:>5}                ║")
    print(f"  ║  QAOA depth:          p={args.p:>3}                ║")
    print(f"  ║  Reference n (GA):       89                ║")
    print(f"  ╚══════════════════════════════════════════════╝")
    
    print(f"\n  {'#':>3} {'N':>6} {'Atom':>5} {'M(Y1)':>10} {'M(Y2)':>10} "
          f"{'S(Y1)':>10} {'S(Y2)':>10} {'n_h':>6}")
    print("  " + "-" * 60)
    for idx, row in agg.iterrows():
        print(f"  {int(row['cluster'])+1:>3} {int(row['N']):>6} {int(row['n_atomic']):>5} "
              f"{row['M1']:>10.1f} {row['M2']:>10.1f} "
              f"{row['S1']:>10.1f} {row['S2']:>10.1f} {alloc[idx]:>6.0f}")
    
    # -- 8. EXPORT --
    result = {
        'mode': args.mode,
        'K': K,
        'n_qubits': n_qubits,
        'qaoa_depth': args.p,
        'sample_size': n_sample,
        'cvs': cvs,
        'energy': float(energy),
        'n_macro': n_macro,
        'strata': agg.to_dict('records'),
        'allocation': alloc.tolist(),
        'convergence': obj_values,
    }
    
    outfile = f'result_ibm_{args.mode}_K{K}.json'
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {outfile}")
    
    return result


if __name__ == '__main__':
    main()
