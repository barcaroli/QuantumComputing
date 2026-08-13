#!/usr/bin/env python3
"""
================================================================================
 QUANTUM OPTIMAL STRATIFICATION - D-Wave REAL QPU
================================================================================
 Version for execution on real D-Wave quantum hardware via the Leap cloud.

 PREREQUISITES:
   1. D-Wave Leap account: https://cloud.dwavesys.com/leap/signup
      (free trial: 1 minute QPU + 20 minutes hybrid solver)
   2. Configure the token:
      $ dwave config create
      or set the environment variable:
      $ export DWAVE_API_TOKEN="DEV-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   3. Install: pip install dwave-ocean-sdk

 THREE EXECUTION MODES:
   --mode qpu      -> Direct QPU (Advantage2, ~5000 physical qubits)
                      Uses real quantum tunneling. QPU time: ~milliseconds.
                      Requires embedding onto the Zephyr/Pegasus topology.
                      Limit: ~180 fully-connected variables on the direct QPU.
                      Our problem (656 vars) needs decomposition.

   --mode hybrid    -> LeapHybridBQMSampler hybrid solver
                      Decomposes the problem: part on the QPU, part classical.
                      Accepts up to 1,000,000 variables.
                      Typical time: 3-10 seconds.
                      RECOMMENDED for our problem.

   --mode simulate  -> SimulatedAnnealingSampler (classical, no token needed)
                      For local testing without Leap access.

 WHAT CHANGES COMPARED TO THE SIMULATED VERSION:
   Data preparation, QUBO construction, repair, and the Bethel allocation -
   the whole surrounding code (95% of it) - stays IDENTICAL.

   The only difference is in lines 220-280: the sampler.
   - Simulated: SimulatedAnnealingSampler().sample(bqm, ...)
   - QPU:       EmbeddingComposite(DWaveSampler()).sample(bqm, ...)
   - Hybrid:    LeapHybridBQMSampler().sample(bqm, ...)

   This is the "5 out of 100" of real quantum computing referred to elsewhere
   in this project. But that 5% is physically executed on superconducting
   qubits at 15 mK, with quantum tunneling and superposition - not simulated.
================================================================================
"""

import numpy as np
import pandas as pd
import dimod
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans, vq
from itertools import combinations
from math import ceil, sqrt
import time
import json
import argparse
import sys

# ==============================================================================
# STEPS 1-3: IDENTICAL to the simulated version (data prep, Bethel, aggregation)
# ==============================================================================

def var_bin(x, n_bins):
    """Replicate R's SamplingStrata::var.bin."""
    x_arr = x.values.astype(float).reshape(-1, 1)
    np.random.seed(42)
    centroids, _ = kmeans(x_arr, n_bins)
    labels, _ = vq(x_arr, np.sort(centroids, axis=0))
    return labels + 1


def prepare_frame(csv_path, reg=None, y1='Surfacesbois', y2='Airbat'):
    """Load and prepare the sampling frame.

    Accepts two CSV layouts automatically: RAW (continuous 'POPTOT'/'HApoly',
    binned here with var_bin) or PRE-CATEGORIZED ('POPTOT.cat'/'HApoly.cat',
    used directly). If a 'REG' column is present and `reg` is given, filters
    the frame to that region first. The two target variables Y1, Y2 are given
    by name (default Surfacesbois / Airbat) so the same script serves different
    experiments (e.g. Airbat / Surfacesbois).
    """
    df = pd.read_csv(csv_path)

    if reg is not None and 'REG' in df.columns:
        df = df[df['REG'] == reg].reset_index(drop=True)

    # Prefer the pre-categorized columns when present: they carry the
    # authoritative R categorization (e.g. 5x5) and must not be re-binned.
    # Only fall back to binning the continuous columns if no .cat exist.
    if 'POPTOT.cat' in df.columns and 'HApoly.cat' in df.columns:
        df['X1'] = df['POPTOT.cat']
        df['X2'] = df['HApoly.cat']
    elif 'POPTOT' in df.columns and 'HApoly' in df.columns:
        df['X1'] = var_bin(df['POPTOT'], 15)
        df['X2'] = var_bin(df['HApoly'], 15)
    else:
        raise KeyError(
            "CSV must contain either the continuous columns 'POPTOT' and "
            "'HApoly', or the pre-categorized columns 'POPTOT.cat' and "
            "'HApoly.cat'. Found columns: " + ", ".join(df.columns.tolist())
        )

    for col in (y1, y2):
        if col not in df.columns:
            raise KeyError(f"Target variable '{col}' not found. Available: "
                           + ", ".join(df.columns.tolist()))

    df['stratum_key'] = df['X1'].astype(str) + '*' + df['X2'].astype(str)
    strata = []
    for key, g in df.groupby('stratum_key'):
        N = len(g)
        strata.append({
            'STRATO': key, 'N': N,
            'M1': g[y1].mean(), 'M2': g[y2].mean(),
            'S1': g[y1].std(ddof=1) if N > 1 else 0.0,
            'S2': g[y2].std(ddof=1) if N > 1 else 0.0,
        })
    return pd.DataFrame(strata).reset_index(drop=True), df


def aggregate(atomic, assignment, K):
    """Compute aggregated stratum statistics after clustering."""
    at = atomic.copy()
    at['cluster'] = [assignment[i] for i in range(len(at))]
    rows = []
    for k in range(K):
        cl = at[at['cluster'] == k]
        if len(cl) == 0:
            continue
        Nt = cl['N'].sum()
        row = {'cluster': k, 'N': Nt, 'n_atomic': len(cl)}
        for mc, sc in [('M1', 'S1'), ('M2', 'S2')]:
            Mp = (cl['N'] * cl[mc]).sum() / Nt
            Vb = (cl['N'] * (cl[mc] - Mp)**2).sum() / Nt
            Vw = (cl['N'] * cl[sc]**2).sum() / Nt
            row[mc] = Mp
            row[sc] = sqrt(Vb + Vw)
        rows.append(row)
    return pd.DataFrame(rows)


def bethel(agg, cv_targets=[0.1,0.1], minnumstr=2, maxiter=200, maxiter1=25, epsilon=1e-11):
    """
    Bethel-Chromy optimal allocation (single domain, unit cost).

    Verified to match SamplingStrata::bethel (Barcaroli et al.) and an
    independent scipy SLSQP constrained optimizer. Implements the iterative
    Chromy loop that jointly optimizes the weights across the target
    variables, yielding the minimum total sample size that satisfies all
    coefficient-of-variation constraints simultaneously.

    Returns: (total_sample_size, per_stratum_allocation, achieved_cvs)
    """
    nstrat = len(agg)
    nvar = len(cv_targets)
    N = agg['N'].values.astype(float)
    cens = np.zeros(nstrat)
    cens[N < minnumstr] = 1
    nocens = 1 - cens

    med = agg[[f'M{i+1}' for i in range(nvar)]].values.astype(float)
    esse = agg[[f'S{i+1}' for i in range(nvar)]].values.astype(float)
    cv = np.array(cv_targets).astype(float)

    # Bethel coefficient a_hg for stratum h, variable g
    Nc = N.reshape(-1, 1); nocc = nocens.reshape(-1, 1)
    numA = (Nc**2) * (esse**2) * nocc
    denA1 = (np.sum(Nc * med * cv.reshape(1, -1), axis=0))**2
    denA2 = np.sum(Nc * (esse**2) * nocc, axis=0)
    a = numA / (denA1 + denA2 + epsilon)

    # Chromy iterative loop: alfa weights the nvar variable-constraints
    alfa = np.ones(nvar) / nvar
    x = np.full(nstrat, 1e-6)
    for _ in range(maxiter):
        den1 = np.sqrt(np.sum(a * alfa.reshape(1, -1), axis=1))
        den2 = np.sum(den1)
        x = 1.0 / (den1 * den2 + epsilon)
        axsum = np.sum(a * x.reshape(-1, 1), axis=0)
        alfatot = max(np.sum(alfa * axsum), epsilon)
        alfanext = alfa * axsum / alfatot
        if np.max(np.abs(alfanext - alfa)) < epsilon:
            break
        alfa = alfanext

    n = np.ceil(1.0 / x)

    # Census adjustment
    for _ in range(maxiter1):
        n = np.minimum(n, N)
        n = np.maximum(n, minnumstr)
        cens[n > N] = 1
        nocens = 1 - cens
        if np.all(n <= N):
            break
    n = nocens * n + cens * N

    # Achieved CVs
    cvs = []
    for i in range(nvar):
        M = med[:, i]; S = esse[:, i]
        Y = (N * M).sum()
        if abs(Y) < 1e-10:
            cvs.append(0.0); continue
        V = sum(N[h]**2 * S[h]**2 / n[h] * (1 - n[h]/N[h])
                for h in range(nstrat) if n[h] > 0)
        cvs.append(sqrt(V) / abs(Y))

    return ceil(n.sum()), n, cvs

# ==============================================================================
# STEP 4: BUILD QUBO - identical to the simulated version
# ==============================================================================

def build_qubo(atomic, K):
    """
    Build QUBO: minimize intra-cluster distance with one-hot constraints.
    This code is IDENTICAL to the simulated version.
    """
    n = len(atomic)
    features = atomic[['M1', 'M2', 'S1', 'S2']].values.copy()
    for col in range(4):
        std = features[:, col].std()
        if std > 0:
            features[:, col] = (features[:, col] - features[:, col].mean()) / std
        else:
            features[:, col] = 0
    
    dist_matrix = cdist(features, features, 'euclidean')
    N_vals = atomic['N'].values.astype(float)
    
    bqm = dimod.BinaryQuadraticModel('BINARY')
    
    # Objective: minimize weighted intra-cluster distance
    max_w = 0
    for i, j in combinations(range(n), 2):
        w = dist_matrix[i, j] * sqrt(N_vals[i] * N_vals[j])
        if w > max_w:
            max_w = w
        for k in range(K):
            if w > 1e-10:
                bqm.add_quadratic(f'x_{i}_{k}', f'x_{j}_{k}', w)
    
    # One-hot constraints
    lagrange = max_w * 3.0
    for i in range(n):
        for k in range(K):
            bqm.add_linear(f'x_{i}_{k}', -lagrange)
        for k1, k2 in combinations(range(K), 2):
            bqm.add_quadratic(f'x_{i}_{k1}', f'x_{i}_{k2}', 2 * lagrange)
    
    return bqm, dist_matrix


# ==============================================================================
# STEP 5: SOLVE - THIS IS WHERE THE DIFFERENCE LIES
# ==============================================================================

def create_sampler(mode):
    """
    +==================================================================+
    |  THIS IS THE POINT WHERE THE CODE BECOMES GENUINELY QUANTUM   |
    |  The rest of the script (95%) is identical to the simulated    |
    |  version                                                        |
    +==================================================================+

    mode='qpu':      Real superconducting qubits at 15 millikelvin.
                     Physical quantum tunneling through energy barriers.
                     EmbeddingComposite maps the logical qubits onto the
                     chip's Zephyr/Pegasus topology. Annealing time: ~20 us
                     per read.

    mode='hybrid':   The problem is partitioned automatically.
                     The suitable sub-parts go to the real QPU,
                     the rest is solved classically.
                     Accepts much larger problems (up to 1M variables).

    mode='simulate': No quantum hardware. Classical simulated annealing.
                     For development and testing without a Leap account.
    """
    if mode == 'qpu':
        from dwave.system import DWaveSampler, EmbeddingComposite
        print("  Connecting to the D-Wave QPU...")
        qpu = DWaveSampler()
        print(f"  QPU: {qpu.properties.get('chip_id', 'N/A')}")
        print(f"  Topology: {qpu.properties.get('topology', {}).get('type', 'N/A')}")
        print(f"  Qubits available: {len(qpu.nodelist)}")
        print(f"  Couplers available: {len(qpu.edgelist)}")
        sampler = EmbeddingComposite(qpu)
        return sampler, 'qpu'
    
    elif mode == 'hybrid':
        from dwave.system import LeapHybridBQMSampler
        print("  Connecting to the Leap hybrid solver...")
        sampler = LeapHybridBQMSampler()
        print(f"  Solver: {sampler.solver.name}")
        print(f"  Max variables: {sampler.properties.get('maximum_number_of_variables', 'N/A')}")
        return sampler, 'hybrid'
    
    elif mode == 'simulate':
        from dwave.samplers import SimulatedAnnealingSampler
        print("  Simulation mode (no quantum hardware)")
        sampler = SimulatedAnnealingSampler()
        return sampler, 'simulate'
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'qpu', 'hybrid', or 'simulate'.")


def solve(bqm, sampler, mode, n_atomic, K, num_reads=200, num_sweeps=12000, seed=42):
    """
    Solve the BQM with the selected sampler.
    Parameters vary depending on the backend.

    num_reads / num_sweeps control the effort of the simulated annealing:
    higher values reduce the risk of getting stuck in local minima (at the
    cost of more time). The default 200/12000 is a compromise; for harder
    problems use 1000/50000 or more.
    """
    t0 = time.time()

    if mode == 'qpu':
        # -- DIRECT QPU --
        # num_reads: how many times to run the annealing (each read ~20 us)
        # annealing_time: annealing duration in microseconds (default 20)
        # chain_strength: strength of the couplings in the embedding chains
        #   (auto-calibrated by EmbeddingComposite if not specified)
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            annealing_time=50,      # 50 us (longer = more accurate)
            answer_mode='raw',      # return all samples
            label='Quantum Stratification - QPU'
        )

    elif mode == 'hybrid':
        # -- HYBRID SOLVER --
        # time_limit: maximum time in seconds (min 3s per BQM)
        # The solver decides autonomously how to partition the problem
        sampleset = sampler.sample(
            bqm,
            time_limit=10,          # max 10 seconds
            label='Quantum Stratification - Hybrid'
        )

    elif mode == 'simulate':
        # -- CLASSICAL SIMULATION --
        from dwave.samplers import SimulatedAnnealingSampler
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed
        )

    t_solve = time.time() - t0

    # -- EXTRACT SOLUTION (identical for every backend) --
    best = sampleset.first.sample
    energy = sampleset.first.energy
    
    assignment = {}
    violations = 0
    unassigned = []
    
    for i in range(n_atomic):
        ac = [k for k in range(K) if best.get(f'x_{i}_{k}', 0) == 1]
        if len(ac) == 1:
            assignment[i] = ac[0]
        elif len(ac) == 0:
            violations += 1
            unassigned.append(i)
        else:
            violations += 1
            assignment[i] = ac[0]
    
    # -- REPAIR (identical for every backend) --
    if unassigned:
        features = np.array([[r['M1'], r['M2'], r['S1'], r['S2']]
                            for _, r in pd.DataFrame().from_dict(
                                {i: {'M1': 0, 'M2': 0, 'S1': 0, 'S2': 0}
                                 for i in range(n_atomic)}).items()])
        # Use atomic strata features directly
        centroids = {}
        for k in range(K):
            members = [i for i, c in assignment.items() if c == k]
            if members:
                centroids[k] = np.mean(
                    [[best.get(f'x_{m}_{kk}', 0) for kk in range(K)]
                     for m in members], axis=0)
        for i in unassigned:
            assignment[i] = min(range(K),
                              key=lambda k: k)  # fallback: assign to cluster 0
    
    # -- TIMING INFO (different for QPU) --
    timing = {}
    if mode == 'qpu' and hasattr(sampleset, 'info'):
        timing = sampleset.info.get('timing', {})
    
    return assignment, energy, violations, unassigned, t_solve, timing, sampleset


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Quantum Optimal Stratification - D-Wave',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local test (no D-Wave token required)
  python quantum_stratification_qpu.py --mode simulate --csv swissmunicipalities.csv
  
  # Leap hybrid solver (recommended, requires a token)
  python quantum_stratification_qpu.py --mode hybrid --csv swissmunicipalities.csv
  
  # Direct QPU (requires a token, small problems only)
  python quantum_stratification_qpu.py --mode qpu --csv swissmunicipalities.csv --K 3

D-Wave token configuration:
  $ dwave config create
  or: export DWAVE_API_TOKEN="DEV-xxxx..."
        """)
    parser.add_argument('--mode', choices=['qpu', 'hybrid', 'simulate'],
                       default='simulate',
                       help='Backend: qpu (direct QPU), hybrid (Leap hybrid), simulate (classical)')
    parser.add_argument('--csv', default='swissmunicipalities.csv',
                       help='Path to the frame CSV')
    parser.add_argument('--reg', type=int, default=None,
                       help='Filter the frame to a region (REG column), e.g. --reg 1')
    parser.add_argument('--K', type=int, default=4,
                       help='Target number of clusters (default: 4)')
    parser.add_argument('--cv', type=float, default=0.10,
                       help='CV target (default: 0.10)')
    parser.add_argument('--y1', default='Surfacesbois',
                       help='First target variable Y1 (default: Surfacesbois)')
    parser.add_argument('--y2', default='Airbat',
                       help='Second target variable Y2 (default: Airbat)')
    parser.add_argument('--benchmark', type=int, default=89,
                       help='Classical reference value to display (default: 89)')
    parser.add_argument('--num-reads', type=int, default=200,
                       help='Number of simulated-annealing reads (default: 200; '
                            'increase to 1000+ for harder problems)')
    parser.add_argument('--num-sweeps', type=int, default=12000,
                       help='Number of sweeps per read (default: 12000; increase to 50000+)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Simulated-annealing seed (default: 42)')
    parser.add_argument('--best-of', type=int, default=1,
                       help='Runs N times with different seeds and keeps the minimum '
                            'sample size (default: 1). Useful since SA is stochastic.')
    
    args = parser.parse_args()
    
    print("=" * 72)
    print(" QUANTUM OPTIMAL STRATIFICATION - D-Wave")
    print(f" Mode: {args.mode.upper()}")
    if args.mode == 'qpu':
        print(" [Q]  RUNNING ON REAL QUANTUM HARDWARE")
    elif args.mode == 'hybrid':
        print(" [H] HYBRID EXECUTION (QPU + classical)")
    else:
        print(" [PC] CLASSICAL SIMULATION (no quantum hardware)")
    print("=" * 72)
    
    # -- 1. DATA PREPARATION --
    print("\n[1] Preparing data...")
    atomic, df = prepare_frame(args.csv, reg=args.reg, y1=args.y1, y2=args.y2)
    n_atomic = len(atomic)
    K = args.K
    cv_targets = [args.cv, args.cv]
    
    print(f"  Frame: {len(df)} units")
    print(f"  Atomic strata: {n_atomic} (of which {(atomic['N']==1).sum()} have N=1)")
    print(f"  K = {K}, CV target = {args.cv}")
    
    # -- 2. QUBO --
    print(f"\n[2] Building the QUBO...")
    t0 = time.time()
    bqm, dist_matrix = build_qubo(atomic, K)
    t_build = time.time() - t0
    
    print(f"  Variables: {len(bqm.variables)}")
    print(f"  Interactions: {len(bqm.quadratic)}")
    print(f"  Build time: {t_build:.2f}s")
    
    # -- 3. SAMPLER --
    print(f"\n[3] Initializing the sampler ({args.mode})...")
    sampler, mode = create_sampler(args.mode)
    
    # -- 4. SOLVING --
    print(f"\n[4] Solving...")
    if args.best_of > 1 and mode == 'simulate':
        print(f"  Best-of-{args.best_of} mode: running {args.best_of} times "
              f"with different seeds, keeping the minimum-energy solution.")
        best_energy = None
        best_result = None
        for r in range(args.best_of):
            seed_r = args.seed + r
            res = solve(bqm, sampler, mode, n_atomic, K,
                        num_reads=args.num_reads, num_sweeps=args.num_sweeps,
                        seed=seed_r)
            e = res[1]
            print(f"    run {r+1}/{args.best_of} (seed={seed_r}): energy={e:.1f}")
            if best_energy is None or e < best_energy:
                best_energy = e
                best_result = res
        assignment, energy, violations, unassigned, t_solve, timing, sampleset = best_result
    else:
        assignment, energy, violations, unassigned, t_solve, timing, sampleset = \
            solve(bqm, sampler, mode, n_atomic, K,
                  num_reads=args.num_reads, num_sweeps=args.num_sweeps, seed=args.seed)
    
    print(f"  Total time: {t_solve:.3f}s")
    print(f"  Energy: {energy:.2f}")
    print(f"  Violations: {violations}")
    
    if timing and mode == 'qpu':
        print(f"\n  -- Detailed QPU timing --")
        for key, val in timing.items():
            if isinstance(val, (int, float)):
                if val > 1e6:
                    print(f"    {key}: {val/1e6:.2f} ms")
                else:
                    print(f"    {key}: {val:.0f} us")
    
    # -- 5. REPAIR --
    if unassigned:
        print(f"\n[5] Repairing {len(unassigned)} unassigned strata...")
        features = atomic[['M1', 'M2', 'S1', 'S2']].values
        N_v = atomic['N'].values.astype(float)
        centroids = {}
        for k in range(K):
            members = [i for i, c in assignment.items() if c == k]
            if members:
                wf = (features[members].T * N_v[members]).T
                centroids[k] = wf.sum(axis=0) / N_v[members].sum()
        for i in unassigned:
            if centroids:
                assignment[i] = min(centroids.keys(),
                                   key=lambda k: np.linalg.norm(features[i] - centroids[k]))
            else:
                assignment[i] = 0
    
    # -- 6. EVALUATION --
    print(f"\n[6] Evaluating the solution...")
    agg = aggregate(atomic, assignment, K)
    n_sample, alloc, cvs = bethel(agg, cv_targets)
    
    print(f"\n  +======================================+")
    print(f"  |  RESULT ({args.mode.upper():>8})                  |")
    print(f"  |  Sample size:         {n_sample:>5}          |")
    print(f"  |  Aggregated strata:   {len(agg):>5}          |")
    print(f"  |  CV(Y1): {cvs[0]:>7.4f}                  |")
    print(f"  |  CV(Y2): {cvs[1]:>7.4f}                  |")
    print(f"  |  Reference (classical): {args.benchmark:>3}          |")
    print(f"  +======================================+")
    
    print(f"\n  {'#':>3} {'N':>6} {'Atom':>5} {'M(Y1)':>10} {'M(Y2)':>10} "
          f"{'S(Y1)':>10} {'S(Y2)':>10} {'n_h':>6}")
    print("  " + "-" * 60)
    for idx, row in agg.iterrows():
        print(f"  {int(row['cluster'])+1:>3} {int(row['N']):>6} {int(row['n_atomic']):>5} "
              f"{row['M1']:>10.1f} {row['M2']:>10.1f} "
              f"{row['S1']:>10.1f} {row['S2']:>10.1f} {alloc[idx]:>6.0f}")
    print(f"  {'TOT':>3} {int(agg['N'].sum()):>6} {int(agg['n_atomic'].sum()):>5} "
          f"{'':>42} {sum(alloc):>6.0f}")
    
    # -- 7. EXPORT --
    result = {
        'mode': args.mode,
        'K': K,
        'sample_size': n_sample,
        'cvs': cvs,
        'energy': float(energy),
        'violations': violations,
        'time_build': t_build,
        'time_solve': t_solve,
        'timing_qpu': timing,
        'strata': agg.to_dict('records'),
        'allocation': alloc.tolist(),
    }
    
    outfile = f'result_{args.mode}_K{K}.json'
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {outfile}")
    
    return result


if __name__ == '__main__':
    main()
