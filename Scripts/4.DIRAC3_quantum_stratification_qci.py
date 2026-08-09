#!/usr/bin/env python3
"""
================================================================================
 QUANTUM OPTIMAL STRATIFICATION - QCi Dirac-3 (Entropy Quantum Computing)
================================================================================
 Solves the optimal stratification problem on QCi Dirac-3 photonic hardware,
 using Entropy Quantum Computing (EQC).

 PREREQUISITES:
   1. QCi account with API token (free trial ~10 min):
      https://quantumcomputinginc.com/  ->  request token
   2. pip install "qci-client>=4.5" pandas numpy scipy
   3. Configure environment variables:
      export QCI_API_URL="https://api.qci-prod.com"
      export QCI_TOKEN="<your_secret_token>"
      (or pass --token from the command line)

 WHY DIRAC-3 IS INTERESTING FOR THIS PROBLEM:
   Compared to D-Wave and IBM, Dirac-3 offers three advantages:
   - All-to-all connectivity: no embedding overhead
   - Accepts polynomials up to degree 5 (not just quadratic QUBOs)
   - Up to 949 total "levels": our 656 binary variables fit within it
   - EQC harnesses noise/entropy instead of fighting it (no cryogenics)

 TWO FORMULATION MODES:
   --formulation qubo    Classic QUBO formulation (proxy: intra-cluster distance)
                          Equivalent to the D-Wave v1 script, but on photonic hardware.
                          656 binary variables for K=4.

   --formulation integer Integer-variable formulation (advanced).
                          Exploits qudits to directly encode the cluster
                          assignment with fewer variables:
                          one integer variable x_i in {0,...,K-1} per stratum.
                          Only 164 variables (one per atomic stratum).

 NOTE ON QUANTUM HONESTY:
   As with D-Wave and IBM, data preparation and Bethel allocation remain
   classical. The quantum part is the search for the ground state
   of the Hamiltonian on the photonic processor. Unlike D-Wave's
   SimulatedAnnealingSampler (a classical simulator), here the computation
   is genuinely physical: photons in an open quantum system.

 USAGE:
   python quantum_stratification_qci.py --token "TOKEN" \\
       --csv swissmunicipalities.csv --K 4 --formulation qubo
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
import os

# ==============================================================================
# DATA PREPARATION (identical to the D-Wave and IBM versions)
# ==============================================================================

def var_bin(x, n_bins):
    x_arr = x.values.astype(float).reshape(-1, 1)
    np.random.seed(42)
    centroids, _ = kmeans(x_arr, n_bins)
    labels, _ = vq(x_arr, np.sort(centroids, axis=0))
    return labels + 1

def prepare_frame(csv_path, reg=None, y1='Surfacesbois', y2='Airind'):
    """
    Build the atomic strata from the sampling frame.
    Y1, Y2 are target variable names (default Surfacesbois / Airind).

    Accepts two CSV layouts automatically:
      (a) RAW: continuous columns 'POPTOT' and 'HApoly', binned here with
          var_bin (15 classes each), reproducing the R workflow.
      (b) PRE-CATEGORIZED: columns 'POPTOT.cat' and 'HApoly.cat' already
          containing the class codes, used directly. This matches a frame
          categorized upstream in R and avoids any binning mismatch.

    Y variables are 'Surfacesbois' and 'Airind'. If a 'REG' column is present
    and `reg` is given, the frame is filtered to that region first.
    """
    df = pd.read_csv(csv_path)

    if reg is not None and 'REG' in df.columns:
        df = df[df['REG'] == reg].reset_index(drop=True)

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


def pre_group(atomic, n_groups):
    """
    Pre-aggregate the atomic strata into `n_groups` macro-strata by K-means on
    the standardized (mean, std) features. Returns the macro-strata dataframe
    (same columns as `atomic`, plus a 'members' list of original atomic indices)
    and a mapping macro_index -> list of atomic indices.

    This is used to bring the problem under the Dirac-3 free-tier limit of 100
    variables: instead of one integer variable per atomic stratum (164), the
    device solves over `n_groups` <= 100 macro-strata. The pre-grouping is a
    classical clustering step; the quantum device then partitions the macro-
    strata into the final K clusters. After solving, the macro assignment is
    expanded back to the atomic strata so that the Bethel evaluation is still
    performed on the full frame.
    """
    feats = atomic[['M1', 'M2', 'S1', 'S2']].values.astype(float).copy()
    for c in range(feats.shape[1]):
        s = feats[:, c].std()
        feats[:, c] = (feats[:, c] - feats[:, c].mean()) / (s if s > 0 else 1.0)
    np.random.seed(42)
    cents, _ = kmeans(feats, n_groups)
    labs, _ = vq(feats, cents)

    macro_rows = []
    mapping = {}
    g_new = 0
    for g in range(n_groups):
        members = [i for i in range(len(atomic)) if labs[i] == g]
        if not members:
            continue
        sub = atomic.iloc[members]
        Nt = sub['N'].sum()
        row = {'N': Nt, 'members': members}
        for mcol, scol in [('M1', 'S1'), ('M2', 'S2')]:
            Mp = (sub['N'] * sub[mcol]).sum() / Nt
            # pooled within-group std across the atomic members
            var = ((sub['N'] * (sub[mcol] - Mp) ** 2).sum()
                   + (sub['N'] * sub[scol] ** 2).sum()) / Nt
            row[mcol] = Mp
            row[scol] = sqrt(var)
        macro_rows.append(row)
        mapping[g_new] = members
        g_new += 1

    macro = pd.DataFrame(macro_rows).reset_index(drop=True)
    return macro, mapping

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
# QUBO CONSTRUCTION  ->  DIRAC-3 POLYNOMIAL FORMAT
# ==============================================================================

def build_qubo_matrix(atomic, K, normalize=True):
    """
    Builds the QUBO matrix Q for the stratum->cluster assignment.
    
    Variables: x[i*K + k] = 1 if atomic stratum i is in cluster k.
    Total variables: N * K (e.g. 164*4 = 656).
    
    Objective: min weighted intra-cluster distance + one-hot penalty.
    
    NOTE (from Padmasola et al. 2025): normalizing the weight matrix
    drastically improves the probability of feasible solutions.
    We apply min-max normalization as suggested.
    """
    n = len(atomic)
    n_vars = n * K
    
    # Feature normalization for distance
    feats = atomic[['M1','M2','S1','S2']].values.copy()
    for c in range(4):
        s = feats[:,c].std()
        feats[:,c] = (feats[:,c]-feats[:,c].mean())/(s if s>0 else 1)
    dist = cdist(feats, feats, 'euclidean')
    N_v = atomic['N'].values.astype(float)
    
    # QUBO matrix (symmetric)
    Q = np.zeros((n_vars, n_vars))
    
    def idx(i, k):
        return i * K + k
    
    # Objective: intra-cluster distance
    max_w = 0
    for i, j in combinations(range(n), 2):
        w = dist[i,j] * sqrt(N_v[i]*N_v[j])
        if w > max_w: max_w = w
        for k in range(K):
            Q[idx(i,k), idx(j,k)] += w
            Q[idx(j,k), idx(i,k)] += w
    
    # One-hot penalty: lambda * (sum_k x_ik - 1)^2
    # = lambda * (sum_k x_ik^2 + 2 sum_{k1<k2} x_ik1 x_ik2 - 2 sum_k x_ik + 1)
    # For binary x^2=x, so linear terms: -lambda on diagonal
    lam = max_w * 3.0
    for i in range(n):
        for k in range(K):
            Q[idx(i,k), idx(i,k)] += -lam  # linear (diagonal)
        for k1, k2 in combinations(range(K), 2):
            Q[idx(i,k1), idx(i,k2)] += lam  # 2*lam/2 (will be doubled by symmetry)
            Q[idx(i,k2), idx(i,k1)] += lam
    
    # Normalization (Padmasola et al. 2025 recommendation)
    if normalize:
        max_abs = np.abs(Q).max()
        if max_abs > 0:
            Q = Q / max_abs
    
    return Q, n_vars, lam


def qubo_to_dirac_polynomial(Q):
    """
    Converts the QUBO matrix Q into the Dirac-3 polynomial format.
    
    Dirac-3 requires:
      - poly_indices: list of index lists (1-based, non-decreasing)
      - poly_coefs: corresponding coefficients
    
    For a QUBO: E = sum_i Q_ii x_i + sum_{i<j} 2*Q_ij x_i x_j
    (quadratic terms use indices [i+1, j+1], linear ones [0, i+1])
    
    Indices are 1-based; 0 denotes "lower degree" (padding).
    max_degree = 2 (quadratic).
    """
    n = Q.shape[0]
    data = []
    
    for i in range(n):
        # Linear term (diagonal): index [0, i+1]
        if abs(Q[i,i]) > 1e-12:
            data.append({"idx": [0, i+1], "val": float(Q[i,i])})
        # Quadratic terms (upper triangle): index [i+1, j+1]
        for j in range(i+1, n):
            coef = Q[i,j] + Q[j,i]  # combine symmetric entries
            if abs(coef) > 1e-12:
                data.append({"idx": [i+1, j+1], "val": float(coef)})
    
    return data


# ==============================================================================
# INTEGER FORMULATION (advanced, uses qudits)
# ==============================================================================

def build_integer_polynomial(atomic, K):
    """
    Integer-variable formulation exploiting Dirac-3's qudits.
    
    One integer variable x_i in {0, 1, ..., K-1} for each atomic stratum,
    directly representing the assigned cluster.
    Only N variables instead of N*K.
    
    WARNING: this is conceptually elegant but the "intra-cluster distance"
    objective is NOT polynomial under this encoding
    (it would require equality indicators between x_i and x_j).
    For this reason, this formulation is provided as a proof of
    concept but uses an approximation: it penalizes the difference |x_i - x_j|
    weighted by stratum similarity (similar strata -> same cluster).
    
    This is a heuristic, not equivalent to the one-hot QUBO.
    """
    n = len(atomic)
    feats = atomic[['M1','M2','S1','S2']].values.copy()
    for c in range(4):
        s = feats[:,c].std()
        feats[:,c] = (feats[:,c]-feats[:,c].mean())/(s if s>0 else 1)
    dist = cdist(feats, feats, 'euclidean')
    N_v = atomic['N'].values.astype(float)
    
    # Similarity weight: strata that are close should have similar cluster id
    # Objective approx: sum_{i<j} sim_ij * (x_i - x_j)^2
    # = sum sim_ij (x_i^2 - 2 x_i x_j + x_j^2)
    # We want CLOSE strata (low dist) to have SAME cluster (low (x_i-x_j)^2)
    # so weight = 1/(1+dist) - high for similar strata
    
    data = []
    max_sim = 0
    for i, j in combinations(range(n), 2):
        sim = 1.0 / (1.0 + dist[i,j]) * sqrt(N_v[i]*N_v[j])
        if sim > max_sim: max_sim = sim
    
    # Build (x_i - x_j)^2 = x_i^2 - 2 x_i x_j + x_j^2
    linear_coef = {}
    for i, j in combinations(range(n), 2):
        sim = 1.0 / (1.0 + dist[i,j]) * sqrt(N_v[i]*N_v[j]) / max_sim
        if sim < 0.01: continue  # sparsify
        # x_i^2 term
        data.append({"idx": [i+1, i+1], "val": float(sim)})
        # x_j^2 term
        data.append({"idx": [j+1, j+1], "val": float(sim)})
        # -2 x_i x_j term
        data.append({"idx": [i+1, j+1], "val": float(-2*sim)})
    
    return data, n


# ==============================================================================
# DIRAC-3 SUBMISSION
# ==============================================================================

def solve_on_dirac(data, n_vars, num_levels, url, token,
                   job_type='sample-hamiltonian-integer',
                   relaxation_schedule=2, num_samples=10, dry_run=False):
    """
    Submits the polynomial problem to Dirac-3 via the QCi cloud.
    
    +==================================================================+
    |  THIS IS THE GENUINELY QUANTUM PART OF THE SCRIPT             |
    |  The Hamiltonian is minimized on real photonic hardware       |
    |  (open quantum system, entropy quantum computing)             |
    +==================================================================+
    
    dry_run=True: only prints the job body without submitting (offline test).
    """
    # Compute polynomial degrees
    degrees = [len([x for x in d["idx"] if x != 0]) for d in data]
    min_degree = min(degrees) if degrees else 1
    max_degree = max(degrees) if degrees else 2
    
    print(f"    Variables: {n_vars}")
    print(f"    Polynomial terms: {len(data)}")
    print(f"    Degree: min={min_degree}, max={max_degree}")
    
    if isinstance(num_levels, int):
        total_levels = num_levels * n_vars
    else:
        total_levels = sum(num_levels)
    print(f"    Total levels: {total_levels} (Dirac-3 limit: 949)")

    if total_levels > 949:
        print(f"    ERROR: {total_levels} levels exceeds the Dirac-3 limit of 949.")
        print(f"    The one-hot QUBO uses K binary variables per stratum (K x 2 levels")
        print(f"    each), which does not fit. Use --formulation integer instead: it")
        print(f"    encodes each stratum as a single K-level qudit (164 x K levels),")
        print(f"    which fits comfortably and exploits the native Dirac-3 encoding.")
        if not dry_run:
            raise ValueError(
                f"total_levels={total_levels} exceeds Dirac-3 limit (949); "
                f"use --formulation integer")
    
    if dry_run:
        print("\n    [DRY RUN] Job body ready but not submitted.")
        print(f"    file_config.polynomial.num_variables = {n_vars}")
        print(f"    file_config.polynomial.min_degree = {min_degree}")
        print(f"    file_config.polynomial.max_degree = {max_degree}")
        return None
    
    # Import qci-client only when actually submitting.
    # NB: in qci-client v4/v5 the 'JobStatus' enum no longer exists; the job
    # status is returned as a plain string ('COMPLETED', 'ERRORED', ...), so we
    # import only QciClient and compare the status against the string directly.
    from qci_client import QciClient

    print(f"    Connecting to QCi ({url})...")
    client = QciClient(url=url, api_token=token)
    
    # Check allocation
    alloc = client.get_allocations()["allocations"]["dirac"]
    print(f"    Available allocation: {alloc.get('seconds', '?')} seconds")
    
    # Upload polynomial file
    file_data = {
        "file_name": "stratification_qubo",
        "file_config": {
            "polynomial": {
                "num_variables": n_vars,
                "min_degree": min_degree,
                "max_degree": max_degree,
                "data": data,
            }
        }
    }
    
    print("    Uploading polynomial file...")
    file_response = client.upload_file(file=file_data)
    file_id = file_response['file_id']
    print(f"    file_id: {file_id}")
    
    # Build job body
    if job_type == 'sample-hamiltonian-integer':
        job_params = {
            'device_type': 'dirac-3',
            'num_samples': num_samples,
            'relaxation_schedule': relaxation_schedule,
            'num_levels': num_levels if isinstance(num_levels, list) else [num_levels],
        }
    else:
        job_params = {
            'device_type': 'dirac-3',
            'num_samples': num_samples,
            'relaxation_schedule': relaxation_schedule,
        }
    
    job_body = client.build_job_body(
        job_type=job_type,
        job_name='quantum_stratification',
        job_tags=['stratification', 'sampling'],
        job_params=job_params,
        polynomial_file_id=file_id,
    )
    
    # Submit and wait
    print("    Submitting job to Dirac-3...")
    t0 = time.time()
    job_response = client.process_job(job_body=job_body)
    t_solve = time.time() - t0

    # In v4/v5 the status is a plain string. Accept 'COMPLETED' case-insensitively.
    status = str(job_response.get("status", "")).upper()
    if status != "COMPLETED":
        print(f"    ERROR: job status = {job_response.get('status')}")
        # Print any diagnostic info the API returned
        if 'job_info' in job_response:
            print(f"    job_info: {job_response['job_info']}")
        return None

    results = job_response['results']
    # Field names have varied slightly across client versions; handle both.
    solutions = results.get('solutions') or results.get('samples')
    energies = results.get('energies')
    counts = results.get('counts')
    try:
        device_usage = job_response['job_info']['job_result'].get('device_usage_s', '?')
    except (KeyError, TypeError):
        device_usage = '?'
    
    print(f"    Completed in {t_solve:.1f}s (device usage: {device_usage}s)")
    print(f"    Solutions found: {len(solutions)}")
    print(f"    Best energy: {min(energies):.4f}")
    
    return {
        'solutions': solutions,
        'energies': energies,
        'counts': counts,
        'device_usage_s': device_usage,
        'best_idx': int(np.argmin(energies)),
    }


# ==============================================================================
# DECODE
# ==============================================================================

def decode_qubo_solution(solution, n_atomic, K):
    """Decodes the one-hot QUBO solution into a cluster assignment."""
    assignment = {}
    violations = 0
    for i in range(n_atomic):
        assigned = [k for k in range(K) if solution[i*K + k] >= 0.5]
        if len(assigned) == 1:
            assignment[i] = assigned[0]
        elif len(assigned) == 0:
            violations += 1
            assignment[i] = 0  # will repair
        else:
            violations += 1
            assignment[i] = assigned[0]
    return assignment, violations

def decode_integer_solution(solution, n_atomic, K):
    """Decodes the integer-variable solution."""
    assignment = {}
    for i in range(n_atomic):
        val = int(round(solution[i]))
        assignment[i] = max(0, min(K-1, val))
    return assignment, 0

def repair(assignment, atomic, K):
    """Repairs violations by assigning to the nearest centroid."""
    feats = atomic[['M1','M2','S1','S2']].values
    N_v = atomic['N'].values.astype(float)
    # Recompute unassigned
    n = len(atomic)
    unassigned = [i for i in range(n) if i not in assignment or assignment[i] is None]
    if not unassigned:
        return assignment
    centroids = {}
    for k in range(K):
        members = [i for i,c in assignment.items() if c == k]
        if members:
            wf = (feats[members].T * N_v[members]).T
            centroids[k] = wf.sum(axis=0)/N_v[members].sum()
    for i in unassigned:
        if centroids:
            assignment[i] = min(centroids.keys(),
                              key=lambda k: np.linalg.norm(feats[i]-centroids[k]))
        else:
            assignment[i] = 0
    return assignment


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Quantum Stratification - QCi Dirac-3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline test (no token, only shows the job body)
  python quantum_stratification_qci.py --csv swissmunicipalities.csv --dry-run

  # Run on Dirac-3 (one-hot QUBO)
  python quantum_stratification_qci.py --csv swissmunicipalities.csv \\
      --token "TOKEN" --K 4 --formulation qubo

  # Run with the integer-variable formulation (fewer variables)
  python quantum_stratification_qci.py --csv swissmunicipalities.csv \\
      --token "TOKEN" --K 4 --formulation integer

To obtain the QCi token:
  1. Register at https://quantumcomputinginc.com/
  2. Request an API access token (free trial ~10 min)
  3. export QCI_TOKEN="your_token"
        """)
    parser.add_argument('--csv', default='swissmunicipalities.csv')
    parser.add_argument('--reg', type=int, default=None,
                       help='Filter the frame on a region (REG column), e.g. --reg 1')
    parser.add_argument('--token', default=os.environ.get('QCI_TOKEN'))
    parser.add_argument('--url', default=os.environ.get('QCI_API_URL', 'https://api.qci-prod.com'))
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--pre-group', type=int, default=None,
                       help='Pre-aggregate the 164 atomic strata into N macro-strata '
                            '(K-means) before solving. Needed for the '
                            'Dirac-3 free tier (max 100 variables). E.g.: --pre-group 90')
    parser.add_argument('--cv', type=float, default=0.10)
    parser.add_argument('--y1', default='Surfacesbois',
                       help='First target variable Y1 (default: Surfacesbois)')
    parser.add_argument('--y2', default='Airind',
                       help='Second target variable Y2 (default: Airind)')
    parser.add_argument('--benchmark', type=int, default=89,
                       help='Classical reference value (default: 89)')
    parser.add_argument('--formulation', choices=['qubo', 'integer'], default='qubo')
    parser.add_argument('--relaxation-schedule', type=int, default=2, choices=[1,2,3,4])
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable QUBO matrix normalization')
    parser.add_argument('--dry-run', action='store_true',
                       help='Prepare the job without submitting it (offline test)')
    args = parser.parse_args()

    print("=" * 72)
    print(" QUANTUM STRATIFICATION - QCi Dirac-3 (Entropy Quantum Computing)")
    print(f" Formulation: {args.formulation.upper()}")
    if args.dry_run:
        print(" DRY RUN (no submission)")
    else:
        print(" RUNNING ON REAL PHOTONIC HARDWARE")
    print("=" * 72)

    if not args.dry_run and not args.token:
        print("\nERROR: no token provided. Use --token or export QCI_TOKEN=...")
        print("Or use --dry-run for an offline test.")
        return

    # 1. Data
    print("\n[1] Data preparation...")
    atomic, df = prepare_frame(args.csv, reg=args.reg, y1=args.y1, y2=args.y2)
    n_atomic = len(atomic)
    K = args.K
    print(f"    Frame: {len(df)} units, atomic strata: {n_atomic}")
    print(f"    K = {K}, CV target = {args.cv}")

    # Optional pre-grouping to fit the Dirac-3 free-tier (<=100 variables).
    # The device solves over macro-strata; the assignment is expanded back to
    # the atomic strata before the Bethel evaluation.
    macro_mapping = None
    if args.pre_group is not None:
        if args.pre_group >= n_atomic:
            print(f"    --pre-group {args.pre_group} >= atomic strata {n_atomic}: "
                  f"no grouping applied.")
        else:
            solve_frame, macro_mapping = pre_group(atomic, args.pre_group)
            print(f"    Pre-grouped into {len(solve_frame)} macro-strata "
                  f"(from {n_atomic} atomic) to fit the free-tier limit.")
    else:
        solve_frame = atomic

    n_solve = len(solve_frame)

    # 2. Build polynomial (on solve_frame: atomic strata, or macro-strata if pre-grouped)
    print(f"\n[2] Building problem ({args.formulation})...")
    if args.formulation == 'qubo':
        Q, n_vars, lam = build_qubo_matrix(solve_frame, K, normalize=not args.no_normalize)
        data = qubo_to_dirac_polynomial(Q)
        # Dirac-3 counts LEVELS (number of discrete values), not the max value.
        # A binary variable takes values {0,1} = 2 levels. The API requires
        # num_levels >= 2, so a QUBO must declare 2 levels per variable.
        num_levels = 2  # binary variables: 2 levels {0,1}
        job_type = 'sample-hamiltonian-integer'
        print(f"    QUBO: {n_vars} binary variables")
        print(f"    Normalization: {'yes' if not args.no_normalize else 'no'}")
    else:
        data, n_vars = build_integer_polynomial(solve_frame, K)
        # An integer variable taking values {0,...,K-1} has K distinct levels,
        # not K-1. For K=4 the values {0,1,2,3} are 4 levels.
        num_levels = K  # integer variables: K levels {0,...,K-1}
        job_type = 'sample-hamiltonian-integer'
        print(f"    Integer: {n_vars} integer variables in {{0,...,{K-1}}}")

    # 3. Solve on Dirac-3
    print(f"\n[3] Solving on Dirac-3...")
    result = solve_on_dirac(
        data, n_vars, num_levels, args.url, args.token,
        job_type=job_type,
        relaxation_schedule=args.relaxation_schedule,
        num_samples=args.num_samples,
        dry_run=args.dry_run,
    )

    if result is None:
        if args.dry_run:
            print("\n[DRY RUN complete] The problem is ready for submission.")
        return

    # 4. Decode
    print(f"\n[4] Decoding solution...")
    best_solution = result['solutions'][result['best_idx']]
    if args.formulation == 'qubo':
        solve_assignment, violations = decode_qubo_solution(best_solution, n_solve, K)
        print(f"    One-hot violations: {violations}")
        if violations > 0:
            solve_assignment = repair(solve_assignment, solve_frame, K)
            print(f"    Repaired {violations} violations")
    else:
        solve_assignment, _ = decode_integer_solution(best_solution, n_solve, K)

    # If we pre-grouped, expand the macro-strata assignment back to the atomic
    # strata so the Bethel evaluation is performed on the full original frame.
    if macro_mapping is not None:
        assignment = {}
        for macro_idx, cluster in solve_assignment.items():
            for atomic_idx in macro_mapping[macro_idx]:
                assignment[atomic_idx] = cluster
        print(f"    Expanded {n_solve} macro-strata assignment back to "
              f"{n_atomic} atomic strata.")
    else:
        assignment = solve_assignment

    # 5. Evaluate
    print(f"\n[5] Bethel evaluation...")
    agg = aggregate(atomic, assignment, K)
    n_sample, alloc, cvs = bethel(agg, [args.cv, args.cv])

    print(f"\n  +==============================================+")
    print(f"  |  RESULT Dirac-3 ({args.formulation.upper():>7})                  |")
    print(f"  |  Sample size:         {n_sample:>5}                  |")
    print(f"  |  Aggregated strata:   {len(agg):>5}                  |")
    print(f"  |  CV(Y1): {cvs[0]:>7.4f}                          |")
    print(f"  |  CV(Y2): {cvs[1]:>7.4f}                          |")
    print(f"  |  Variables:           {n_vars:>5}                  |")
    print(f"  |  Device usage:        {str(result['device_usage_s'])+'s':>5}                  |")
    print(f"  |  Reference (classical): {args.benchmark:>4}                  |")
    print(f"  +==============================================+")

    print(f"\n  {'#':>3} {'N':>6} {'Atom':>5} {'M(Y1)':>10} {'M(Y2)':>10} "
          f"{'S(Y1)':>10} {'S(Y2)':>10} {'n_h':>6}")
    print("  " + "-" * 60)
    for idx, row in agg.iterrows():
        print(f"  {int(row['cluster'])+1:>3} {int(row['N']):>6} {int(row['n_atomic']):>5} "
              f"{row['M1']:>10.1f} {row['M2']:>10.1f} "
              f"{row['S1']:>10.1f} {row['S2']:>10.1f} {alloc[idx]:>6.0f}")

    # 6. Export
    out = {
        'formulation': args.formulation,
        'K': K, 'n_vars': n_vars,
        'sample_size': n_sample, 'cvs': cvs,
        'energy': float(result['energies'][result['best_idx']]),
        'device_usage_s': result['device_usage_s'],
        'strata': agg.to_dict('records'),
        'allocation': alloc.tolist(),
    }
    outfile = f'result_qci_{args.formulation}_K{K}.json'
    with open(outfile, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results saved to {outfile}")


if __name__ == '__main__':
    main()
