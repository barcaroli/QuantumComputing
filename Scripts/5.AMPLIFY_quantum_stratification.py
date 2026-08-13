"""
========================================================================
 QUANTUM STRATIFICATION - Fixstars Amplify
 Dense QUBO on purpose-built Ising machines
========================================================================

Solves the same one-hot QUBO used for D-Wave and Dirac-3, but on Ising
machines that are DESIGNED for densely connected problems:

  ae        Fixstars Amplify AE   GPU annealer, >100k bits, FREE token
  toshiba   Toshiba SQBM+         Simulated Bifurcation Machine
  fujitsu   Fujitsu Digital Annealer  fully-connected CMOS QUBO hardware
  dwave     D-Wave Advantage      (needs your own D-Wave token)
  gurobi    Gurobi                exact/branch-and-bound (needs licence)

WHY THIS MATTERS FOR THIS PROBLEM
---------------------------------
Our QUBO is DENSE: every atomic stratum interacts with every other one.
That is exactly what killed the gate-based run (SWAP routing overhead)
and what forces D-Wave into long minor-embedding chains. The machines
above accept a fully-connected QUBO natively, with no embedding and no
routing. So they measure something the other runs could not: how good
the BEST achievable solution of the surrogate really is.

Interpretation of the result:
  * if it lands near the simulated-annealing value (n=162), that CONFIRMS
    162 is the ceiling imposed by the QUBO formulation itself;
  * if it lands clearly below 162, then part of the gap we attributed to
    the formulation was in fact our own solver being too weak.
Either way the answer is informative, and on Amplify AE it is free.

GETTING A FREE TOKEN
--------------------
  1. Register at https://amplify.fixstars.com  (free edition)
  2. Copy the 32-character token from the "Access Token" page
  3. Pass it with --token, or set the env var AMPLIFY_TOKEN

USAGE
-----
  pip install amplify

  # Free GPU annealer (Amplify AE)
  python quantum_stratification_amplify.py --client ae ^
      --csv swiss_municipalities.csv --K 4 --cv 0.05 ^
      --y1 Airbat --y2 Surfacesbois --benchmark 129 --token "YOUR_TOKEN"

  # Toshiba SQBM+ (also available to registered Amplify users)
  python quantum_stratification_amplify.py --client toshiba ... --token "YOUR_TOKEN"

  # Build the model without solving (no token needed)
  python quantum_stratification_amplify.py --dry-run --csv swiss_municipalities.csv
========================================================================
"""

import argparse
import json
import os
import time
from datetime import timedelta
from itertools import combinations
from math import ceil, sqrt

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------
# 1. FRAME PREPARATION  (identical to the D-Wave / IBM / Dirac-3 scripts)
# ----------------------------------------------------------------------
def var_bin(x, nbins):
    """K-means binning of a continuous variable (mimics R var.bin)."""
    xa = x.values.astype(float).reshape(-1, 1)
    np.random.seed(42)
    centroids, _ = kmeans(xa, nbins)
    labels, _ = vq(xa, np.sort(centroids, axis=0))
    return labels + 1


def prepare_frame(csv_path, reg=None, y1='Surfacesbois', y2='Airbat'):
    """
    Build the atomic strata. Accepts either the pre-categorized layout
    (POPTOT.cat / HApoly.cat, used as-is) or the raw one (POPTOT / HApoly,
    binned here). The .cat columns take priority: they carry the
    authoritative R categorization and must not be re-binned.
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
            "CSV must contain either 'POPTOT'/'HApoly' or "
            "'POPTOT.cat'/'HApoly.cat'. Found: " + ", ".join(df.columns))

    for col in (y1, y2):
        if col not in df.columns:
            raise KeyError(f"Target variable '{col}' not found. "
                           f"Available: {', '.join(df.columns)}")

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


# ----------------------------------------------------------------------
# 2. BETHEL-CHROMY ALLOCATION  (identical to the other scripts)
# ----------------------------------------------------------------------
def bethel(agg, cv_targets=(0.1, 0.1), minnumstr=2,
           maxiter=200, maxiter1=25, epsilon=1e-11):
    """Multivariate optimal allocation. Matches SamplingStrata::bethel."""
    nstr = len(agg)
    nvar = len(cv_targets)
    N = agg['N'].values.astype(float)

    cens = np.zeros(nstr)
    cens[N < minnumstr] = 1
    noc = 1 - cens

    med = agg[[f'M{i+1}' for i in range(nvar)]].values.astype(float)
    esse = agg[[f'S{i+1}' for i in range(nvar)]].values.astype(float)
    cv = np.array(cv_targets, dtype=float)

    Nc = N.reshape(-1, 1)
    nocc = noc.reshape(-1, 1)
    numA = (Nc ** 2) * (esse ** 2) * nocc
    dA1 = (np.sum(Nc * med * cv.reshape(1, -1), axis=0)) ** 2
    dA2 = np.sum(Nc * (esse ** 2) * nocc, axis=0)
    a = numA / (dA1 + dA2 + epsilon)

    alfa = np.ones(nvar) / nvar
    x = np.full(nstr, 1e-6)
    for _ in range(maxiter):
        d1 = np.sqrt(np.sum(a * alfa.reshape(1, -1), axis=1))
        d2 = np.sum(d1)
        x = 1.0 / (d1 * d2 + epsilon)
        axs = np.sum(a * x.reshape(-1, 1), axis=0)
        tot = max(np.sum(alfa * axs), epsilon)
        alfa_new = alfa * axs / tot
        if np.max(np.abs(alfa_new - alfa)) < epsilon:
            break
        alfa = alfa_new

    n = np.ceil(1.0 / x)
    for _ in range(maxiter1):
        n = np.minimum(n, N)
        n = np.maximum(n, minnumstr)
        cens[n > N] = 1
        noc = 1 - cens
        if np.all(n <= N):
            break
    n = noc * n + cens * N

    cvs = []
    for i in range(nvar):
        M, S = med[:, i], esse[:, i]
        Y = (N * M).sum()
        if abs(Y) < 1e-10:
            cvs.append(0.0)
            continue
        V = sum(N[h] ** 2 * S[h] ** 2 / n[h] * (1 - n[h] / N[h])
                for h in range(nstr) if n[h] > 0)
        cvs.append(sqrt(V) / abs(Y))
    return ceil(n.sum()), n, cvs


def aggregate(atomic, assignment, K):
    """Pool the atomic strata into K aggregated strata."""
    at = atomic.copy()
    at['cluster'] = [assignment[i] for i in range(len(atomic))]
    rows = []
    for k in range(K):
        cl = at[at['cluster'] == k]
        if len(cl) == 0:
            continue
        Nt = cl['N'].sum()
        row = {'cluster': k, 'N': Nt, 'n_atomic': len(cl)}
        for mcol, scol in [('M1', 'S1'), ('M2', 'S2')]:
            Mp = (cl['N'] * cl[mcol]).sum() / Nt
            var = ((cl['N'] * (cl[mcol] - Mp) ** 2).sum()
                   + (cl['N'] * cl[scol] ** 2).sum()) / Nt
            row[mcol] = Mp
            row[scol] = sqrt(var)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. THE QUBO  (same objective as the D-Wave / Dirac-3 formulation)
# ----------------------------------------------------------------------
def weight_matrix(atomic):
    """
    Population-weighted distance between atomic strata, IDENTICAL to the matrix
    used by the working D-Wave QUBO:

        w_ij = euclidean_distance(feat_i, feat_j) * sqrt(N_i * N_j)

    on standardised (M1, M2, S1, S2) features.

    IMPORTANT - why distance and not similarity. The objective MINIMISES the
    weighted intra-cluster distance, i.e. w_ij is applied as a POSITIVE cost
    whenever i and j land in the same cluster. Putting everything into one
    cluster is then the WORST possible solution, not the best, so the
    formulation has no degenerate minimum.

    The opposite formulation, rewarding similar strata for sharing a cluster
    (a negative coefficient on same-cluster pairs), looks equivalent but is
    NOT: since all similarities are positive, its global minimum is the
    single-cluster collapse. That is the same defect that makes the Dirac-3
    integer encoding unusable, and it must be avoided here too.
    """
    feats = atomic[['M1', 'M2', 'S1', 'S2']].values.astype(float).copy()
    for c in range(feats.shape[1]):
        s = feats[:, c].std()
        feats[:, c] = (feats[:, c] - feats[:, c].mean()) / (s if s > 0 else 1.0)
    dist = cdist(feats, feats, 'euclidean')

    n = len(atomic)
    Nv = atomic['N'].values.astype(float)
    w = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        val = dist[i, j] * sqrt(Nv[i] * Nv[j])
        w[i, j] = w[j, i] = val
    return w


def build_model(atomic, K, penalty=None):
    """
    One-hot QUBO:  x[h,k] = 1  <=>  atomic stratum h assigned to cluster k.

      objective  : minimise  sum_{i<j} w_ij * sum_k x[i,k] x[j,k]
                   (cost of keeping distant strata in the same cluster)
      constraint : one_hot(x[h, :])  for every h

    Amplify handles the constraint natively, so we do not hand-tune a
    Lagrange multiplier the way we must for D-Wave / Dirac-3; penalty
    calibration is done by the SDK unless --penalty is given.
    """
    from amplify import VariableGenerator, Model, one_hot, sum as asum

    n = len(atomic)
    w = weight_matrix(atomic)

    gen = VariableGenerator()
    q = gen.array("Binary", shape=(n, K))

    # MINIMISE the weighted intra-cluster distance: a pair (i, j) sharing a
    # cluster costs w_ij. Same objective as the D-Wave / Dirac-3 QUBO.
    objective = asum(
        w[i, j] * asum(q[i, k] * q[j, k] for k in range(K))
        for i, j in combinations(range(n), 2)
        if w[i, j] > 1e-10
    )

    # One-hot constraint weight.
    # CRITICAL: Amplify gives a constraint weight of 1.0 by default, but our
    # objective coefficients reach max(w) ~ 160, so with weight 1 the solver
    # simply violates the constraint and every returned solution is infeasible
    # (the SDK then filters them out and hands back an empty result).
    # We therefore scale the penalty to the objective, exactly as the D-Wave
    # script does with lagrange = 3 * max_w.
    if penalty is None:
        penalty = 3.0 * float(w.max())

    constraints = penalty * asum(one_hot(q[h]) for h in range(n))

    model = Model(objective, constraints)
    return model, q, n * K, penalty


# ----------------------------------------------------------------------
# 4. CLIENTS
# ----------------------------------------------------------------------
def make_client(name, token, timeout_ms, url=''):
    """Return a configured Amplify client for the requested machine."""
    if name == 'ae':
        from amplify import FixstarsClient
        c = FixstarsClient()
        c.token = token
        c.parameters.timeout = timedelta(milliseconds=timeout_ms)
        return c, "Fixstars Amplify AE (GPU annealer)"

    if name == 'toshiba':
        from amplify import ToshibaSQBM2Client
        if not url:
            raise ValueError(
                "Toshiba SQBM+ needs its OWN endpoint URL: it is a separate "
                "subscription (e.g. via AWS Marketplace), not covered by the "
                "free Fixstars token. Pass it with --url https://<your-sqbm-host>")
        c = ToshibaSQBM2Client()
        c.url = url
        c.token = token
        c.solver = "Qubo"
        c.parameters.timeout = timedelta(milliseconds=timeout_ms)
        return c, "Toshiba SQBM+ (Simulated Bifurcation Machine)"

    if name == 'fujitsu':
        from amplify import FujitsuDA4Client
        c = FujitsuDA4Client()
        c.token = token
        c.parameters.time_limit_sec = max(1, timeout_ms // 1000)
        return c, "Fujitsu Digital Annealer (fully-connected CMOS)"

    if name == 'dwave':
        from amplify import DWaveSamplerClient
        c = DWaveSamplerClient()
        c.token = token
        c.solver = "Advantage_system4.1"
        c.parameters.num_reads = 1000
        return c, "D-Wave Advantage (real QPU)"

    if name == 'gurobi':
        from amplify import GurobiClient
        c = GurobiClient()
        c.parameters.time_limit = timedelta(milliseconds=timeout_ms)
        return c, "Gurobi (exact branch-and-bound)"

    raise ValueError(f"Unknown client '{name}'")


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Optimal stratification QUBO on Fixstars Amplify machines")
    ap.add_argument('--client', default='ae',
                    choices=['ae', 'toshiba', 'fujitsu', 'dwave', 'gurobi'],
                    help="Machine to solve on (default: ae = free Amplify AE)")
    ap.add_argument('--csv', default='swiss_municipalities.csv')
    ap.add_argument('--reg', type=int, default=None)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--cv', type=float, default=0.10)
    ap.add_argument('--y1', default='Surfacesbois')
    ap.add_argument('--y2', default='Airbat')
    ap.add_argument('--benchmark', type=int, default=129,
                    help="Classical reference to display (default: 129)")
    ap.add_argument('--timeout', type=int, default=3000,
                    help="Solver time limit in ms (default: 3000)")
    ap.add_argument('--penalty', type=float, default=None,
                    help="One-hot penalty weight (default: 3 x max objective coefficient, "
                         "as in the D-Wave script). Raise it if the solver returns "
                         "no feasible solution.")
    ap.add_argument('--token', default=os.environ.get('AMPLIFY_TOKEN', ''))
    ap.add_argument('--url', default='',
                    help="Endpoint URL (required for --client toshiba / fujitsu: "
                         "these are separate subscriptions with their own host)")
    ap.add_argument('--dry-run', action='store_true',
                    help="Build the model but do not solve (no token needed)")
    args = ap.parse_args()

    print("=" * 72)
    print(" QUANTUM STRATIFICATION - Fixstars Amplify")
    print(f" Machine: {args.client.upper()}")
    if args.dry_run:
        print(" DRY RUN (model built, not submitted)")
    print("=" * 72)

    # -- 1. data
    print("\n[1] Data preparation...")
    atomic, df = prepare_frame(args.csv, reg=args.reg, y1=args.y1, y2=args.y2)
    n_atomic = len(atomic)
    K = args.K
    print(f"    Frame: {len(df)} units, atomic strata: {n_atomic} "
          f"(singletons: {int((atomic['N'] == 1).sum())})")
    print(f"    K = {K}, CV target = {args.cv}, Y = ({args.y1}, {args.y2})")

    # -- 2. model
    print("\n[2] Building the one-hot QUBO...")
    model, q, n_vars, penalty_used = build_model(atomic, K, penalty=args.penalty)
    print(f"    Binary variables: {n_vars}  ({n_atomic} strata x {K} clusters)")
    print(f"    Interaction graph: dense (every stratum pair couples)")
    print(f"    One-hot constraints: {n_atomic}, penalty weight = {penalty_used:.1f}"
          f"{' (user-supplied)' if args.penalty else ' (= 3 x max objective coefficient)'}")

    if args.dry_run:
        print("\n[DRY RUN] Model is well-formed and ready to submit.")
        print("Re-run without --dry-run and with a valid --token to solve.")
        return

    if not args.token and args.client != 'gurobi':
        print("\nERROR: no token. Register free at https://amplify.fixstars.com,")
        print("then pass --token \"YOUR_TOKEN\" or set AMPLIFY_TOKEN.")
        return

    # -- 3. solve
    from amplify import solve
    client, label = make_client(args.client, args.token, args.timeout, url=args.url)
    print(f"\n[3] Solving on {label}...")
    print(f"    Time limit: {args.timeout} ms")

    t0 = time.time()
    result = solve(model, client)
    elapsed = time.time() - t0

    if len(result) == 0:
        print("    ERROR: no feasible solution returned.")
        print(f"    The one-hot penalty ({penalty_used:.1f}) may be too weak relative")
        print("    to the objective, so every sampled solution violates a constraint")
        print("    and gets filtered out. Retry with a larger --penalty (e.g. "
              f"{penalty_used * 3:.0f}) or a longer --timeout.")
        return

    best = result.best
    print(f"    Solved in {elapsed:.2f}s")
    print(f"    Solutions returned: {len(result)}")
    print(f"    Best objective (QUBO energy): {best.objective:.4f}")
    if hasattr(result, 'execution_time'):
        print(f"    Machine execution time: {result.execution_time}")

    # -- 4. decode
    print("\n[4] Decoding...")
    values = q.evaluate(best.values)          # shape (n_atomic, K), 0/1
    assignment = {}
    violations = 0
    for h in range(n_atomic):
        row = np.asarray(values[h]).astype(int)
        if row.sum() != 1:
            violations += 1
            assignment[h] = int(np.argmax(row)) if row.sum() > 0 else 0
        else:
            assignment[h] = int(np.argmax(row))
    print(f"    One-hot violations: {violations}"
          + (" (repaired)" if violations else " (constraints satisfied)"))

    used = sorted(set(assignment.values()))
    print(f"    Clusters actually used: {len(used)} of {K}")

    # -- 5. Bethel evaluation
    print("\n[5] Bethel evaluation...")
    agg = aggregate(atomic, assignment, K)
    n_sample, alloc, cvs = bethel(agg, cv_targets=[args.cv, args.cv])

    print(f"\n  +{'=' * 52}+")
    print(f"  |  RESULT - {label[:38]:<38}  |")
    print(f"  |  Sample size:        {n_sample:>6}                       |")
    print(f"  |  Aggregated strata:  {len(agg):>6}                       |")
    print(f"  |  CV(Y1): {cvs[0]:>7.4f}   CV(Y2): {cvs[1]:>7.4f}              |")
    print(f"  |  Variables:          {n_vars:>6}                       |")
    print(f"  |  Classical reference:{args.benchmark:>6}                       |")
    print(f"  +{'=' * 52}+")

    print(f"\n    {'#':>3} {'N':>6} {'Atom':>5} {'M(Y1)':>10} {'M(Y2)':>10} "
          f"{'S(Y1)':>10} {'S(Y2)':>10} {'n_h':>6}")
    print("  " + "-" * 68)
    for i, r in agg.iterrows():
        print(f"    {int(r['cluster']) + 1:>3} {int(r['N']):>6} {int(r['n_atomic']):>5} "
              f"{r['M1']:>10.1f} {r['M2']:>10.1f} {r['S1']:>10.1f} {r['S2']:>10.1f} "
              f"{int(alloc[i]):>6}")
    print(f"  {'TOT':>5} {int(agg['N'].sum()):>6} {n_atomic:>5} "
          f"{'':>10} {'':>10} {'':>10} {'':>10} {n_sample:>6}")

    # -- 6. save
    out = {
        'machine': label,
        'client': args.client,
        'n_atomic': n_atomic,
        'K': K,
        'cv_target': args.cv,
        'y1': args.y1, 'y2': args.y2,
        'n_variables': n_vars,
        'qubo_energy': float(best.objective),
        'sample_size': int(n_sample),
        'aggregated_strata': int(len(agg)),
        'cvs': [float(c) for c in cvs],
        'violations': violations,
        'benchmark': args.benchmark,
        'elapsed_s': round(elapsed, 3),
        'assignment': [assignment[i] for i in range(n_atomic)],
    }
    fname = f"result_amplify_{args.client}_K{K}.json"
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {fname}")

    # -- 7. reading of the result
    SA_REF = 162          # D-Wave QUBO solved by classical simulated annealing
    print("\n  How to read this:")

    if len(agg) == 1:
        print("    WARNING: the solution collapsed into a SINGLE cluster.")
        print("    That is a degenerate solution, not a useful stratification.")
        print("    It means the objective handed to the solver has its minimum at")
        print("    'everything in one cluster'. Check that the objective MINIMISES")
        print("    intra-cluster distance (positive w_ij on same-cluster pairs)")
        print("    rather than rewarding similarity (negative coefficient), which")
        print("    always collapses. The result below is not comparable.")
    elif n_sample <= args.benchmark:
        print(f"    n={n_sample} matches or beats the classical benchmark "
              f"({args.benchmark}).")
        print("    The QUBO surrogate is far better aligned with the Bethel")
        print("    objective than the earlier runs suggested.")
    elif n_sample < SA_REF - 5:
        print(f"    n={n_sample} is clearly BELOW the simulated-annealing value "
              f"({SA_REF}).")
        print("    Part of the gap previously blamed on the QUBO formulation was")
        print("    in fact our own solver being too weak: the surrogate ceiling")
        print("    is lower than we thought, and Section 5.2 needs revising.")
    elif n_sample <= SA_REF + 5:
        print(f"    n={n_sample} essentially matches the simulated-annealing value "
              f"({SA_REF}).")
        print("    A machine purpose-built for dense QUBOs does no better, which")
        print("    CONFIRMS that ~162 is the ceiling imposed by the surrogate")
        print("    objective itself, not by any particular solver or hardware.")
    else:
        print(f"    n={n_sample} is WORSE than classical simulated annealing "
              f"({SA_REF}) on the same QUBO.")
        print("    The solver did not reach as good a minimum. Try a longer")
        print("    --timeout, or a different --penalty for the one-hot constraint.")


if __name__ == '__main__':
    main()
