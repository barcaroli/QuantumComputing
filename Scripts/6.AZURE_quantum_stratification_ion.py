"""
========================================================================
 QUANTUM STRATIFICATION - Trapped-ion QAOA (IonQ / Quantinuum)
 via Azure Quantum
========================================================================

THE QUESTION THIS ANSWERS
-------------------------
The IBM run failed by NOISE, not by algorithm. Its circuit was inflated to
~5,300 two-qubit gates and depth ~4,900, because a DENSE QUBO must be routed
onto a SPARSE heavy-hex lattice: every interaction between non-adjacent
qubits costs SWAP gates.

Trapped-ion processors have ALL-TO-ALL connectivity: any pair of ions can
interact directly, through the collective vibrational modes of the chain.
There is no routing, hence no SWAP overhead. Transpiling our own circuit
against both topologies gives:

                        CX gates    depth    2q fidelity   signal left
  K=2 heavy-hex (IBM)       834       268       ~99%           0.02%
  K=2 all-to-all (ion)      380       114       ~99.9%         68%
  K=4 heavy-hex (IBM)      5911      1583       ~99%           ~0%
  K=4 all-to-all (ion)     1900       355       ~99.9%         15%

So on a trapped-ion machine the K=2 circuit should actually WORK. That is
the experiment: same problem, same algorithm, different connectivity.

WHY K=2 AND NOT K=4
-------------------
K=4 needs 40 qubits. The trapped-ion machines reachable through Azure top
out below that: Quantinuum H1-1 has 20 qubits, H2-1 has 32, IonQ Forte 36.
K=2 needs exactly 20 qubits and fits H1-1. It is a coarser stratification,
but it isolates the variable we care about: connectivity.

TARGETS
-------
  FREE (no credit consumed):
    quantinuum.sim.h1-1sc   Syntax checker. Validates the circuit compiles
                            for H1-1. Returns all zeros - use it to check,
                            not to get results.
    ionq.simulator          Ideal (noiseless) IonQ simulator.

  PAID (consumes credit):
    quantinuum.sim.h1-1e    H1-1 EMULATOR with a realistic noise model of
                            the real device. The best value for money: it
                            predicts hardware behaviour without the queue.
    quantinuum.qpu.h1-1     Real H1-1 hardware (20 qubits, all-to-all).
    ionq.qpu.forte-1        Real IonQ Forte (36 qubits, all-to-all).

SETUP
-----
  1. Azure subscription (a free account works).
  2. Create an Azure Quantum workspace:
       https://learn.microsoft.com/azure/quantum/how-to-create-workspace
     Enable the IonQ and/or Quantinuum providers. Each participating
     provider grants USD 500 of free credit automatically.
  3. pip install "azure-quantum[qiskit]"
  4. Copy the workspace Resource ID from the Azure portal.

USAGE
-----
  # 1. Always start here: costs nothing, shows the circuit size
  python quantum_stratification_ion.py --dry-run --csv swiss_municipalities.csv \
      --K 2 --cv 0.05 --y1 Airbat --y2 Surfacesbois --benchmark 129

  # 2. Estimate the price of a single circuit BEFORE running it
  python quantum_stratification_ion.py --estimate-cost --target quantinuum.qpu.h1-1 \
      --resource-id "/subscriptions/.../Microsoft.Quantum/Workspaces/..." ...

  # 3. Free syntax check on the real compiler
  python quantum_stratification_ion.py --target quantinuum.sim.h1-1sc ...

  # 4. Noise-model emulator (recommended before touching hardware)
  python quantum_stratification_ion.py --target quantinuum.sim.h1-1e --maxiter 8 ...

  # 5. Real hardware
  python quantum_stratification_ion.py --target quantinuum.qpu.h1-1 --maxiter 5 ...
========================================================================
"""

import argparse
import json
import os
import time
from itertools import combinations
from math import ceil, sqrt

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------
# FRAME + BETHEL  (identical to every other script in this study)
# ----------------------------------------------------------------------
def var_bin(x, nbins):
    xa = x.values.astype(float).reshape(-1, 1)
    np.random.seed(42)
    cent, _ = kmeans(xa, nbins)
    lab, _ = vq(xa, np.sort(cent, axis=0))
    return lab + 1


def prepare_frame(csv_path, reg=None, y1='Surfacesbois', y2='Airbat'):
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
        raise KeyError("CSV needs POPTOT/HApoly or POPTOT.cat/HApoly.cat")
    for col in (y1, y2):
        if col not in df.columns:
            raise KeyError(f"Target '{col}' not in CSV")
    df['key'] = df['X1'].astype(str) + '*' + df['X2'].astype(str)
    rows = []
    for k, g in df.groupby('key'):
        N = len(g)
        rows.append({'STRATO': k, 'N': N,
                     'M1': g[y1].mean(), 'M2': g[y2].mean(),
                     'S1': g[y1].std(ddof=1) if N > 1 else 0.0,
                     'S2': g[y2].std(ddof=1) if N > 1 else 0.0})
    return pd.DataFrame(rows).reset_index(drop=True), df


def bethel(agg, cv_targets=(0.1, 0.1), minnumstr=2, maxiter=200,
           maxiter1=25, epsilon=1e-11):
    nstr, nvar = len(agg), len(cv_targets)
    N = agg['N'].values.astype(float)
    cens = np.zeros(nstr); cens[N < minnumstr] = 1; noc = 1 - cens
    med = agg[[f'M{i+1}' for i in range(nvar)]].values.astype(float)
    esse = agg[[f'S{i+1}' for i in range(nvar)]].values.astype(float)
    cv = np.array(cv_targets, dtype=float)
    Nc, nocc = N.reshape(-1, 1), noc.reshape(-1, 1)
    numA = (Nc ** 2) * (esse ** 2) * nocc
    dA1 = (np.sum(Nc * med * cv.reshape(1, -1), axis=0)) ** 2
    dA2 = np.sum(Nc * (esse ** 2) * nocc, axis=0)
    a = numA / (dA1 + dA2 + epsilon)
    alfa = np.ones(nvar) / nvar
    x = np.full(nstr, 1e-6)
    for _ in range(maxiter):
        d1 = np.sqrt(np.sum(a * alfa.reshape(1, -1), axis=1))
        x = 1.0 / (d1 * np.sum(d1) + epsilon)
        axs = np.sum(a * x.reshape(-1, 1), axis=0)
        tot = max(np.sum(alfa * axs), epsilon)
        an = alfa * axs / tot
        if np.max(np.abs(an - alfa)) < epsilon:
            break
        alfa = an
    n = np.ceil(1.0 / x)
    for _ in range(maxiter1):
        n = np.minimum(n, N); n = np.maximum(n, minnumstr)
        cens[n > N] = 1; noc = 1 - cens
        if np.all(n <= N):
            break
    n = noc * n + cens * N
    cvs = []
    for i in range(nvar):
        M, S = med[:, i], esse[:, i]
        Y = (N * M).sum()
        if abs(Y) < 1e-10:
            cvs.append(0.0); continue
        V = sum(N[h] ** 2 * S[h] ** 2 / n[h] * (1 - n[h] / N[h])
                for h in range(nstr) if n[h] > 0)
        cvs.append(sqrt(V) / abs(Y))
    return ceil(n.sum()), n, cvs


def aggregate(atomic, assignment, K):
    at = atomic.copy()
    at['c'] = [assignment[i] for i in range(len(atomic))]
    rows = []
    for k in range(K):
        cl = at[at['c'] == k]
        if len(cl) == 0:
            continue
        Nt = cl['N'].sum()
        row = {'cluster': k, 'N': Nt, 'n_atomic': len(cl)}
        for mc, sc in [('M1', 'S1'), ('M2', 'S2')]:
            Mp = (cl['N'] * cl[mc]).sum() / Nt
            var = ((cl['N'] * (cl[mc] - Mp) ** 2).sum()
                   + (cl['N'] * cl[sc] ** 2).sum()) / Nt
            row[mc] = Mp
            row[sc] = sqrt(var)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# ISING HAMILTONIAN  (K=2: one qubit per stratum)
# ----------------------------------------------------------------------
def build_ising(atomic, K=2):
    """
    K=2: z_i = +1 -> cluster 0, z_i = -1 -> cluster 1.
    Two strata in the same cluster cost w_ij; the indicator of 'same cluster'
    is (1 + z_i z_j)/2, so the cost operator is sum_ij w_ij/2 * Z_i Z_j
    (up to a constant). Weights are the same population-weighted distances
    used by every other formulation in this study.
    """
    if K != 2:
        raise ValueError("This script implements K=2 (one qubit per stratum). "
                         "K=4 would need 40 qubits, beyond the trapped-ion "
                         "machines reachable via Azure (H1-1: 20, H2-1: 32).")
    n = len(atomic)
    feats = atomic[['M1', 'M2', 'S1', 'S2']].values.astype(float).copy()
    for c in range(feats.shape[1]):
        s = feats[:, c].std()
        feats[:, c] = (feats[:, c] - feats[:, c].mean()) / (s if s > 0 else 1.0)
    dist = cdist(feats, feats, 'euclidean')
    Nv = atomic['N'].values.astype(float)

    terms, wmax = [], 0.0
    for i, j in combinations(range(n), 2):
        w = dist[i, j] * sqrt(Nv[i] * Nv[j])
        wmax = max(wmax, w)
        if w > 1e-10:
            label = ['I'] * n
            label[i] = 'Z'
            label[j] = 'Z'
            terms.append((''.join(reversed(label)), w / 2.0))
    return terms, n, wmax


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Trapped-ion QAOA for optimal stratification (Azure Quantum)")
    ap.add_argument('--csv', default='swiss_municipalities.csv')
    ap.add_argument('--reg', type=int, default=None)
    ap.add_argument('--K', type=int, default=2, choices=[2],
                    help="Only K=2 (20 qubits) fits the Azure trapped-ion machines")
    ap.add_argument('--cv', type=float, default=0.05)
    ap.add_argument('--y1', default='Airbat')
    ap.add_argument('--y2', default='Surfacesbois')
    ap.add_argument('--benchmark', type=int, default=129)
    ap.add_argument('--target', default='quantinuum.sim.h1-1sc',
                    help="Azure target. FREE: quantinuum.sim.h1-1sc, ionq.simulator. "
                         "PAID: quantinuum.sim.h1-1e (noise emulator), "
                         "quantinuum.qpu.h1-1, ionq.qpu.forte-1")
    ap.add_argument('--resource-id', default=os.environ.get('AZURE_QUANTUM_RESOURCE_ID', ''),
                    help="Azure Quantum workspace Resource ID")
    ap.add_argument('--location', default=os.environ.get('AZURE_QUANTUM_LOCATION', 'westus'))
    ap.add_argument('--p', type=int, default=1, help="QAOA layers (default 1)")
    ap.add_argument('--shots', type=int, default=200)
    ap.add_argument('--maxiter', type=int, default=5,
                    help="COBYLA iterations. EACH ONE IS A PAID JOB.")
    ap.add_argument('--dry-run', action='store_true',
                    help="Build and transpile locally; submit nothing. Free.")
    ap.add_argument('--estimate-cost', action='store_true',
                    help="Ask Azure the price of ONE circuit, then stop.")
    args = ap.parse_args()

    print("=" * 72)
    print(" QUANTUM STRATIFICATION - Trapped-ion QAOA (Azure Quantum)")
    print(f" Target: {args.target}")
    print("=" * 72)

    # -- data
    print("\n[1] Data preparation...")
    atomic, df = prepare_frame(args.csv, reg=args.reg, y1=args.y1, y2=args.y2)
    n_atomic = len(atomic)
    print(f"    Frame: {len(df)} units, atomic strata: {n_atomic}")
    print(f"    K = {args.K}, CV target = {args.cv}, Y = ({args.y1}, {args.y2})")

    # -- Hamiltonian
    print("\n[2] Building the Ising cost Hamiltonian...")
    terms, n_qubits, wmax = build_ising(atomic, K=args.K)
    print(f"    Qubits: {n_qubits} (one per atomic stratum)")
    print(f"    ZZ terms: {len(terms)} (dense: every stratum pair couples)")

    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    H = SparsePauliOp.from_list(terms)
    ansatz = QAOAAnsatz(cost_operator=H, reps=args.p)
    print(f"    QAOA layers p = {args.p}, variational parameters: {ansatz.num_parameters}")

    # -- the free, decisive measurement: all-to-all vs heavy-hex
    print("\n[3] Connectivity comparison (computed locally, costs nothing)...")
    probe = ansatz.assign_parameters(np.random.rand(ansatz.num_parameters))
    t_all = transpile(probe, coupling_map=CouplingMap.from_full(n_qubits),
                      basis_gates=['rz', 'ry', 'rx', 'cx'],
                      optimization_level=1, seed_transpiler=1)
    t_hex = transpile(probe, coupling_map=CouplingMap.from_heavy_hex(5),
                      basis_gates=['rz', 'ry', 'rx', 'cx'],
                      optimization_level=1, seed_transpiler=1)
    cx_all = t_all.count_ops().get('cx', 0)
    cx_hex = t_hex.count_ops().get('cx', 0)
    print(f"    all-to-all (trapped ion): {cx_all:>5} CX, depth {t_all.depth():>5}")
    print(f"    heavy-hex  (IBM)        : {cx_hex:>5} CX, depth {t_hex.depth():>5}")
    print(f"    SWAP routing costs {cx_hex / max(cx_all, 1):.1f}x more gates on IBM")
    for name, fid in [("trapped ion (99.9%)", 0.999), ("IBM (99%)", 0.99)]:
        cx = cx_all if 'ion' in name else cx_hex
        print(f"    Expected signal on {name}: {fid ** cx:.1%}")

    if args.dry_run:
        print("\n[DRY RUN] Nothing submitted. This is the free part of the experiment.")
        return

    if not args.resource_id:
        print("\nERROR: no --resource-id. Create an Azure Quantum workspace and copy")
        print("its Resource ID from the Azure portal, or set AZURE_QUANTUM_RESOURCE_ID.")
        return

    # -- connect
    print(f"\n[4] Connecting to Azure Quantum ({args.location})...")
    from azure.quantum import Workspace
    from azure.quantum.qiskit import AzureQuantumProvider
    ws = Workspace(resource_id=args.resource_id, location=args.location)
    provider = AzureQuantumProvider(ws)
    print("    Available targets:")
    for b in provider.backends():
        print(f"      - {b.name()}")
    backend = provider.get_backend(args.target)

    # -- cost estimate
    isa = transpile(probe, backend=backend, optimization_level=1)
    if hasattr(backend, 'estimate_cost'):
        try:
            cost = backend.estimate_cost(isa, shots=args.shots)
            print(f"\n[5] COST ESTIMATE for ONE circuit at {args.shots} shots:")
            print(f"    {cost.estimated_total} {cost.currency_code}")
            total = args.maxiter + 1
            print(f"    With --maxiter {args.maxiter} you will submit {total} jobs.")
            print(f"    Rough total: {cost.estimated_total * total} {cost.currency_code}")
        except Exception as e:
            print(f"\n[5] Cost estimate unavailable: {e}")
    if args.estimate_cost:
        print("\n[--estimate-cost] Stopping here. Nothing was submitted.")
        return

    # -- QAOA loop
    print(f"\n[6] QAOA on {args.target}...")
    print(f"    WARNING: each COBYLA iteration is a separate PAID job.")
    from qiskit.quantum_info import Statevector

    history = []

    def energy(params):
        qc = ansatz.assign_parameters(params)
        qc.measure_all()
        isa_qc = transpile(qc, backend=backend, optimization_level=1)
        job = backend.run(isa_qc, shots=args.shots)
        counts = job.result().get_counts()
        tot = sum(counts.values())
        e = 0.0
        for bits, cnt in counts.items():
            b = bits.replace(' ', '')[::-1]           # qubit 0 first
            z = np.array([1 if b[i] == '0' else -1 for i in range(n_qubits)])
            val = sum(coef.real * z[i] * z[j]
                      for (lab, coef), (i, j) in
                      zip(terms, combinations(range(n_qubits), 2)))
            e += val * cnt / tot
        history.append(e)
        print(f"      iter {len(history)}: energy = {e:.4f}")
        return e

    x0 = np.random.default_rng(0).uniform(0, np.pi, ansatz.num_parameters)
    t0 = time.time()
    res = minimize(energy, x0, method='COBYLA',
                   options={'maxiter': args.maxiter, 'disp': False})
    elapsed = time.time() - t0
    print(f"    Optimisation time: {elapsed:.1f}s over {len(history)} jobs")

    # -- final sampling
    print("\n[7] Final sampling of the optimised circuit...")
    qc = ansatz.assign_parameters(res.x)
    qc.measure_all()
    isa_qc = transpile(qc, backend=backend, optimization_level=1)
    counts = backend.run(isa_qc, shots=args.shots).result().get_counts()
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print("    Top bitstrings:")
    for b, ct in top:
        print(f"      {b}: {ct}")
    spread = len(counts) / max(sum(counts.values()), 1)
    print(f"    Distinct outcomes / shots = {spread:.2f} "
          f"({'FLAT -> noise dominated' if spread > 0.7 else 'peaked -> signal present'})")

    best = top[0][0].replace(' ', '')[::-1]
    assignment = {i: (0 if best[i] == '0' else 1) for i in range(n_qubits)}

    # -- Bethel
    print("\n[8] Bethel evaluation...")
    agg = aggregate(atomic, assignment, args.K)
    n_sample, alloc, cvs = bethel(agg, cv_targets=[args.cv, args.cv])
    print(f"\n  +{'=' * 52}+")
    print(f"  |  RESULT - {args.target:<38}  |")
    print(f"  |  Sample size:        {n_sample:>6}                       |")
    print(f"  |  Aggregated strata:  {len(agg):>6}                       |")
    print(f"  |  CV(Y1): {cvs[0]:>7.4f}   CV(Y2): {cvs[1]:>7.4f}              |")
    print(f"  |  Qubits:             {n_qubits:>6}                       |")
    print(f"  |  Classical reference:{args.benchmark:>6}                       |")
    print(f"  +{'=' * 52}+")

    json.dump({'target': args.target, 'n_qubits': n_qubits, 'K': args.K,
               'cx_all_to_all': cx_all, 'cx_heavy_hex': cx_hex,
               'sample_size': int(n_sample), 'clusters': int(len(agg)),
               'cvs': [float(x) for x in cvs], 'energy_history': history,
               'assignment': [assignment[i] for i in range(n_qubits)]},
              open(f"result_ion_{args.target.replace('.', '_')}.json", 'w'), indent=2)
    print(f"\n  Saved to result_ion_{args.target.replace('.', '_')}.json")


if __name__ == '__main__':
    main()
