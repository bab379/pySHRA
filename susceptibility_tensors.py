import sympy as sp
from itertools import product
from pymatgen.symmetry.groups import PointGroup
import numpy as np
import os
import pickle

import sympy as sp
import pickle
import os

def get_reduced_tensor(N, name, point_group):
    filename = f"{name}_{N}_{point_group}.pkl"

    # Try to load cached tensor
    if os.path.exists(filename):
        print(f"\nLoading cached tensor of rank {N} for point group {point_group} from {filename}")
        print('Located in:', os.getcwd())
        with open(filename, 'rb') as f:
            T = pickle.load(f)
    else:
        print(f"\nComputing tensor and saving to {filename}")
        T = reduce_tensor(N, name, point_group)
        with open(filename, 'wb') as f:
            pickle.dump(T, f)
    return T

def getX2TensorLab(point_group,rotation_matrix):
    name = 'chi'
    N = 3
    X2 = get_reduced_tensor(N, name, point_group)
    print_slices(X2)
    free_syms = list({s for e in X2 for s in e.free_symbols})
    print('Nonvanishing elements', free_syms)
    Q = rotation_matrix
    X2_lab = np.einsum('im,jn,ko,mno->ijk',Q,Q,Q,X2)
    return X2_lab, X2

def getX3TensorLab(point_group,rotation_matrix):
    name = 'chi'
    N = 4
    X3 = get_reduced_tensor(N, name, point_group)
    print_slices(X3)
    free_syms = list({s for e in X3 for s in e.free_symbols})
    print('Nonvanishing elements', free_syms)
    Q = rotation_matrix
    X3_lab = np.einsum('im,jn,ko,lp,mnop->ijkl',Q,Q,Q,Q,X3)
    return X3_lab, X3

def symbolic_tensor(rank,name):
    """Return a rank-'rank' MutableDenseNDimArray with each axis size 'dim'."""
    dim = 3
    idx_ranges = [range(1, dim+1)] * rank
    symbols = [sp.Symbol(f"{name}_" + "".join(map(str, idx)))
               for idx in product(*idx_ranges)]
    return sp.MutableDenseNDimArray(symbols, [dim]*rank)


def print_slices(T):
    print('\nPrinting tensor...')
    for i in range(T.shape[0]):
        print(f"Slice {i+1}:")
        slice_2d = T[i, :, :].tolist()
        # use pretty() on each element and align columns
        pretty_rows = [[sp.pretty(e, use_unicode=True) for e in row] for row in slice_2d]
        col_widths = [max(len(s) for s in col) for col in zip(*pretty_rows)]
        for row in pretty_rows:
            print("  ".join(s.ljust(w) for s, w in zip(row, col_widths)))
        print()

def symmetrize_jk(T):
    N, r = T.shape[0], T.rank()
    for idx in product(*(range(N) for _ in range(r))):
        i, j, k, *rest = idx
        if k < j:
            canon = (i, k, j, *rest)
            T[idx] = T[canon]
    return T

def get_symmetry_ops(pg_label):
    """Return a list of 3x3 SymPy matrices for the point group, including 'inf m'."""
    if pg_label == 'infm':
        phi = sp.Symbol('phi')
        Rz = sp.Matrix([[sp.cos(phi), -sp.sin(phi), 0],
                     [sp.sin(phi),  sp.cos(phi), 0],
                     [0, 0, 1]])
        sigma_xz = sp.Matrix([[1,0,0],[0,-1,0],[0,0,1]])
        return [sigma_xz,Rz], phi
    else:
        return [sp.Matrix(op.rotation_matrix) for op in PointGroup(pg_label).symmetry_ops], None

def apply_symmetry_constraint(T, R):
    """Return equations enforcing R·T = T for any-rank tensor T (shape may be arbitrary)."""
    shape = T.shape
    rank = len(shape)
    ranges = [range(s) for s in shape]
    eqs = []
    for idx in product(*ranges):
        transformed = 0
        for a in product(*ranges):
            term = T[a]
            for n in range(rank):
                term *= R[idx[n], a[n]]
            transformed += term
        eqs.append(sp.Eq(T[idx], sp.simplify(transformed)))
    return eqs

def enforce_for_all_phi(eqs, phi):
    new_eqs = []
    for eq in eqs:
        diff = sp.simplify(eq.lhs - eq.rhs)
        diff = sp.expand_trig(diff)
        # Express trig functions as powers of sin(phi) and cos(phi)
        diff_poly = sp.Poly(sp.together(diff), sp.sin(phi), sp.cos(phi))
        # Each coefficient must vanish separately
        new_eqs += [sp.Eq(c, 0) for c in diff_poly.coeffs()]
    return new_eqs

def canonicalize_solutions(solutions):
    new_solutions = {}
    for sol in solutions:
        for k, v in sol.items():
            # Case 1: pure renaming (one symbol)
            if isinstance(v, sp.Symbol):
                # always rename the larger one to the smaller one
                if v.name > k.name:
                    new_solutions[v] = k
                else:
                    new_solutions[k] = v

            # Case 2: expression (linear combo, etc.)
            else:
                # recursively replace any symbols in v using canonical rules
                v_canon = v
                for s in v.free_symbols:
                    smaller = min(k, s, key=lambda x: x.name)
                    larger  = max(k, s, key=lambda x: x.name)
                    if smaller != larger:
                        v_canon = v_canon.subs(larger, smaller)
                new_solutions[k] = sp.simplify(v_canon)

    return [new_solutions]


def reduce_tensor(N,name,pg_label):
    T = symbolic_tensor(N,name)
    T = symmetrize_jk(T)
    ops,phi = get_symmetry_ops(pg_label)
    print('Applying symmetry operations to tensor of rank: ', str(N))
    for i in range(0,len(ops)):
        op = ops[i]
        eqs = apply_symmetry_constraint(T,op)
        eqs = [e for e in eqs if e != True]
        if phi != None:
            eqs = enforce_for_all_phi(eqs, phi)
        symbols = list({s for e in eqs for s in e.free_symbols if s.name.startswith(name + '_')})
        solutions = sp.solve(eqs, symbols, dict=True)
        solutions = canonicalize_solutions(solutions)
        if len(solutions) > 1:
            print(solutions)
            input('multiple solutions found! Please modify code if needed.')
        elif len(solutions) == 1:
            solutions = solutions[0]
            [T.__setitem__(idx, T[idx].subs(solutions)) for idx in product(*(range(s) for s in T.shape))]
    print('Arrived at solution. Checking invariance...')
    successBool = check_invariance(T,ops)
    if successBool:
        print('Success!')
    else:
        print('Failure!')
    return T

def check_invariance(T,ops):
    success = True
    for i in range(0,len(ops)):
        op = ops[i]
        print('op['+str(i)+']: ',op)
        eqs = apply_symmetry_constraint(T,op)
        eqs = [e for e in eqs if e != True]
        if len(eqs) == 0:
            print('Invariance under op[' + str(i) + ']: True.')
        else:
            print('Invariance under op[' + str(i) + ']: False.')
            success = False
    return success





