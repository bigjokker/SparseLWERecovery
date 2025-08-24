import numpy as np

# Define centered function
def ctr(x, q=2**32):
    x_mod = x % q
    return np.where(x_mod > q/2, x_mod - q, x_mod)

n, m, q, B, w, m0 = 128, 128, 2**32, 3, 11, 8
# Mock placeholders for missing variables (replace with actual in real use)
A_sparse = np.random.choice([-1, 0, 1], size=(m, n), p=[0.1, 0.8, 0.1])  # Mock sparse matrix
t_sparse = np.random.randint(0, q, m)  # Mock t
A_dense_sieve = np.random.randint(0, q, (m0, n))  # 8 dense rows
t_dense_sieve = np.random.randint(0, q, m0)  # corresponding t' slice
A_prime = np.random.randint(0, q, (m, n))  # Full A' for final ML
t_prime = np.random.randint(0, q, m)  # Full t' for final ML
var_e = 4.0  # Variance for ML

# Mock branch_bound_enum (placeholder for actual enumeration; in real, implement branch-and-bound)
def branch_bound_enum(left_cols, prune_func):
    # Mock: yield a dummy s_left
    yield np.random.randint(-3, 4, len(left_cols))

# Mock partial_norm_prune (placeholder)
def partial_norm_prune(*args):
    return True

# Mock match_right_sparse (placeholder; in real, implement matching)
def match_right_sparse(key_sparse_l, t_sparse, right_partial):
    # Mock: yield a dummy key and list
    yield tuple(np.zeros(m)), [np.random.randint(-3, 4, 64)]

# Mock right_partial dictionary (placeholder)
right_partial = {tuple(np.zeros(m)): {'dense_partial': np.zeros(m0)}}

# Precompute max remain: 6 per unassigned coord (worst |d_i|=6)
def dense_budget(remaining_cols): return 6 * remaining_cols  # admissible prune: cannot remove true path

# For sparse: max remain 6J, J≤remaining w fraction
def sparse_budget(remaining_cols, row_w= w): return 6 * min(row_w, remaining_cols)  # conservative

# Enum left: branch-and-bound over {-3..3}^64, prune partial norms
left_partial = {}  # (sparse_partial tuple, dense_partial tuple) -> s_left list
for s_left in branch_bound_enum(range(64), partial_norm_prune):  # impl branch-bound with prune
    partial_sparse = ctr(A_sparse[:, :64] @ s_left)
    partial_dense = ctr(t_dense_sieve - A_dense_sieve[:, :64] @ s_left)
    if all(abs(partial_sparse[j]) <= B + sparse_budget(64, w) for j in range(m)) and \
       all(abs(partial_dense[j]) <= B + dense_budget(64) for j in range(m0)):
        key_sparse = tuple(partial_sparse)
        key_dense = tuple(partial_dense)
        left_partial.setdefault((key_sparse, key_dense), []).append(s_left)

# Symmetric for right_partial (mock)
right_partial = {tuple(np.zeros(m)): {'dense_partial': np.zeros(m0)}}

# Join: match t_sparse - left_sparse - right_sparse small, and dense partial
survivors = []
for (key_sparse_l, key_dense_l), s_left_list in left_partial.items():
    for key_sparse_r, s_right_list in match_right_sparse(key_sparse_l, t_sparse, right_partial):  # find right keys where ctr(t_sparse - l - r) <=B coord-wise
        partial_dense_r = right_partial[key_sparse_r]['dense_partial']  # from match
        r_partial = ctr(t_dense_sieve - (key_dense_l + partial_dense_r))  # tuple add
        if all(abs(r_partial[j]) <= B + dense_budget(0) for j in range(m0)):  # remaining=0 after join
            for s_left in s_left_list:  # nested to handle unequal list lengths
                for s_right in s_right_list:
                    s_cand = np.concatenate([s_left, s_right])
                    survivors.append(s_cand)  # ~1 expected

# Final ML on full dense for survivors
for s_cand in survivors:
    r_full = ctr(t_prime - A_prime @ s_cand)
    score = - np.dot(r_full, r_full) / (2 * var_e)  # Gaussian ML; pick max (unique w.o.p.)