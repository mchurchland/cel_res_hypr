"""Directed graph controls with explicit NumPy random generators.

Rows are sources and columns are targets. Preserve NetworkX edge ordering: it
affects seeded edge matching and therefore historical graph realizations.
"""
import networkx as nx
import numpy as np

def _match_edge_count(mask: np.ndarray, target_m: int, rng: np.random.Generator) -> np.ndarray:
    """
    Given a boolean mask (no self-loops), randomly add/remove edges to match target_m nnz.
    Returns a boolean mask with exactly target_m ones (and zero diagonal).

    Random edge add/drop to match density (Newman, 2010, Networks: An Introduction).
    Uses NetworkX DiGraph utilities; see https://github.com/networkx/networkx/blob/main/networkx/classes/digraph.py
    """
    G = nx.from_numpy_array(mask.astype(bool), create_using=nx.DiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))
    current = G.number_of_edges()
    nodes = list(G.nodes())

    if current > target_m:
        edges = list(G.edges())
        drop_idx = rng.choice(len(edges), size=current - target_m, replace=False)
        G.remove_edges_from([edges[i] for i in drop_idx])
    elif current < target_m:
        candidates = [(u, v) for u in nodes for v in nodes if u != v and not G.has_edge(u, v)]
        need = target_m - current
        if need > 0 and candidates:
            add_idx = rng.choice(len(candidates), size=min(need, len(candidates)), replace=False)
            G.add_edges_from([candidates[i] for i in add_idx])

    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.to_numpy_array(G, dtype=np.float32)# pyright: ignore[reportArgumentType]


def er_adjacency(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """
    Directed Erdos-Renyi graph with independent directions (Erdos & Renyi, 1959, Publ. Math. Debrecen 6:290-297).
    Uses NetworkX gnp_random_graph with directed=True; see https://github.com/networkx/networkx/blob/main/networkx/generators/random_graphs.py#L310
    """
    G = nx.gnp_random_graph(n, p, seed=rng, directed=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.to_numpy_array(G, dtype=np.float32) # pyright: ignore[reportArgumentType]


def ws_directed_adjacency_exact(
    N: int,
    nnz_target: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Directed Watts-Strogatz-like adjacency with exactly nnz_target edges.

    Construction:
    1. Start from a directed ring lattice.
       - Every node gets k_base local outgoing edges.
       - r nodes get one extra local outgoing edge.
    2. Rewire each outgoing edge with probability p, preserving each row's out-degree.
    3. No self-loops, no duplicate edges.

    Returns:
        A: [N, N] float32 adjacency mask with exactly nnz_target ones.
    """
    if nnz_target < 0 or nnz_target > N * (N - 1):
        raise ValueError("nnz_target must be in [0, N*(N-1)]")

    k_base = nnz_target // N # 3018 // 299 = 10
    r = nnz_target % N ## 3108 % 299 = 118

    if k_base >= N:
        raise ValueError("nnz_target too large: would require self-loops or duplicate edges")

    # Decide which rows get one extra outgoing edge
    row_degrees = np.full(N, k_base, dtype=np.int32) ##array of size n filled with k_base = 10, then we select r many to add 1 to!
    if r > 0:
        extra_rows = rng.choice(N, size=r, replace=False)
        row_degrees[extra_rows] += 1

    A = np.zeros((N, N), dtype=np.float32)

    # Step 1: directed ring lattice N-r nodes with out deg 10, r nodes with out deg 11
    for i in range(N):
        kout = row_degrees[i]
        for d in range(1, kout + 1):
            j = (i + d) % N
            A[i, j] = 1.0

    # Step 2: rewire while preserving row out-degree
    for i in range(N):
        current_targets = np.where(A[i] > 0)[0].tolist()

        for j in current_targets:
            if rng.random() < p:
                # remove old edge
                A[i, j] = 0.0

                # choose a new target not already connected and not self
                forbidden = set(np.where(A[i] > 0)[0])
                forbidden.add(i)

                candidates = [v for v in range(N) if v not in forbidden]
                if not candidates:
                    # restore original edge if no valid rewiring target exists
                    A[i, j] = 1.0
                    continue

                new_j = rng.choice(candidates)
                A[i, new_j] = 1.0

    # Safety checks
    np.fill_diagonal(A, 0.0)
    assert int(A.sum()) == nnz_target, (A.sum(), nnz_target)

    return A


def degree_matched_shuffle_directed(
    A: np.ndarray,
    tries: int,
    rng: np.random.Generator | None = None,
    return_stats: bool = False,
):
    """
    Degree-preserving double-edge swap randomization for directed graphs
    (Milo et al., 2002, Science 298:824-827).

    Each successful swap consumes two entries from one random permutation of
    the edge list, so ``floor(|E| / 2)`` is this implementation's one-pass
    target. ``tries`` caps cumulative failed pair proposals, not total swaps.
    Set ``return_stats=True`` to return ``(shuffled_matrix, diagnostics)``.
    """
    def can_swap(a,b,c,d):
        if (len({a,b,c,d}) < 4) or ( A[a,d] or A[c,b]): ## makes sure all the values are unique, if the connection between a and d already exists or c and b
            return False
        else:
            return True
    def edge_swap(a: int, b: int, c: int, d: int):
        edge1_weight = A[a,b].copy()
        edge2_weight = A[c,d].copy()
        A[a,b] = 0  # remove edge 1
        A[c,d] = 0  # remove edge 2
        A[a,d] = edge1_weight  # add edge 1 to new nodes
        A[c,b] = edge2_weight  # add edge 2 to new nodes
    if rng is None:
        raise ValueError("Need to pass in a random number generator")
    if tries < 0:
        raise ValueError("tries must be non-negative")

    A = A.copy() ## make a copy of A
    np.fill_diagonal(A, 0)
    A_orig = A.copy()
    edges = np.argwhere(A!=0)
    m = edges.shape[0]
    if m < 2:
        raise ValueError(f"Not enough edges to perform randomization. Found {m} edges. make sure the matrix is correct")

    retries = 0
    successful_swaps = 0
    
    idx = rng.permutation(m)  
    #while not(idx.isempty) and retries < max_retries:
    i=0
    
    while i + 1 < len(idx) and retries < tries:
        a, b = edges[idx[i]] ## edge 1
        c, d = edges[idx[i+1]] ## edge 2
        if can_swap(a,b,c,d):
            edge_swap(a,b,c,d) ## swap the edges
            edges[idx[i]] = [a,d]
            edges[idx[i+1]] = [c,b]
            i+=2
            successful_swaps += 1
            
        else:
            retries += 1
            rem = idx[i:]
            rng.shuffle(rem) ## shuffle the indices
            idx[i:] = rem ## replace the indices with the shuffled one
    A = A.astype(np.float32)

    if return_stats:
        original_edges = A_orig != 0
        shuffled_edges = A != 0
        max_swaps = m // 2
        retained_fraction = (
            np.count_nonzero(original_edges & shuffled_edges)
            / np.count_nonzero(original_edges)
        )
        reached_max_swaps = successful_swaps == max_swaps
        hit_failed_attempt_limit = retries >= tries

        stats = {
            "edges": m,
            "max_swaps": max_swaps,
            "successful_swaps": successful_swaps,
            "successful_fraction": successful_swaps / max_swaps,
            "failed_swap_attempts": retries,
            "retained_edge_fraction": retained_fraction,
            "reached_max_swaps": reached_max_swaps,
            "hit_failed_attempt_limit": hit_failed_attempt_limit,
            "termination_reason": (
                "all_edge_pairs_swapped"
                if reached_max_swaps
                else "retry_limit_reached"
            ),
        }
        return A, stats

    return A
