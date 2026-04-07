from numba import jit
import numpy as np


def tour_to_edge_attribute(G, tour):
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges or tuple(reversed(e)) in tour_edges
    return in_tour


def tour_cost(dis_m, tour):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += dis_m[e]
    return c


@jit(nopython=True)
def tour_cost_2End(dis_m, tour2End):
    c = 0
    s = 0
    e = tour2End[0, 1]
    for i in range(tour2End.shape[0]):
        c += dis_m[s, e]
        s = e
        e = tour2End[s, 1]
    return c


@jit(nopython=True)
def two_opt(tour, i, j):
    if i == j:
        return tour
    a = tour[i, 0]
    b = tour[j, 0]
    tour[i, 0] = tour[i, 1]
    tour[i, 1] = j
    tour[j, 0] = i
    tour[a, 1] = b
    tour[b, 1] = tour[b, 0]
    tour[b, 0] = a
    c = tour[b, 1]
    while tour[c, 1] != j:
        d = tour[c, 0]
        tour[c, 0] = tour[c, 1]
        tour[c, 1] = d
        c = d
    return tour


@jit(nopython=True)
def two_opt_cost(tour, D, i, j):
    if i == j:
        return 0
    a = tour[i, 0]
    b = tour[j, 0]
    delta = D[a, b] + D[i, j] - D[a, i] - D[b, j]
    return delta


@jit(nopython=True)
def two_opt_a2a(tour, D, N, first_improvement=False, set_delta=0):
    best_move = None
    best_delta = set_delta
    idxs = range(0, len(tour) - 1)
    for i in idxs:
        for j in N[i]:
            if i in tour[j] or j in tour[i]:
                continue
            delta = two_opt_cost(tour, D, i, j)
            if delta < best_delta and not np.isclose(0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break
    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0, tour


@jit(nopython=True)
def two_opt_o2a(tour, D, i, first_improvement=False):
    assert i > 0 and i < len(tour) - 1
    best_move = None
    best_delta = 0
    idxs = range(1, len(tour) - 1)
    for j in idxs:
        if abs(i - j) < 2:
            continue
        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break
    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0, tour


@jit(nopython=True)
def two_opt_o2a_all(tour, D, N, i):
    best_move = None
    best_delta = 0
    idxs = N[i]
    for j in idxs:
        if i in tour[j] or j in tour[i]:
            continue
        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            tour = two_opt(tour, *best_move)
    return best_delta, tour


@jit(nopython=True)
def relocate(tour, i, j):
    a = tour[i, 0]
    b = tour[i, 1]
    tour[a, 1] = b
    tour[b, 0] = a
    d = tour[j, 1]
    tour[d, 0] = i
    tour[i, 0] = j
    tour[i, 1] = d
    tour[j, 1] = i
    return tour


@jit(nopython=True)
def relocate_cost(tour, D, i, j):
    if i == j:
        return 0
    a = tour[i, 0]
    b = i
    c = tour[i, 1]
    d = j
    e = tour[j, 1]
    delta = -D[a, b] - D[b, c] + D[a, c] - D[d, e] + D[d, b] + D[b, e]
    return delta


@jit(nopython=True)
def relocate_o2a(tour, D, i, first_improvement=False):
    assert i > 0 and i < len(tour) - 1
    best_move = None
    best_delta = 0
    idxs = range(1, len(tour) - 1)
    for j in idxs:
        if i == j:
            continue
        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break
    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0, tour


@jit(nopython=True)
def relocate_o2a_all(tour, D, N, i):
    best_move = None
    best_delta = 0
    for j in N[i]:
        if tour[j, 1] == i:
            continue
        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            tour = relocate(tour, *best_move)
    return best_delta, tour


@jit(nopython=True)
def relocate_a2a(tour, D, N, first_improvement=False, set_delta=0):
    best_move = None
    best_delta = set_delta
    idxs = range(0, len(tour) - 1)
    for i in idxs:
        for j in N[i]:
            if tour[j, 1] == i:
                continue
            delta = relocate_cost(tour, D, i, j)
            if delta < best_delta and not np.isclose(0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break
    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0, tour


def nearest_neighbor(dis_matrix, depot):
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)
    tour.append(depot)
    return tour


def nearest_neighbor_2End(dis_matrix, depot):
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)
    tour.append(depot)
    route2End = np.zeros((n, 2))
    route2End[0, 0] = tour[-2]
    route2End[0, 1] = tour[1]
    for i in range(1, n):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    return route2End


@jit(nopython=True)
def local_search(init_tour, init_cost, D, N, first_improvement=False):
    cur_route, cur_cost = init_tour, init_cost
    improved = True
    while improved:
        improved = False
        delta, new_tour = two_opt_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour
        delta, new_tour = relocate_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour
    return cur_route, cur_cost


@jit(nopython=True)
def route2tour(route):
    s = 0
    tour = []
    for i in range(len(route)):
        tour.append(route[s, 1])
        s = route[s, 1]
    return tour


@jit(nopython=True)
def tour2route(tour):
    n = len(tour)
    route2End = np.zeros((n, 2))
    route2End[tour[0], 0] = tour[-1]
    route2End[tour[0], 1] = tour[1]
    for i in range(1, n - 1):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    route2End[tour[n - 1], 0] = tour[n - 2]
    route2End[tour[n - 1], 1] = tour[0]
    return route2End


def check_valid_route(route, size):
    if len(route) != size:
        return False
    return set(route) == set(range(size))
