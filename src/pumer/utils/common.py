def get_neighbor_indices(h, w, r=1):
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            # all indices for neighbors with radius r
            neighbor_indices = []
            for ii in range(-1 * r, r + 1):
                for jj in range(-1 * r, r + 1):
                    if ii == jj == 0:  # skip i,j itself
                        continue
                    i_idx = i + ii
                    j_idx = j + jj
                    if 0 <= i_idx <= h - 1 and 0 <= j_idx <= w - 1:
                        neighbor_idx = i_idx * w + j_idx
                        neighbor_indices.append(neighbor_idx)
            yield idx, neighbor_indices


def _kl_bernoulli_bernoulli(p, q):
    t1 = p.probs * (p.probs / q.probs).log()
    t1[q.probs == 0] = torch.inf
    t1[p.probs == 0] = 0
    t2 = (1 - p.probs) * ((1 - p.probs) / (1 - q.probs)).log()
    t2[q.probs == 1] = torch.inf
    t2[p.probs == 1] = 0
    return t1 + t2
