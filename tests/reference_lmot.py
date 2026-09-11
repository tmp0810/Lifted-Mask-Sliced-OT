"""Small dense oracle independent of lmot's fibers, block builder and moments."""
import numpy as np


def dense_reference(x, alpha, y, beta, theta, *, masked=True, fiber_tol=1e-12, mass_tol=1e-12):
    theta = np.asarray(theta)/np.linalg.norm(theta)
    def groups(points, weights):
        z = points @ theta
        fibers = []
        for i in np.argsort(z, kind="stable"):
            if not fibers or z[i] - z[fibers[-1][0]] > fiber_tol:
                fibers.append([])
            fibers[-1].append(i)
        fibers = [np.asarray(g) for g in fibers]
        masses = np.array([weights[g].sum() for g in fibers])
        edges = np.r_[0, np.cumsum(masses)]
        edges[-1] = 1
        return fibers, masses, edges
    source, A, ca = groups(x, alpha)
    target, B, cb = groups(y, beta)
    out = np.zeros((len(x), len(y)))
    blocks = []
    for a, I in enumerate(source):
        for b, J in enumerate(target):
            lam = max(0., min(ca[a+1], cb[b+1])-max(ca[a], cb[b]))
            if lam == 0:
                continue
            u, v = alpha[I]/A[a], beta[J]/B[b]
            if masked:
                mask = np.all(x[I, None, :] == y[None, J, :], axis=2)
                H = mask*np.minimum(u[:, None], v[None, :])
                r, t = u-H.sum(1), v-H.sum(0)
                eta = 1-H.sum()
                Q = H + np.outer(r, t)/eta if eta > mass_tol else H
            else:
                Q = np.outer(u, v)
            out[np.ix_(I, J)] = lam*Q
            blocks.append((I, J, lam))
    return out, blocks
