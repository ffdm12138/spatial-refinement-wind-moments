"""Independent scoring kernels; no imports from historical execution runners."""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad


def finite(x):
    x = np.asarray(x, dtype=float)
    if not x.size or not np.isfinite(x).all():
        raise ValueError('Empty or nonfinite input')
    return x


def rmse(prediction, truth):
    p, y = finite(prediction), finite(truth)
    if p.shape != y.shape:
        raise ValueError('Prediction and truth shapes differ')
    return float(np.sqrt(np.mean(np.square(p-y))))


def empirical_crps(y, sample):
    """Equal-weight empirical forecast (ensemble CRPS, not fair CRPS)."""
    y, x = finite(y), np.sort(finite(sample))
    if x.ndim != 1:
        raise ValueError('Forecast sample must be one dimensional')
    n = len(x)
    prefix = np.r_[0., np.cumsum(x)]
    pos = np.searchsorted(x, y, side='right')
    eabs = (pos*y-prefix[pos]+prefix[n]-prefix[pos]-(n-pos)*y)/n
    half_pair_distance = np.sum((2*np.arange(n)-n+1)*x)/(n*n)
    return eabs-half_pair_distance


def pinball(y, q, tau):
    y, q, tau = finite(y), finite(q), finite(tau)
    if np.any((tau <= 0) | (tau >= 1)):
        raise ValueError('Quantile levels must be strictly between 0 and 1')
    error = y-q
    return np.maximum(tau*error, (tau-1)*error)


def cdf_crps(distribution, y):
    """Independent CDF integration on full support, including both tails."""
    a, b = distribution.support()
    y = float(y)
    if not np.isfinite(y):
        raise ValueError('Nonfinite observation')
    left = quad(lambda z: float(distribution.cdf(z))**2, a, min(max(y,a),b), epsabs=1e-8)[0] if y>a else 0.
    right = quad(lambda z: float(distribution.sf(z))**2, max(min(y,b),a), b, epsabs=1e-8)[0] if y<b else 0.
    return float(left+right+max(a-y, 0)+max(y-b, 0))


def paired_gain(prediction, baseline, truth, indices, *, loss='rmse', batch=64):
    """Resample complete time x space fields; compute RMSE inside each replicate.

    For precomputed CRPS/W1 losses use loss='mean' and truth=None explicitly.
    An undefined zero-baseline relative gain is returned as None, never epsilon.
    """
    p, b = finite(prediction), finite(baseline)
    if p.shape != b.shape or p.ndim != 2:
        raise ValueError('Require matching time x spatial-unit matrices')
    idx = np.asarray(indices)
    if idx.ndim != 2 or idx.shape[1] != p.shape[0] or not np.issubdtype(idx.dtype, np.integer) or idx.min()<0 or idx.max()>=len(p):
        raise ValueError('Invalid shared resampling indices')
    if loss == 'rmse':
        if truth is None:
            raise ValueError('RMSE requires explicit truth')
        y = finite(truth)
        if y.shape != p.shape:
            raise ValueError('Truth shape mismatch')
        mp, bp = np.mean((p-y)**2, axis=1), np.mean((b-y)**2, axis=1)
        transform = np.sqrt
    elif loss == 'mean':
        if truth is not None or np.any(p<0) or np.any(b<0):
            raise ValueError('Mean loss expects nonnegative precomputed losses and no truth')
        mp, bp = p.mean(axis=1), b.mean(axis=1)
        transform = lambda x: x
    else:
        raise ValueError(loss)
    point_m, point_b = float(transform(mp.mean())), float(transform(bp.mean()))
    gains = []
    for start in range(0,len(idx),batch):
        selection = idx[start:start+batch]
        bm, mm = transform(bp[selection].mean(axis=1)), transform(mp[selection].mean(axis=1))
        gains.extend(np.divide(bm-mm,bm,out=np.full_like(bm,np.nan),where=bm>0))
    gains=np.asarray(gains); valid=np.isfinite(gains)
    # Do not silently condition a CI on only the replicates with defined gains.
    ci_valid=bool(point_b>0 and valid.all())
    return dict(model_loss=point_m, baseline_loss=point_b,
                point_gain=1-point_m/point_b if point_b>0 else None,
                gain_lower95=float(np.percentile(gains,2.5)) if ci_valid else None,
                gain_upper95=float(np.percentile(gains,97.5)) if ci_valid else None,
                gain_mean=float(gains.mean()) if ci_valid else None,
                valid_replicates=int(valid.sum()), replicates=len(idx),
                status='DEFINED' if ci_valid else 'UNDEFINED_ZERO_BASELINE')


def block_indices(n, block, reps, seed):
    if not (n>=block>0 and reps>0):
        raise ValueError('Require 0 < block <= n and positive replicate count')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-block+1,size=(reps,int(np.ceil(n/block))))
    return (starts[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:n].astype(np.int32)
