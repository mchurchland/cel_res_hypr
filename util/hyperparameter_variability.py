"""Exact variance attribution on an equally weighted, complete factorial grid.

No regression fit or stochastic sampling of hyperparameters is required. All
interactions are retained. Results describe the chosen discrete parameter levels.
"""
from __future__ import annotations
import itertools
import numpy as np


def factorial_variability(values, levels):
    """Return variance components for one or a batch of complete grid surfaces.

    Columns follow lexicographically sorted grid coordinates. The first axis is
    a surface/seed axis; remaining axes enumerate parameter levels. A main
    effect is a conditional mean minus the grand mean. Each higher interaction
    subtracts every lower-order component. Orthogonality follows from the full
    factorial grid and equal weighting of its levels.

    ``total_fraction`` includes every interaction involving that parameter and
    therefore overlaps across parameters. ``allocated_fraction`` splits each
    interaction equally among its participants and sums to one. This allocation
    is an explicit reporting convention, not a uniquely identified causal share.
    """
    levels=tuple(int(n) for n in levels)
    if not levels or min(levels)<2:
        raise ValueError("Require dimensions with at least two levels each")
    y=np.asarray(values,float)
    if y.ndim==1:
        y=y[None,:]
    if y.ndim!=2 or y.shape[1]!=np.prod(levels) or not np.isfinite(y).all():
        raise ValueError("Require finite complete surfaces with the specified grid size")
    x=y.reshape((len(y),*levels))
    dimensions=len(levels)
    grid_axes=tuple(range(1,dimensions+1))
    constant=np.ptp(y,axis=1)==0
    grand=x.mean(axis=grid_axes,keepdims=True)
    components={():grand}
    names=[];variances=[]
    for order in range(1,dimensions+1):
        for subset in itertools.combinations(range(dimensions),order):
            average_axes=tuple(i+1 for i in range(dimensions) if i not in subset)
            conditional=x.mean(axis=average_axes,keepdims=True) if average_axes else x
            effect=conditional.copy()
            for lower_order,component in components.items():
                if set(lower_order).issubset(subset):
                    effect=effect-component
            components[subset]=effect
            v=np.mean(effect**2,axis=grid_axes)
            v[constant]=0
            names.append(subset);variances.append(v)
    variance=np.stack(variances,axis=1)
    total=y.var(axis=1)
    total[constant]=0
    if not np.allclose(variance.sum(axis=1),total,rtol=1e-10,atol=1e-12):
        raise ArithmeticError("Variance decomposition identity failed")
    main=np.stack([variance[:,names.index((i,))] for i in range(dimensions)],axis=1)
    involving=np.stack([sum(variance[:,j] for j,s in enumerate(names) if i in s) for i in range(dimensions)],axis=1)
    allocated=np.stack([sum(variance[:,j]/len(s) for j,s in enumerate(names) if i in s) for i in range(dimensions)],axis=1)
    fraction=lambda a:np.divide(a,total[:,None],out=np.full_like(a,np.nan),where=total[:,None]>0)
    ranges=[]
    for i in range(dimensions):
        span=np.ptp(x,axis=i+1)
        ranges.append(span.mean(axis=tuple(range(1,span.ndim))))
    return dict(subsets=names,component_variance=variance,total_variance=total,
                main_variance=main,total_involving_variance=involving,allocated_variance=allocated,
                main_fraction=fraction(main),total_fraction=fraction(involving),allocated_fraction=fraction(allocated),
                interaction_fraction=np.divide(total-main.sum(axis=1),total,out=np.full(len(y),np.nan),where=total>0),
                matched_sweep_rms_sd=np.sqrt(involving),mean_matched_sweep_range=np.stack(ranges,axis=1))
