"""Finite-basis held-out IPC estimates using library polynomials and readouts.

Dambre et al. (2012), https://doi.org/10.1038/srep00514 . Targets are
orthonormal products for iid Uniform[-1,1] input, over delayed inputs only.
This is a finite, regularized held-out estimate, not full population IPC.
"""
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from math import comb
import numpy as np
from scipy.special import eval_legendre
from network_stats.audited_temporal_tasks import validated_ridge


@dataclass(frozen=True)
class IPCSettings:
    max_delay: int = 10
    max_degree: int = 3

    def __post_init__(self):
        if any(int(v)!=v or v<1 for v in (self.max_delay,self.max_degree)):
            raise ValueError('IPC delay and degree must be positive integers')
        if self.target_count > 5000:
            raise ValueError('IPC basis exceeds 5,000 targets; choose a smaller explicit basis')

    @property
    def target_count(self):
        return comb(self.max_delay+self.max_degree,self.max_degree)-1

    def definition(self):
        return dict(**asdict(self),lags=list(range(1,self.max_delay+1)),
                    target_count=self.target_count,input_distribution='iid Uniform[-1,1]',
                    basis='products of sqrt(2*k+1)*scipy.special.eval_legendre(k,u[t-delay]); all multi-indices of total degree 1..max_degree; no constant/current-input target',
                    estimator='per target: 1 - test MSE / mean(test target squared); validation-selected sklearn SVD Ridge with intercept; sum signed estimates',
                    clipping='None for IPC; positive-part sum is a separately named diagnostic',
                    scope='finite delayed-input basis, not full IPC; readout protocol is a regularized held-out adaptation of Dambre 2012')


def ipc_targets(signal,settings):
    u=np.asarray(signal,dtype=np.float64)
    if u.ndim!=2 or u.shape[1]!=1 or not np.isfinite(u).all() or (abs(u)>1).any() or len(u)<=settings.max_delay:
        raise ValueError('IPC requires a finite scalar input in [-1,1] and sufficient history')
    exponents=[]
    for degree in range(1,settings.max_degree+1):
        for indices in combinations_with_replacement(range(settings.max_delay),degree):
            exponents.append(np.bincount(indices,minlength=settings.max_delay))
    exponents=np.asarray(exponents,dtype=np.int16)
    assert len(exponents)==settings.target_count
    history=np.column_stack([u[settings.max_delay-d:len(u)-d,0] for d in range(1,settings.max_delay+1)])
    polys={k:np.sqrt(2*k+1)*eval_legendre(k,history) for k in range(1,settings.max_degree+1)}
    targets=np.full((len(u),len(exponents)),np.nan)
    for j,exponent in enumerate(exponents):
        column=np.ones(len(history))
        for dim in np.flatnonzero(exponent):column*=polys[int(exponent[dim])][:,dim]
        targets[settings.max_delay:,j]=column
    return targets,exponents


def add_ipc_data(data,task_settings,settings):
    if settings.max_delay>task_settings.washout:
        raise ValueError('IPC delays must not exceed washout')
    targets,exponents=ipc_targets(data['mc_input'],settings)
    return dict(data,ipc_targets=targets,ipc_exponents=exponents)


def target_capacities(target,prediction):
    y,p=np.asarray(target,float),np.asarray(prediction,float)
    if y.ndim!=2 or y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('IPC predictions/targets must have identical finite 2D shapes')
    denominator=np.mean(y*y,axis=0)
    if not np.isfinite(denominator).all() or (denominator<=0).any():
        raise ValueError('IPC targets need positive finite second moments')
    scores=1-np.mean((y-p)**2,axis=0)/denominator
    if not np.isfinite(scores).all() or (scores>1+1e-12).any():
        raise ValueError('Invalid held-out IPC estimate')
    return scores


def score_ipc(states,data,tasks,settings):
    x=np.asarray(states,dtype=float);y=data['ipc_targets'];exponents=data['ipc_exponents']
    if x.ndim!=2 or len(x)!=tasks.length or not np.isfinite(x).all():
        raise ValueError('IPC needs a complete finite state trajectory')
    if y.shape!=(tasks.length,settings.target_count) or exponents.shape!=(settings.target_count,settings.max_delay):
        raise ValueError('IPC cache does not match the configured basis')
    a=tasks.washout;b=a+tasks.train;c=b+tasks.validation
    prediction,alpha,loss,scale=validated_ridge(x[a:b],y[a:b],x[b:c],y[b:c],x[c:],tasks.ridge_alphas)
    capacities=target_capacities(y[c:],prediction)
    baseline=target_capacities(y[c:],np.broadcast_to(y[a:b].mean(axis=0),y[c:].shape))
    degrees=exponents.sum(axis=1);cross=(exponents>0).sum(axis=1)>1
    result=dict(IPC=float(capacities.sum()),IPC_positive_part=float(np.maximum(capacities,0).sum()),
                IPC_mean_predictor_baseline=float(baseline.sum()),IPC_excess_over_mean=float((capacities-baseline).sum()),
                IPC_linear=float(capacities[degrees==1].sum()),IPC_nonlinear=float(capacities[degrees>1].sum()),
                IPC_crossdelay=float(capacities[cross].sum()),IPC_single_delay=float(capacities[~cross].sum()),
                IPC_negative_terms=int(np.count_nonzero(capacities<0)),IPC_target_count=settings.target_count,
                IPC_max_delay=settings.max_delay,IPC_max_degree=settings.max_degree,
                IPC_validation_mse=float(loss.mean()),IPC_readout_train_state_rms=scale,
                IPC_ridge_alpha_min=float(alpha.min()),IPC_ridge_alpha_max=float(alpha.max()))
    for degree in range(1,settings.max_degree+1):
        result[f'IPC_degree_{degree}']=float(capacities[degrees==degree].sum())
    return result
