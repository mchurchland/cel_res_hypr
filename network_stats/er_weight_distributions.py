"""Positive magnitude controls on exactly paired, archived ER networks."""
import numpy as np
from scipy import optimize,stats
from network_stats.architecture_sweep import rng_for,spectrum

FAMILIES=('empirical','binary','halfnormal','cap99','cap95','cap90','gamma','lognormal','hetero50','hetero75')
NORMALIZATIONS=('own_radius','shared_ce_scale','frobenius_match')
PARENT_FAMILIES={'empirical':'er_signed_empirical','binary':'er_binary','halfnormal':'er_halfnormal'}
LABELS={'empirical':'Empirical CE magnitudes','binary':'Binary','halfnormal':'Half-normal',
        'cap99':'Empirical capped at Q99','cap95':'Empirical capped at Q95','cap90':'Empirical capped at Q90',
        'gamma':'Moment-matched gamma','lognormal':'Moment-matched lognormal',
        'hetero50':'Heterogeneity a = 0.50','hetero75':'Heterogeneity a = 0.75'}
COLORS=dict(zip(FAMILIES,['#d55e00','#009e73','#0072b2','#e69f00','#cc79a7','#882255','#332288','#44aa99','#777777','#88ccee']))
SETTINGS=dict(caps=[.99,.95,.90],heterogeneity=[.5,.75],moment_matching='finite positive mid-quantile vector; population variance ddof=0',
    moment_methods='scipy.stats.gamma/lognorm.ppf and scipy.optimize.brentq; loc=0; sample mean restored',
    placement='monotone map of parent ER empirical magnitudes; deterministic shared random tie-breaking per realization',
    draw_policy='fixed mid-quantile distribution, not iid samples; parent topology, signs and rank-ties carry realization variability',
    cap_policy='cap before mean restoration; record original cutoff and restoration factor',
    frobenius_reference='unmodified CE empirical matrix at the same E/I fraction after shared-CE scaling',
    frobenius_equality_tolerance=1e-10,variance_ddof=0)


def magnitudes(values):
    w=np.asarray(values,dtype=np.float64)
    if w.ndim!=1 or len(w)<2 or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('Require at least two finite, strictly positive edge magnitudes')
    return w


def moment_matched(reference,kind):
    """Match realized mean/variance without affine shifts or negative magnitudes."""
    w=magnitudes(reference);mean=w.mean();target=w.std()/mean
    if kind not in ('gamma','lognormal'):raise ValueError('Unknown distribution')
    if target<1e-12:return np.full(len(w),mean),0.
    q=(np.arange(len(w),dtype=float)+.5)/len(w)
    distribution=stats.gamma if kind=='gamma' else stats.lognorm
    def vector(log_shape):
        v=distribution.ppf(q,np.exp(log_shape))
        if not np.isfinite(v).all() or v.sum()<=0:raise ValueError('Unrepresentable distribution shape')
        return v*(mean/v.mean())
    def objective(log_shape):return vector(log_shape).std()/mean-target
    lower,upper=-1.,1.
    for _ in range(12):
        if objective(lower)*objective(upper)<=0:break
        # Gamma CV decreases with shape; lognormal CV increases with shape.
        if (objective(lower)<0)==(kind=='gamma'):lower-=1.
        else:upper+=1.
    else:raise ValueError('Could not bracket a positive moment-matched distribution')
    root=optimize.brentq(objective,lower,upper,xtol=1e-12,rtol=1e-13)
    result=vector(root)
    if np.any(result<=0) or not np.allclose([result.mean(),result.var()],[mean,w.var()],rtol=1e-10,atol=1e-13):
        raise ValueError('Finite-vector moment matching failed; no clipping or sign changes allowed')
    return result,float(np.exp(root))


def make_magnitude_controls(reference,seed):
    w=magnitudes(reference);mean=w.mean();order=np.lexsort((rng_for(seed,'ER distribution magnitude rank ties').random(len(w)),w))
    result={'empirical':w.copy()};details={family:dict(fit_shape=0.,cap_cutoff=0.,mean_restoration=1.) for family in FAMILIES}
    for family,quantile in [('cap99',.99),('cap95',.95),('cap90',.90)]:
        cutoff=float(np.quantile(w,quantile,method='linear'));capped=np.minimum(w,cutoff);restoration=float(mean/capped.mean())
        result[family]=capped*restoration;details[family].update(cap_cutoff=cutoff,mean_restoration=restoration)
    for family in ['gamma','lognormal']:
        sorted_weights,shape=moment_matched(w,family);vector=np.empty_like(w);vector[order]=sorted_weights
        result[family]=vector;details[family]['fit_shape']=shape
    for family,a in [('hetero50',.5),('hetero75',.75)]:result[family]=mean+a*(w-mean)
    for family,weights in result.items():
        if not np.isfinite(weights).all() or np.any(weights<=0) or not np.isclose(weights.mean(),mean,rtol=1e-12):
            raise ValueError(f'Invalid transformed magnitudes: {family}')
    return result,details


def build_cache(parent,parent_fractions,fractions,seed,families):
    """Reuse ER masks/signs and all CE normalization references, without redrawing."""
    reference_index=len(parent_fractions);original=np.abs(parent[f'er_signed_empirical__{reference_index}__values'])
    controls,details=make_magnitude_controls(original,seed)
    out={k:np.asarray(v).copy() for k,v in parent.items() if '__' not in k}
    indices=[parent_fractions.index(f) for f in fractions]+[reference_index]
    for index,old in enumerate(indices):
        for baseline in ['ce_empirical','er_signed_empirical','er_binary','er_halfnormal']:
            for attribute in ['edges','values','eigenvalues','radius']:
                out[f'{baseline}__{index}__{attribute}']=parent[f'{baseline}__{old}__{attribute}'].copy()
        edges=parent[f'er_signed_empirical__{old}__edges'];signed=parent[f'er_signed_empirical__{old}__values']
        np.testing.assert_array_equal(np.abs(signed),original)
        for family in families:
            if family in PARENT_FAMILIES:
                source=PARENT_FAMILIES[family]
                np.testing.assert_array_equal(parent[f'{source}__{old}__edges'],edges)
                weights=parent[f'{source}__{old}__values'].copy()
                ev=parent[f'{source}__{old}__eigenvalues'].copy();radius=float(parent[f'{source}__{old}__radius'])
            else:
                weights=controls[family]*np.sign(signed);matrix=np.zeros((int(parent['n_nodes']),)*2)
                matrix[edges[:,0],edges[:,1]]=weights;ev,radius=spectrum(matrix)
            np.testing.assert_array_equal(np.sign(weights),np.sign(signed))
            prefix=f'{family}__{index}'
            out.update({prefix+'__edges':edges.copy(),prefix+'__values':weights,prefix+'__eigenvalues':ev,prefix+'__radius':np.array(radius)})
            for key,value in details[family].items():out[prefix+'__'+key]=np.array(value)
    return out


def scaling_factor(mode,target,radius,ce_radius,frobenius,reference_frobenius):
    if mode not in NORMALIZATIONS or not np.isfinite([target,radius,ce_radius,frobenius,reference_frobenius]).all() or min(target,radius,ce_radius,frobenius,reference_frobenius)<=0:
        raise ValueError('Invalid scaling mode/reference')
    if mode=='own_radius':return float(target/radius)
    shared=float(target/ce_radius)
    if mode=='shared_ce_scale':return shared
    # Canonicalize moment-equal matrices so paired score reuse is bitwise valid.
    ratio=1. if np.isclose(frobenius,reference_frobenius,rtol=SETTINGS['frobenius_equality_tolerance'],atol=0) else reference_frobenius/frobenius
    return float(shared*ratio)
