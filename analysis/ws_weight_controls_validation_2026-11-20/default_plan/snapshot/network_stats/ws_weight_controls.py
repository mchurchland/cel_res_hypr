"""Two WS magnitude controls paired exactly to an existing completed sweep.

Parent WS topology/signs and CE half-normal magnitudes are reused from validated
caches. No existing network is redrawn or changed.
"""
import numpy as np
from network_stats.architecture_sweep import rng_for,spectrum

WS_FAMILIES=('ws_binary','ws_halfnormal')
WS_DEFINITIONS={
 'ws_binary':'Parent directed WS-like topology and edge-sign mask, p=.1; all raw magnitudes one.',
 'ws_halfnormal':'Same parent WS topology/sign mask; exact signed half-normal multiset of the parent CE/ER controls; original mean-one raw magnitude units.'}


def extend_parent_cache(parent,parent_fractions,fractions,seed):
    """Copy selected cached parent conditions and append two new WS spectra."""
    n=int(parent['n_nodes']);reference_index=len(parent_fractions)
    half=np.abs(parent[f'ce_halfnormal__{reference_index}__values'])
    m=len(half);mapping=rng_for(seed,'WS parent weights').permutation(m)
    output={k:v for k,v in parent.items() if '__' not in k}
    # Preserve all reference matrices used by normalizations and preflight.
    indices=[parent_fractions.index(f) for f in fractions]+[reference_index]
    for index,old_index in enumerate(indices):
        suffix=f'__{old_index}__'
        for key,value in parent.items():
            if suffix in key:
                family,_,attribute=key.split('__')
                output[f'{family}__{index}__{attribute}']=value
        edges=parent[f'ws_signed_empirical__{old_index}__edges']
        empirical=parent[f'ws_signed_empirical__{old_index}__values']
        ce_empirical=parent[f'ce_empirical__{old_index}__values']
        ce_edges=parent[f'ce_empirical__{old_index}__edges']
        np.testing.assert_array_equal(parent[f'ce_halfnormal__{old_index}__edges'],ce_edges)
        np.testing.assert_array_equal(np.abs(parent[f'ce_halfnormal__{old_index}__values']),half)
        np.testing.assert_array_equal(empirical,ce_empirical[mapping],err_msg='WS parent-weight permutation no longer matches')
        if len(edges)!=m or np.any(edges[:,0]==edges[:,1]) or not np.isfinite(empirical).all() or np.any(empirical==0):
            raise ValueError('Invalid parent WS edges/values')
        signs=np.sign(empirical)
        values={'ws_binary':signs,'ws_halfnormal':half[mapping]*signs}
        for family,weights in values.items():
            paired='ce_binary' if family=='ws_binary' else 'ce_halfnormal'
            np.testing.assert_array_equal(np.sort(weights),np.sort(parent[f'{paired}__{old_index}__values']))
            w=np.zeros((n,n));w[edges[:,0],edges[:,1]]=weights
            eigenvalues,radius=spectrum(w)
            prefix=f'{family}__{index}'
            output.update({prefix+'__edges':edges.copy(),prefix+'__values':weights,
                           prefix+'__eigenvalues':eigenvalues,prefix+'__radius':np.array(radius)})
    return output
