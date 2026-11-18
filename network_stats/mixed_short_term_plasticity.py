"""Heterogeneous STP using the unchanged homogeneous pilot's RK4 update."""
import numpy as np
from numba import njit
from network_stats.short_term_plasticity import PRESETS,_rk4_step


def mixed_assignment(n_nodes,seed):
    """Nearly equal neuron counts, randomized independently of graph and signs."""
    if n_nodes<2:raise ValueError("Mixed STP requires at least two neurons.")
    facilitating=np.zeros(n_nodes,dtype=bool)
    facilitating[np.random.default_rng(seed).choice(n_nodes,n_nodes//2,replace=False)]=True
    return facilitating


@njit(cache=True)
def _simulate_mixed(a,b,signal,U,tf,td,substeps,washout,facilitating):
    n=len(a);steps=len(signal)
    h=np.zeros(n,dtype=np.float32);utilization=U.copy();resources=np.ones(n,dtype=np.float64)
    transmitted=np.zeros(n,dtype=np.float32);states=np.empty((steps,n),dtype=np.float32)
    gain_sum=0.;gain_sq=0.;u_sum=0.;x_sum=0.;samples=0
    gain_min=1e30;gain_max=-1e30;state_min=1.;state_max=0.
    fac_sum=0.;dep_sum=0.;fac_count=0;dep_count=0
    for t in range(steps):
        for j in range(n):
            rate=abs(h[j])
            for _ in range(substeps):
                utilization[j],resources[j]=_rk4_step(utilization[j],resources[j],rate,U[j],tf[j],td[j],1./substeps)
            gain=utilization[j]*resources[j]/U[j]
            transmitted[j]=gain*h[j]
            state_min=min(state_min,utilization[j],resources[j]);state_max=max(state_max,utilization[j],resources[j])
            if t>=washout:
                gain_sum+=gain;gain_sq+=gain*gain;u_sum+=utilization[j];x_sum+=resources[j];samples+=1
                gain_min=min(gain_min,gain);gain_max=max(gain_max,gain)
                if facilitating[j]:fac_sum+=gain;fac_count+=1
                else:dep_sum+=gain;dep_count+=1
        h=np.tanh(a@transmitted+b*signal[t]);states[t]=h
    mean=gain_sum/samples
    diagnostics=np.array([mean,np.sqrt(max(0.,gain_sq/samples-mean*mean)),gain_min,gain_max,
                          u_sum/samples,x_sum/samples,state_min,state_max,
                          fac_sum/fac_count if fac_count else np.nan,dep_sum/dep_count if dep_count else np.nan])
    return states,diagnostics,utilization,resources


def run_mixed_stp_states(weights,input_weights,signal,facilitating,washout=500):
    """Same tanh/STP equations and integration as the preceding pilot, per-node presets."""
    w=np.asarray(weights,dtype=np.float32);b=np.asarray(input_weights,dtype=np.float32);signal=np.asarray(signal,dtype=np.float32)
    mask=np.asarray(facilitating)
    if w.ndim!=2 or w.shape[0]!=w.shape[1] or b.shape!=(len(w),1):raise ValueError("Require square W and N×1 input weights.")
    if mask.shape!=(len(w),) or mask.dtype!=np.bool_:raise ValueError("Require one Boolean type per presynaptic neuron.")
    if signal.ndim!=2 or signal.shape[1]!=1 or not 0<=washout<len(signal):raise ValueError("Invalid input signal/washout.")
    if not all(np.isfinite(x).all() for x in (w,b,signal)):raise ValueError("Nonfinite simulation input.")
    fac=PRESETS["facilitating"];dep=PRESETS["depressing"]
    if fac.substeps!=dep.substeps:raise ValueError("Presets must use the same integration resolution.")
    U=np.where(mask,fac.U,dep.U);tf=np.where(mask,fac.tau_f,dep.tau_f);td=np.where(mask,fac.tau_d,dep.tau_d)
    states,values,u,x=_simulate_mixed(np.ascontiguousarray(w.T),np.ascontiguousarray(b[:,0]),np.ascontiguousarray(signal[:,0]),
                                     U,tf,td,fac.substeps,washout,np.ascontiguousarray(mask))
    if not np.isfinite(states).all() or values[6]<-1e-8 or values[7]>1+1e-8:raise ValueError("STP state outside bounds.")
    names=("gain_mean","gain_sd","gain_min","gain_max","utilization_mean","resources_mean","stp_state_min","stp_state_max",
           "gain_facilitating_mean","gain_depressing_mean")
    return states,dict(zip(names,values.tolist())),u,x,U
