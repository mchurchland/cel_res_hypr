"""Rate-based Tsodyks–Markram/Mongillo STP adapter for the existing tanh RC.

Published dynamics: du/dt=(U-u)/tau_f+U(1-u)r,
dx/dt=(1-x)/tau_d-u*x*r. References:
https://pmc.ncbi.nlm.nih.gov/articles/PMC7321806/
https://pmc.ncbi.nlm.nih.gov/articles/PMC3105243/

The benchmark/readouts are reused elsewhere. Here the necessary rate adapter
uses r=abs(previous signed tanh state), one STP state per presynaptic neuron,
and output gain u*x/U. The latter matches resting gain to the static network.
These are exploratory modeling choices, not measured C. elegans parameters.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass(frozen=True)
class STPParameters:
    U: float
    tau_f: float
    tau_d: float
    substeps: int = 4

    def __post_init__(self):
        if not np.isfinite([self.U,self.tau_f,self.tau_d]).all() or not 0<self.U<=1:
            raise ValueError("Require finite parameters and 0<U<=1.")
        if min(self.tau_f,self.tau_d)<=0 or self.substeps<1 or int(self.substeps)!=self.substeps:
            raise ValueError("Require positive time constants and integer substeps.")


PRESETS={"facilitating":STPParameters(.15,75.,5.),
         "depressing":STPParameters(.45,5.,75.)}


@njit(cache=True)
def _rhs(u,x,rate,U,tf,td):
    return (U-u)/tf+U*(1-u)*rate, (1-x)/td-u*x*rate


@njit(cache=True)
def _rk4_step(u,x,rate,U,tf,td,dt):
    ku1,kx1=_rhs(u,x,rate,U,tf,td)
    ku2,kx2=_rhs(u+dt*ku1/2,x+dt*kx1/2,rate,U,tf,td)
    ku3,kx3=_rhs(u+dt*ku2/2,x+dt*kx2/2,rate,U,tf,td)
    ku4,kx4=_rhs(u+dt*ku3,x+dt*kx3,rate,U,tf,td)
    return u+dt*(ku1+2*ku2+2*ku3+ku4)/6,x+dt*(kx1+2*kx2+2*kx3+kx4)/6


def stp_constant_rate(rate,steps,parameters):
    """Small independent synapse diagnostic, initialized at recovered rest."""
    if not np.isfinite(rate) or rate<0 or steps<1:
        raise ValueError("Require nonnegative rate and positive step count.")
    u,x=parameters.U,1.
    history=[]
    for _ in range(steps):
        for _ in range(parameters.substeps):
            u,x=_rk4_step(u,x,rate,parameters.U,parameters.tau_f,parameters.tau_d,1/parameters.substeps)
        history.append([u,x,u*x/parameters.U])
    return np.asarray(history)


@njit(cache=True)
def _simulate(a,b,signal,U,tf,td,substeps,washout):
    n=len(a); steps=len(signal)
    h=np.zeros(n,dtype=np.float32)
    utilization=np.full(n,U,dtype=np.float64)
    resources=np.ones(n,dtype=np.float64)
    transmitted=np.zeros(n,dtype=np.float32)
    states=np.empty((steps,n),dtype=np.float32)
    gain_sum=0.;gain_sq=0.;u_sum=0.;x_sum=0.;samples=0
    gain_min=1e30;gain_max=-1e30;state_min=1.;state_max=0.
    for t in range(steps):
        for j in range(n):
            rate=abs(h[j])
            for _ in range(substeps):
                utilization[j],resources[j]=_rk4_step(utilization[j],resources[j],rate,U,tf,td,1./substeps)
            gain=utilization[j]*resources[j]/U
            transmitted[j]=gain*h[j]
            state_min=min(state_min,utilization[j],resources[j])
            state_max=max(state_max,utilization[j],resources[j])
            if t>=washout:
                gain_sum+=gain;gain_sq+=gain*gain;u_sum+=utilization[j];x_sum+=resources[j];samples+=1
                gain_min=min(gain_min,gain);gain_max=max(gain_max,gain)
        pre=a@transmitted+b*signal[t]
        h=np.tanh(pre)
        states[t]=h
    mean=gain_sum/samples
    diagnostics=np.array([mean,np.sqrt(max(0.,gain_sq/samples-mean*mean)),gain_min,gain_max,
                          u_sum/samples,x_sum/samples,state_min,state_max])
    return states,diagnostics,utilization,resources


def run_stp_states(weights,input_weights,signal,parameters,washout=500):
    """Unit-leak, zero-bias recurrence; plasticity applies only to recurrent output.

    One signal sample is one model time unit. STP is integrated with RK4 using
    held previous-state rate, then the existing discrete tanh update is applied.
    No spectral-radius normalization, clipping or weight learning occurs here.
    """
    w=np.asarray(weights,dtype=np.float32);b=np.asarray(input_weights,dtype=np.float32)
    signal=np.asarray(signal,dtype=np.float32)
    if w.ndim!=2 or w.shape[0]!=w.shape[1] or b.shape!=(len(w),1):
        raise ValueError("Require square W and N×1 input weights.")
    if signal.ndim!=2 or signal.shape[1]!=1 or not 0<=washout<len(signal):
        raise ValueError("Require T×1 signal and valid washout.")
    if not all(np.isfinite(a).all() for a in (w,b,signal)):
        raise ValueError("Nonfinite simulation input.")
    result=_simulate(np.ascontiguousarray(w.T),np.ascontiguousarray(b[:,0]),np.ascontiguousarray(signal[:,0]),
                     parameters.U,parameters.tau_f,parameters.tau_d,parameters.substeps,washout)
    states,values,utilization,resources=result
    if not np.isfinite(states).all() or values[6]<-1e-8 or values[7]>1+1e-8:
        raise ValueError("Invalid STP state; no automatic clipping is applied.")
    names=("gain_mean","gain_sd","gain_min","gain_max","utilization_mean","resources_mean","stp_state_min","stp_state_max")
    return states,dict(zip(names,values.tolist())),utilization,resources
