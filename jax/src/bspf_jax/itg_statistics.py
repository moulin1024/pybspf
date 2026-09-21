"""Batch-mean uncertainty for time-averaged ITG transport.

Intervals are conditional on stationary, sufficiently mixing trajectories;
block-size sensitivity is reported rather than treating output samples as
independent. Accumulated temperature work gives time-integrated Q exactly
up to the same RK stages, without a sparse-sampling trapezoid approximation.
"""
import numpy as np
from scipy.stats import t as student_t


def heat_batch_means(times,temperature_work,a_t,*,start,end=None,width=50.):
    t=np.asarray(times,dtype=float);w=np.asarray(temperature_work,dtype=float)
    if t.ndim!=1 or t.size<2 or w.shape!=t.shape or not np.all(np.isfinite(t)) or not np.all(np.isfinite(w)) or np.any(np.diff(t)<=0):
        raise ValueError('finite, strictly increasing times and matching work required')
    if not np.isfinite(a_t) or a_t==0 or not np.isfinite(width) or width<=0:raise ValueError('nonzero a_t and positive width required')
    end=float(t[-1] if end is None else end)
    if not t[0]<=start<end<=t[-1]:raise ValueError('averaging window is outside the history')
    count=int(np.floor((end-start)/width+1e-10))
    if count<2:raise ValueError('at least two complete batches required')
    edges=start+np.arange(count+1)*width
    values=np.diff(np.interp(edges,t,w))/(a_t*width)
    mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(count))
    half=float(student_t.ppf(.975,count-1)*se)
    lag1=float(np.corrcoef(values[:-1],values[1:])[0,1]) if count>3 and np.std(values[:-1])>0 and np.std(values[1:])>0 else None
    return dict(width=float(width),count=count,start=float(edges[0]),end=float(edges[-1]),
        mean=mean,standard_error=se,ci95=[mean-half,mean+half],half_width=half,
        lag1_correlation=lag1,values=values.tolist())


def heat_uncertainty(times,temperature_work,a_t,*,start,end=None,widths=(25.,50.,100.,200.)):
    batches=[]
    for width in widths:
        try:b=heat_batch_means(times,temperature_work,a_t,start=start,end=end,width=width)
        except ValueError:
            # Invalid input must still raise; only omit widths with insufficient batches.
            heat_batch_means(times,temperature_work,a_t,start=start,end=end,width=min(widths))
            continue
        batches.append(b)
    eligible=[b for b in batches if b['count']>=8]
    if not eligible:raise ValueError('need at least eight batches for an uncertainty estimate')
    # Use the widest estimated CI among adequately populated block sizes.
    selected=max(eligible,key=lambda b:b['half_width'])
    errors=[b['standard_error'] for b in eligible]
    positive=[s for s in errors if s>0]
    sensitivity=max(positive)/min(positive) if positive else 1.
    return dict(blocks=batches,selected=selected,se_block_size_ratio=float(sensitivity),
        interval_assumption='stationary mixing process; batch independence is approximate')


def compare_heat_means(coarse,fine,*,tolerance=.05):
    """Welch difference interval / fine sample mean; overlap alone is insufficient."""
    a=coarse['selected'];b=fine['selected']
    va=a['standard_error']**2;vb=b['standard_error']**2;variance=va+vb
    denominator=va*va/(a['count']-1)+vb*vb/(b['count']-1)
    dof=variance*variance/denominator if denominator>0 else np.inf
    half=float(student_t.ppf(.975,dof)*np.sqrt(variance)) if variance else 0.
    scale=abs(b['mean'])
    if scale==0:raise ValueError('relative transport comparison requires nonzero reference mean')
    delta=a['mean']-b['mean'];interval=[(delta-half)/scale,(delta+half)/scale]
    return dict(relative_difference=float(delta/scale),difference_ci95=interval,
        tolerance=float(tolerance),equivalent_within_tolerance=bool(interval[0]>-tolerance and interval[1]<tolerance),
        welch_degrees_of_freedom=float(dof),
        qualification='conditional difference interval; denominator is the fine sample mean, not a ratio confidence interval')
