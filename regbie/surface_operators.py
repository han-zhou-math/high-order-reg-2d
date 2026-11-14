"""On-curve K, K', and hypersingular candidates from G_delta(r; b=0).

All normal derivatives freeze the on-surface regularizer coefficients.
W denotes +d/dn_y D, consistent with G=+log(r)/(2*pi).
"""
import math
import numpy as np
from .regularized_slp import coefficients
from .regularized_dlp import normals


def radial_factors(u, order, lam=0.):
    """Return A=T/u, A', and T' at fixed lam, finite at u=0.

    The Stokes factor V/u² is -A'. Coefficients must broadcast over u.
    """
    if order not in (3,5,7):
        raise ValueError('order must be 3, 5, or 7')
    u=np.asarray(u,float)
    if np.any(u<0) or not np.all(np.isfinite(u)):
        raise ValueError('u must be finite and nonnegative')
    cs=coefficients(order,lam)
    # P=1+u V, T=1-exp(-u)P.
    v=np.zeros_like(u)
    vp=np.zeros_like(u)
    for k,c in enumerate(cs,1):
        v+=(-1)**k*c*u**(k-1)
        if k>1: vp+=(-1)**k*c*(k-1)*u**(k-2)
    base=np.divide(-np.expm1(-u),u,out=np.ones_like(u),where=u!=0)
    bp=np.empty_like(u)
    small=u<.5
    # derivative of sum_j (-u)^j/(j+1)!
    us=u[small]
    acc=np.zeros_like(us)
    for j in range(20,0,-1):
        acc=acc*us+(-1)**j*j/math.factorial(j+1)
    bp[small]=acc
    ul=u[~small]
    bp[~small]=(ul*np.exp(-ul)+np.expm1(-ul))/ul**2
    e=np.exp(-u)
    a=base-e*v
    ap=bp+e*(v-vp)
    tp=e*(1+u*v-v-u*vp)
    return a,ap,tp


def evaluate_surface(curve, f, g, g_dt, theta, delta, n, order=7, chunk_size=64):
    """Evaluate K[g], K'[f], raw W[g], corrected W[g], and Maue W[g].

    f,g,g_dt take parameter arrays; g_dt is dg/dt, not dg/ds.
    Targets are curve(theta). The curve must be regular and CCW.
    K and K' are average boundary operators; add their analytic jump terms
    separately. Wraw is intentionally uncorrected, to test differentiation.
    Wcorrected adds the Laplacian defect; Wmaue = d_s S_delta d_s.
    """
    if delta<=0 or not np.isfinite(delta) or n<4 or chunk_size<1:
        raise ValueError('require delta>0, n>=4, chunk_size>=1')
    theta=np.asarray(theta,float)
    t=2*np.pi*np.arange(n)/n
    x,dx=curve(t)
    y,dy=curve(theta)
    nx,ny=normals(dx),normals(dy)
    ty=dy/np.linalg.norm(dy,axis=1)[:,None]
    speed=np.linalg.norm(dx,axis=1)
    fv,gv,gdot=f(t),g(t),g_dt(t)
    values={name:np.empty(len(theta)) for name in ['K','Kprime','Wraw','Wcorrected','Wmaue']}
    for start in range(0,len(theta),chunk_size):
        sl=slice(start,start+chunk_size)
        r=x[None,:,:]-y[sl,None,:]
        r2=np.sum(r*r,axis=-1)
        a,ap,tp=radial_factors(r2/delta**2,order)
        nxr=np.sum(nx[None,:,:]*r,axis=-1)
        nyr=np.sum(ny[sl,None,:]*r,axis=-1)
        tyr=np.sum(ty[sl,None,:]*r,axis=-1)
        nn=ny[sl] @ nx.T
        gradscale=a/delta**2
        # dt/(2*pi)=1/n. All displayed kernels here omit 1/(2*pi).
        values['K'][sl]=np.sum(nxr*gradscale*(gv*speed)[None,:],axis=1)/n
        values['Kprime'][sl]=np.sum(-nyr*gradscale*(fv*speed)[None,:],axis=1)/n
        raw=-nn*gradscale-2*nxr*nyr*ap/delta**4
        corrected=raw+2*nn*tp/delta**2
        values['Wraw'][sl]=np.sum(raw*(gv*speed)[None,:],axis=1)/n
        values['Wcorrected'][sl]=np.sum(corrected*(gv*speed)[None,:],axis=1)/n
        # g_s ds = g_dt dt; target derivative acts tangentially.
        values['Wmaue'][sl]=np.sum(-tyr*gradscale*gdot[None,:],axis=1)/n
    return values
