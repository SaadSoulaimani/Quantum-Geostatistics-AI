"""
Quantum geostatistics: core library.
Synthetic + real (Hajjar) data; matched extent; nested CV; uncertainty; diagnostics.
"""
from __future__ import annotations
import numpy as np, re, time, warnings
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from scipy.special import kv, gamma as gfun
from scipy.stats import norm, wilcoxon
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as CK, WhiteKernel
from sklearn.cluster import KMeans
from qutip import basis, tensor, sigmax, sigmay, sigmaz, qeye
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 0. Real data (Hajjar, Soulaimani et al., 2020)
# ----------------------------------------------------------------------
def load_survey(path):
    X,Y,V,L=[],[],[],[]; line=None
    for raw in open(path, encoding="utf-8", errors="ignore"):
        s=raw.strip()
        if not s: continue
        m=re.match(r"^Line\s+(\d+)", s)
        if m: line=int(m.group(1)); continue
        p=s.split()
        if len(p)>=4:
            try: x,y,v=float(p[0]),float(p[1]),float(p[3])
            except ValueError: continue
            X.append(x);Y.append(y);V.append(v);L.append(line)
    X,Y,V,L=map(np.asarray,(X,Y,V,L))
    ux=np.unique(np.round(X,1)); uy=np.unique(np.round(Y,1))
    return dict(X=X,Y=Y,V=V,line=L,nx=len(ux),ny=len(uy),
                dx=float(np.median(np.diff(ux))), dy=float(np.median(np.diff(uy))),
                x0=X.min(),y0=Y.min(),Lx=X.max()-X.min(),Ly=Y.max()-Y.min())

def survey_to_grid(d):
    """Return (xs, ys, Z[ny,nx]) in physical metres; NaN where missing."""
    xs=np.unique(np.round(d["X"],1)); ys=np.unique(np.round(d["Y"],1))
    Z=np.full((len(ys),len(xs)),np.nan)
    ix=np.searchsorted(xs,np.round(d["X"],1)); iy=np.searchsorted(ys,np.round(d["Y"],1))
    Z[iy,ix]=d["V"]
    return xs,ys,Z

# ----------------------------------------------------------------------
# 1. Synthetic fields on a physical grid (matched to the Hajjar window)
# ----------------------------------------------------------------------
def grf_gaussian(nx, ny, dx, ell, seed, anis=None, pad=2):
    """
    Stationary Gaussian random field with covariance C(h)=exp(-h^2/ell^2)
    (Gaussian model) on an ny x nx grid with spacing dx (metres).
    anis=(ell_u, ell_v, theta_deg): anisotropic Gaussian covariance
    exp(-(u^2/ell_u^2 + v^2/ell_v^2)) in axes rotated by theta.
    Spectral synthesis with zero-padding to suppress wrap-around.
    """
    r=np.random.default_rng(seed)
    Nx,Ny=pad*nx,pad*ny
    kx=np.fft.fftfreq(Nx,d=dx); ky=np.fft.fftfreq(Ny,d=dx)
    KX,KY=np.meshgrid(kx,ky)
    if anis is None:
        S=np.exp(-(np.pi*ell)**2*(KX**2+KY**2))
    else:
        eu,ev,th=anis; t=np.deg2rad(th)
        KU= KX*np.cos(t)+KY*np.sin(t); KV=-KX*np.sin(t)+KY*np.cos(t)
        S=np.exp(-np.pi**2*(eu**2*KU**2+ev**2*KV**2))
    noise=r.normal(size=(Ny,Nx))+1j*r.normal(size=(Ny,Nx))
    F=np.fft.ifft2(noise*np.sqrt(S)).real[:ny,:nx]
    return (F-F.mean())/F.std()

def grf_nonstationary(nx, ny, dx, ell_west, ell_east, seed, width_frac=0.15):
    """Locally varying correlation length: smooth blend of two stationary fields."""
    F1=grf_gaussian(nx,ny,dx,ell_west,seed)
    F2=grf_gaussian(nx,ny,dx,ell_east,seed+1000)
    x=np.linspace(0,1,nx)[None,:]*np.ones((ny,1))
    w=1/(1+np.exp((x-0.5)/width_frac))          # 1 in the west, 0 in the east
    F=w*F1+(1-w)*F2
    return (F-F.mean())/F.std()

def make_grid(nx,ny,dx,x0=0.0,y0=0.0):
    xs=x0+dx*np.arange(nx); ys=y0+dx*np.arange(ny)
    XX,YY=np.meshgrid(xs,ys)
    return xs,ys,np.column_stack([XX.ravel(),YY.ravel()])

# ----------------------------------------------------------------------
# 2. Sampling designs
# ----------------------------------------------------------------------
def sample_design(grid, n, design, seed, nx=None, ny=None):
    r=np.random.default_rng(seed); N=len(grid)
    if design=="random":
        return r.choice(N,n,replace=False)
    if design=="clustered":
        k=max(3,n//30); centers=r.choice(N,k,replace=False)
        span=np.ptp(grid,axis=0).max()
        idx=[]
        for c in centers:
            d=np.linalg.norm(grid-grid[c],axis=1)
            p=np.exp(-(d/(0.08*span))**2); p/=p.sum()
            idx+=list(r.choice(N,n//k,replace=False,p=p))
        idx=np.unique(idx)
        while len(idx)<n: idx=np.unique(np.append(idx,r.choice(N)))
        return idx[:n]
    if design=="lines":
        # keep every k-th survey line and every m-th station along a line
        assert nx and ny
        ii=np.arange(N); row=ii//nx; col=ii%nx
        step_l=max(1,int(round(ny/np.sqrt(n*ny/nx)))); step_p=max(1,int(round(nx/np.sqrt(n*nx/ny))))
        m=(row%step_l==0)&(col%step_p==0)
        idx=ii[m]
        if len(idx)>n: idx=r.choice(idx,n,replace=False)
        return idx
    raise ValueError(design)

# ----------------------------------------------------------------------
# 3. Variogram models (practical-range convention), WLS fit, model selection
# ----------------------------------------------------------------------
def experimental_variogram(P,z,nlags=15,hmax=None):
    D=cdist(P,P); dz2=0.5*(z[:,None]-z[None,:])**2
    iu=np.triu_indices_from(D,1); d,v=D[iu],dz2[iu]
    if hmax is None: hmax=0.6*d.max()
    edges=np.linspace(0,hmax,nlags+1); c,g,npairs=[],[],[]
    for a,b in zip(edges[:-1],edges[1:]):
        m=(d>=a)&(d<b)
        if m.sum()>=5: c.append(0.5*(a+b)); g.append(v[m].mean()); npairs.append(m.sum())
    return np.array(c),np.array(g),np.array(npairs)

def directional_variogram(P,z,angle_deg,tol_deg=22.5,nlags=12,hmax=None):
    D=cdist(P,P); dz2=0.5*(z[:,None]-z[None,:])**2
    dxm=P[:,0][None,:]-P[:,0][:,None]; dym=P[:,1][None,:]-P[:,1][:,None]
    ang=np.degrees(np.arctan2(dym,dxm))%180
    iu=np.triu_indices_from(D,1); d,v,a=D[iu],dz2[iu],ang[iu]
    da=np.minimum(np.abs(a-angle_deg),180-np.abs(a-angle_deg))
    sel=da<=tol_deg; d,v=d[sel],v[sel]
    if hmax is None: hmax=0.6*D.max()
    edges=np.linspace(0,hmax,nlags+1); c,g=[],[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(d>=lo)&(d<hi)
        if m.sum()>=5: c.append(0.5*(lo+hi)); g.append(v[m].mean())
    return np.array(c),np.array(g)

def _matern(h,a,nu):
    # correlation with practical range a (95%): scale so that rho(a)=0.05 approximately
    # use scaling factor kappa(nu) computed numerically once
    kappa={0.5:3.0,1.5:4.744,2.5:5.918}[nu]
    x=kappa*h/a; x=np.where(x<1e-12,1e-12,x)
    return (2**(1-nu)/gfun(nu))*(x**nu)*kv(nu,x)

VARIO_MODELS={
 "gaussian":   lambda h,a: 1-np.exp(-3*h**2/a**2),
 "exponential":lambda h,a: 1-np.exp(-3*h/a),
 "matern15":   lambda h,a: 1-_matern(h,a,1.5),
 "spherical":  lambda h,a: np.where(h<a,1.5*h/a-0.5*(h/a)**3,1.0),
}
def vmodel(h,params,model):
    c0,c,a=params; return c0+c*VARIO_MODELS[model](h,a)

def fit_variogram(hc,gc,npairs,model,hmax):
    def res(p):
        return np.sqrt(npairs)*(vmodel(hc,p,model)-gc)/(gc+1e-9)   # Cressie-style weights
    best=None
    for a0 in [0.15*hmax,0.3*hmax,0.6*hmax]:
        p0=[0.05*gc.max(),0.95*gc.max(),a0]
        r=least_squares(res,p0,bounds=([0,1e-6,0.02*hmax],[gc.max(),5*gc.max(),4*hmax]))
        if best is None or r.cost<best.cost: best=r
    return best.x,best.cost

def select_variogram(P,z,models=("gaussian","exponential","matern15","spherical"),nlags=15):
    hc,gc,npairs=experimental_variogram(P,z,nlags)
    hmax=hc.max(); fits={}
    for m in models:
        p,cost=fit_variogram(hc,gc,npairs,m,hmax); fits[m]=(p,cost)
    best=min(fits,key=lambda m:fits[m][1])
    return best,fits,hc,gc

# ----------------------------------------------------------------------
# 4. Kriging (ordinary, with Lagrange constraint) and GP predictors
# ----------------------------------------------------------------------
def ordinary_kriging(P_obs,z_obs,P_pred,params,model,jitter=1e-3):
    c0,c,a=params; n=len(P_obs)
    C=(c0+c)-vmodel(cdist(P_obs,P_obs),params,model); np.fill_diagonal(C,c0+c)
    C=C+jitter*(c0+c)*np.eye(n)          # numerical nugget (Gaussian model is ill-conditioned)
    A=np.ones((n+1,n+1)); A[:n,:n]=C; A[n,n]=0
    Cp=(c0+c)-vmodel(cdist(P_pred,P_obs),params,model)
    B=np.ones((n+1,len(P_pred))); B[:n,:]=Cp.T
    try: W=np.linalg.solve(A,B)
    except np.linalg.LinAlgError: W=np.linalg.lstsq(A,B,rcond=None)[0]
    zp=W[:n,:].T@z_obs
    var=(c0+c)-np.einsum("ij,ji->i",W.T,B)
    return zp,np.sqrt(np.clip(var,1e-12,None))

def gp_fit_predict(K_tr,K_ts,k_ts_diag,y,sig_f,sig_n,jitter=1e-8):
    """Simple kriging with estimated constant mean (GP with constant mean).
    Returns mean and predictive std (incl. noise). K's are correlation matrices (unit diag)."""
    n=len(y); mu=y.mean()
    A=sig_f*K_tr+(sig_n+jitter)*np.eye(n)
    L=np.linalg.cholesky(A)
    alpha=np.linalg.solve(L.T,np.linalg.solve(L,y-mu))
    m=sig_f*K_ts@alpha+mu
    v=np.linalg.solve(L,(sig_f*K_ts).T)
    var=sig_f*k_ts_diag-np.sum(v**2,axis=0)+sig_n
    return m,np.sqrt(np.clip(var,1e-12,None))

def gp_nll(K,y,sig_f,sig_n,jitter=1e-8):
    n=len(y); mu=y.mean(); A=sig_f*K+(sig_n+jitter)*np.eye(n)
    try: L=np.linalg.cholesky(A)
    except np.linalg.LinAlgError: return np.inf
    a=np.linalg.solve(L,y-mu)
    return 0.5*a@a+np.log(np.diag(L)).sum()+0.5*n*np.log(2*np.pi)

def ml_fit_scale_noise(K,y,sf_grid=None,sn_grid=None):
    """Type-II ML on a log grid for amplitude and noise variance (kernel shape fixed)."""
    v=np.var(y)
    if sf_grid is None: sf_grid=v*np.logspace(-1,1,7)
    if sn_grid is None: sn_grid=v*np.logspace(-4,0,9)
    best=(np.inf,None,None)
    for sf in sf_grid:
        for sn in sn_grid:
            nll=gp_nll(K,y,sf,sn)
            if nll<best[0]: best=(nll,sf,sn)
    return best[1],best[2],best[0]

def rbf_corr(A,B,ell): return np.exp(-cdist(A,B)**2/(2*ell**2))

# ----------------------------------------------------------------------
# 5. Quantum feature map (QuTiP reference) and verified fast encoder
# ----------------------------------------------------------------------
def _Ry(t): return (-1j*(t/2)*sigmay()).expm()
def _Rz(t): return (-1j*(t/2)*sigmaz()).expm()
def _cnot(nq,c,t):
    P0=basis(2,0)*basis(2,0).dag(); P1=basis(2,1)*basis(2,1).dag()
    a=[qeye(2)]*nq; b=[qeye(2)]*nq; a[c]=P0; b[c]=P1; b[t]=sigmax()
    return tensor(a)+tensor(b)

def feature_state_qutip(x,y,nq=4,reps=2,beta=1.5,entangle=True):
    """Reference implementation with QuTiP (state-vector simulation)."""
    CN=[_cnot(nq,i,(i+1)%nq) for i in range(nq)]
    psi=tensor([basis(2,0)]*nq); feats=[x,y]
    for _ in range(reps):
        for q in range(nq):
            g=[qeye(2)]*nq; g[q]=_Ry(beta*(q+1)*feats[q%2]); psi=tensor(g)*psi
        if entangle:
            for q in range(nq): psi=CN[q]*psi
        for q in range(nq):
            g=[qeye(2)]*nq; g[q]=_Rz(beta*(q+1)*feats[q%2]); psi=tensor(g)*psi
    return psi.unit().full().ravel()

class FastEncoder:
    """Vectorised state-vector encoder (identical to feature_state_qutip to ~1e-16).
    Precomputes the constant ring-CNOT operator; applies rotation layers per point."""
    def __init__(self,nq=4,reps=2,beta=1.5,entangle=True):
        self.nq,self.reps,self.beta,self.ent=nq,reps,beta,entangle
        I=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],dtype=complex)
        P0=np.diag([1,0]).astype(complex); P1=np.diag([0,1]).astype(complex)
        def kron_list(L):
            M=L[0]
            for k in range(1,len(L)): M=np.kron(M,L[k])
            return M
        U=np.eye(2**nq,dtype=complex)
        for q in range(nq):
            a=[I]*nq; b=[I]*nq; a[q]=P0; b[q]=P1; b[(q+1)%nq]=X
            U=(kron_list(a)+kron_list(b))@U
        self.Ucnot=U if entangle else np.eye(2**nq,dtype=complex)
    def _rot_layer(self,isY,feats):
        # returns (npts, 2^nq, 2^nq) batched tensor-product of single-qubit rotations
        npts=feats.shape[0]; U=np.ones((npts,1,1),dtype=complex)
        for q in range(self.nq):
            th=self.beta*(q+1)*feats[:,q%2]
            c,s=np.cos(th/2),np.sin(th/2)
            if isY:
                g=np.zeros((npts,2,2),dtype=complex); g[:,0,0]=c; g[:,0,1]=-s; g[:,1,0]=s; g[:,1,1]=c
            else:
                e=np.exp(-1j*th/2); g=np.zeros((npts,2,2),dtype=complex); g[:,0,0]=e; g[:,1,1]=np.conj(e)
            U=np.einsum("nij,nkl->nikjl",U,g).reshape(npts,U.shape[1]*2,U.shape[2]*2)
        return U
    def states(self,P,chunk=1500):
        P=np.asarray(P,float)
        if len(P)>chunk:   # bounded memory for large point sets (6 qubits: 64x64 per point)
            return np.vstack([self.states(P[i:i+chunk],chunk) for i in range(0,len(P),chunk)])
        npts=len(P); dim=2**self.nq
        psi=np.zeros((npts,dim),dtype=complex); psi[:,0]=1
        for _ in range(self.reps):
            Ry=self._rot_layer(True,P); psi=np.einsum("nij,nj->ni",Ry,psi)
            psi=psi@self.Ucnot.T
            Rz=self._rot_layer(False,P); psi=np.einsum("nij,nj->ni",Rz,psi)
        return psi/np.linalg.norm(psi,axis=1,keepdims=True)

def fidelity_kernel(SA,SB): return np.abs(SA@SB.conj().T)**2

def kernel_diagnostics(K,jitter=1e-8):
    w=np.linalg.eigvalsh(K); wmin,wmax=w.min(),w.max()
    return dict(min_eig=float(wmin),max_eig=float(wmax),
                cond=float(wmax/max(wmin,1e-300)),
                cond_jitter=float((wmax+jitter)/(max(wmin,0)+jitter)),
                rank=int((w>1e-10*wmax).sum()),symmetric=bool(np.allclose(K,K.T)),
                unit_diag=bool(np.allclose(np.diag(K),1)))

# ----------------------------------------------------------------------
# 6. Random forest with spatial buffer-distance features (RFsp, Hengl 2018)
# ----------------------------------------------------------------------
def rfsp_features(P,anchors): return np.hstack([P,cdist(P,anchors)])

def fit_rf(Ptr,ztr,seed,n_anchors=20,inner_folds=3):
    k=min(n_anchors,len(Ptr))
    anchors=KMeans(k,n_init=3,random_state=seed).fit(Ptr).cluster_centers_
    Ftr=rfsp_features(Ptr,anchors)
    grid=[(leaf,mf) for leaf in (1,5) for mf in ("sqrt",1.0)]
    best=(np.inf,None); kf=KFold(inner_folds,shuffle=True,random_state=seed)
    for leaf,mf in grid:
        err=[]
        for a,b in kf.split(Ftr):
            m=RandomForestRegressor(100,min_samples_leaf=leaf,max_features=mf,random_state=seed,n_jobs=-1).fit(Ftr[a],ztr[a])
            err.append(np.mean((m.predict(Ftr[b])-ztr[b])**2))
        if np.mean(err)<best[0]: best=(np.mean(err),(leaf,mf))
    leaf,mf=best[1]
    model=RandomForestRegressor(300,min_samples_leaf=leaf,max_features=mf,random_state=seed,n_jobs=-1).fit(Ftr,ztr)
    return model,anchors,dict(min_samples_leaf=leaf,max_features=mf)

# ----------------------------------------------------------------------
# 7. Uncertainty scores
# ----------------------------------------------------------------------
def coverage95(y,m,s): return float(np.mean(np.abs(y-m)<=1.96*s))
def nlpd(y,m,s): return float(np.mean(0.5*np.log(2*np.pi*s**2)+0.5*((y-m)/s)**2))
def crps_gauss(y,m,s):
    z=(y-m)/s; return float(np.mean(s*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))))

# ----------------------------------------------------------------------
# 8. Nested cross-validation for all methods
# ----------------------------------------------------------------------
_STAGE1={}
NQ_GRID=(4,5,6); REPS_GRID=(2,3); BETA_GRID=(2.0,2.5,3.0,3.5); ALPHA_GRID=(0.0,0.25,0.5,0.75,1.0); ELL_GRID=(0.05,0.1)
METHODS=["OK","GP-RBF","GP-Matern","GP-ARD","RF","Quantum","Hybrid"]

def _inner_cv_select(Ptr,ztr,candidates,build,inner_folds=3,seed=0):
    """Generic inner CV: candidates -> (K_tr, K_ts builder). Selects by inner RMSE."""
    kf=KFold(inner_folds,shuffle=True,random_state=seed); best=(np.inf,None)
    for cand in candidates:
        err=[]
        for a,b in kf.split(Ptr):
            Ka,Kab,diag_b=build(cand,Ptr[a],Ptr[b])
            sf,sn,_=ml_fit_scale_noise(Ka,ztr[a])
            m,_=gp_fit_predict(Ka,Kab,diag_b,ztr[a],sf,sn)
            err.append(np.mean((m-ztr[b])**2))
        if np.mean(err)<best[0]: best=(np.mean(err),cand)
    return best[1]

def run_method(method,Ptr,ztr,Pte,scale_len,seed=0,nq=4,reps=2,entangle=True,
               beta_grid=(1.5,2.0,2.5,3.0,3.5,4.0),alpha_grid=(0.25,0.5,0.75),ell_grid=(0.05,0.1,0.2),
               inner_folds=3):
    """Train on (Ptr,ztr) with hyperparameters chosen INSIDE the training set; predict Pte.
    Coordinates in metres; kernels use coordinates scaled by scale_len (same for x,y).
    Returns mean, std, info dict."""
    info={}
    Str,Ste=Ptr/scale_len,Pte/scale_len
    if method=="OK":
        best,fits,_,_=select_variogram(Ptr,ztr)
        m,s=ordinary_kriging(Ptr,ztr,Pte,fits[best][0],best)
        info=dict(model=best,params=fits[best][0])
        return m,s,info
    if method in ("GP-RBF","GP-Matern","GP-ARD"):
        if method=="GP-RBF":   base=RBF(0.2,(1e-3,10.0))
        elif method=="GP-Matern": base=Matern(0.2,(1e-3,10.0),nu=1.5)
        else:                  base=RBF([0.2,0.2],(1e-3,10.0))     # anisotropic (ARD) RBF
        k=CK(np.var(ztr),(1e-3,1e3))*base+WhiteKernel(0.05*np.var(ztr),(1e-6,1e1))
        gp=GaussianProcessRegressor(k,normalize_y=True,n_restarts_optimizer=2,random_state=seed).fit(Str,ztr)
        m,s=gp.predict(Ste,return_std=True)
        info=dict(kernel=str(gp.kernel_))
        return m,s,info
    if method=="RF":
        model,anchors,hp=fit_rf(Ptr,ztr,seed,inner_folds=inner_folds)
        Fte=rfsp_features(Pte,anchors); m=model.predict(Fte)
        # quantile-free uncertainty proxy: spread across trees
        allp=np.stack([t.predict(Fte) for t in model.estimators_]); s=allp.std(axis=0)+1e-6
        info=dict(hyper=hp); return m,s,info
    if method in ("Quantum","Hybrid"):
        # Stage 1 (shared): select the register (nq, layers, beta) by inner CV on the quantum kernel alone
        key=(round(float(Str.sum()),6),len(Str),seed,tuple(NQ_GRID),tuple(REPS_GRID),tuple(BETA_GRID))
        if key not in _STAGE1:
            cands=[(q,r,b) for q in NQ_GRID for r in REPS_GRID for b in BETA_GRID]
            def build(c,A,B):
                q,r,b=c; enc=FastEncoder(q,r,b,entangle); SA,SB=enc.states(A),enc.states(B)
                return fidelity_kernel(SA,SA),fidelity_kernel(SB,SA),np.ones(len(B))
            _STAGE1[key]=_inner_cv_select(Str,ztr,cands,build,inner_folds,seed)
        q,r,b=_STAGE1[key]
        enc=FastEncoder(q,r,b,entangle); SA,SB=enc.states(Str),enc.states(Ste)
        KQ=fidelity_kernel(SA,SA); KQts=fidelity_kernel(SB,SA)
        if method=="Quantum":
            sf,sn,_=ml_fit_scale_noise(KQ,ztr)
            m,s=gp_fit_predict(KQ,KQts,np.ones(len(Ste)),ztr,sf,sn)
            return m,s,dict(nq=q,reps=r,beta=b,sig_f=sf,sig_n=sn,diag=kernel_diagnostics(KQ))
        # Stage 2: select the mixing weight and RBF scale (alpha=0 -> pure RBF, alpha=1 -> pure quantum)
        cands=[(a,l) for a in ALPHA_GRID for l in ELL_GRID]
        def build2(c,A,B):
            a,l=c; encA=FastEncoder(q,r,b,entangle); SA2,SB2=encA.states(A),encA.states(B)
            return a*fidelity_kernel(SA2,SA2)+(1-a)*rbf_corr(A,A,l), a*fidelity_kernel(SB2,SA2)+(1-a)*rbf_corr(B,A,l), np.ones(len(B))
        a,l=_inner_cv_select(Str,ztr,cands,build2,inner_folds,seed)
        K=a*KQ+(1-a)*rbf_corr(Str,Str,l); Kts=a*KQts+(1-a)*rbf_corr(Ste,Str,l)
        sf,sn,_=ml_fit_scale_noise(K,ztr)
        m,s=gp_fit_predict(K,Kts,np.ones(len(Ste)),ztr,sf,sn)
        return m,s,dict(nq=q,reps=r,beta=b,alpha=a,ell=l,sig_f=sf,sig_n=sn)
    raise ValueError(method)

def nested_cv(P,z,scale_len,methods=METHODS,outer=10,seed=0,**kw):
    kf=KFold(outer,shuffle=True,random_state=seed); out={}
    for meth in methods:
        yt,yp,ys,infos,t0=[],[],[],[],time.time()
        for tr,te in kf.split(P):
            m,s,info=run_method(meth,P[tr],z[tr],P[te],scale_len,seed=seed,**kw)
            yt.append(z[te]); yp.append(m); ys.append(s); infos.append(info)
        yt,yp,ys=map(np.concatenate,(yt,yp,ys))
        r=dict(rmse=float(np.sqrt(np.mean((yt-yp)**2))),mae=float(np.mean(np.abs(yt-yp))),
               r2=float(1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)),
               cov95=coverage95(yt,yp,ys),nlpd=nlpd(yt,yp,ys),crps=crps_gauss(yt,yp,ys),
               time=time.time()-t0,yt=yt,yp=yp,ys=ys,infos=infos)
        out[meth]=r
    return out

def summarize(runs,methods=METHODS,keys=("rmse","mae","r2","cov95","crps","nlpd","time")):
    """runs: list of nested_cv outputs (one per seed) -> mean/std table"""
    tab={}
    for m in methods:
        tab[m]={k:(np.mean([r[m][k] for r in runs]),np.std([r[m][k] for r in runs])) for k in keys}
    return tab

def paired_test(runs,a,b,key="rmse"):
    xa=np.array([r[a][key] for r in runs]); xb=np.array([r[b][key] for r in runs])
    if np.allclose(xa,xb): return 1.0,float(np.mean(xa-xb))
    try: p=wilcoxon(xa,xb).pvalue
    except ValueError: p=1.0
    return float(p),float(np.mean(xa-xb))


# ----------------------------------------------------------------------
# 9. Ablation of the quantum feature map (beta re-tuned per configuration)
# ----------------------------------------------------------------------
def ablation_quantum(P,z,scale_len,configs,outer=5,seed=0,beta_grid=(1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0)):
    """configs: list of dict(nq,reps,entangle). Returns {label: (rmse, r2, best_betas)}."""
    S=P/scale_len; kf=KFold(outer,shuffle=True,random_state=seed); out={}
    for cfg in configs:
        yt,yp,betas=[],[],[]
        for tr,te in kf.split(S):
            def build(beta,A,B):
                enc=FastEncoder(cfg["nq"],cfg["reps"],beta,cfg["entangle"]); SA,SB=enc.states(A),enc.states(B)
                return fidelity_kernel(SA,SA),fidelity_kernel(SB,SA),np.ones(len(B))
            beta=_inner_cv_select(S[tr],z[tr],beta_grid,build,3,seed)
            enc=FastEncoder(cfg["nq"],cfg["reps"],beta,cfg["entangle"]); SA,SB=enc.states(S[tr]),enc.states(S[te])
            K=fidelity_kernel(SA,SA); Kts=fidelity_kernel(SB,SA)
            sf,sn,_=ml_fit_scale_noise(K,z[tr]); m,_=gp_fit_predict(K,Kts,np.ones(len(te)),z[tr],sf,sn)
            yt.append(z[te]); yp.append(m); betas.append(beta)
        yt,yp=np.concatenate(yt),np.concatenate(yp)
        lab=f"nq={cfg['nq']}, L={cfg['reps']}, ent={'on' if cfg['entangle'] else 'off'}"
        out[lab]=dict(rmse=float(np.sqrt(np.mean((yt-yp)**2))),
                      r2=float(1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)),betas=betas)
    return out

# ----------------------------------------------------------------------
# 10. Normal-score transform (for skewed real data, e.g. magnetic residual)
# ----------------------------------------------------------------------
class NormalScore:
    """Rank-based Gaussian anamorphosis fitted on training values only."""
    def fit(self,v):
        self.sorted=np.sort(np.asarray(v,float)); n=len(self.sorted)
        self.q=norm.ppf((np.arange(1,n+1)-0.5)/n); return self
    def transform(self,v):
        return np.interp(v,self.sorted,self.q,left=self.q[0],right=self.q[-1])
    def inverse(self,y):
        return np.interp(y,self.q,self.sorted,left=self.sorted[0],right=self.sorted[-1])

# ----------------------------------------------------------------------
# 11. Non-stationarity diagnostic for a kernel: how much of the kernel value
#     at a given distance is explained by WHERE the pair sits (eta-squared)
# ----------------------------------------------------------------------
def kernel_location_dependence(P,K,nbins_h=12,nbins_loc=4):
    """Returns eta^2: fraction of residual kernel variance (after removing the
    mean kernel-vs-distance trend) explained by the location of the pair midpoint."""
    D=cdist(P,P); iu=np.triu_indices(len(P),1); d=D[iu]; k=K[iu]
    mid=0.5*(P[iu[0]]+P[iu[1]])
    hb=np.digitize(d,np.linspace(0,d.max()*1.0001,nbins_h+1))-1
    resid=np.empty_like(k)
    for b in np.unique(hb):
        m=hb==b; resid[m]=k[m]-k[m].mean()
    lx=np.digitize(mid[:,0],np.linspace(P[:,0].min(),P[:,0].max()*1.0001,nbins_loc+1))-1
    ly=np.digitize(mid[:,1],np.linspace(P[:,1].min(),P[:,1].max()*1.0001,nbins_loc+1))-1
    cell=lx*nbins_loc+ly
    ss_tot=np.sum((resid-resid.mean())**2); ss_between=0.0
    for c in np.unique(cell):
        m=cell==c; ss_between+=m.sum()*(resid[m].mean()-resid.mean())**2
    return float(ss_between/max(ss_tot,1e-12))
