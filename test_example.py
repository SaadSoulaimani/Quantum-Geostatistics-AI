"""Quick test for Quantum Geostatistics library (run: python test_example.py)."""
import sys, numpy as np
import qgeo as qg
def check(n,c): print(f"[{'PASS' if c else 'FAIL'}] {n}"); return bool(c)
ok=True
pts=np.random.default_rng(0).random((6,2))
S=qg.FastEncoder(4,2,3.0).states(pts); Sq=np.array([qg.feature_state_qutip(x,y,4,2,3.0) for x,y in pts])
ok&=check("vectorised encoder matches QuTiP reference", np.max(np.abs(1-np.abs(np.sum(S*Sq.conj(),1))**2))<1e-9)
K=qg.fidelity_kernel(S,S); d=qg.kernel_diagnostics(K)
ok&=check("fidelity kernel symmetric, unit diagonal, PSD", d["symmetric"] and d["unit_diag"] and d["min_eig"]>-1e-8)
nx,ny,dx=43,105,28.8; xs,ys,grid=qg.make_grid(nx,ny,dx); L0=max(xs.max(),ys.max())
F=qg.grf_gaussian(nx,ny,dx,300,7); z_true=F.ravel(); idx=qg.sample_design(grid,80,"random",7); P=grid[idx]
z=z_true[idx]+np.random.default_rng(7).normal(0,0.08,80)
best,fits,_,_=qg.select_variogram(P,z); ok&=check(f"variogram selection on a Gaussian field picks 'gaussian' (got {best})", best=="gaussian")
res=qg.nested_cv(P,z,L0,methods=["OK","Quantum","Hybrid"],outer=5,seed=0)
for m in res: print(f"       {m:8s} R2={res[m]['r2']:.3f} cov95={res[m]['cov95']:.2f}")
ok&=check("all R2 in (0,1]", all(0<res[m]['r2']<=1 for m in res))
ok&=check("hybrid at least as accurate as quantum alone", res["Hybrid"]["rmse"]<=res["Quantum"]["rmse"]+0.05)
print("\nAll checks passed." if ok else "\nSome checks FAILED."); sys.exit(0 if ok else 1)
