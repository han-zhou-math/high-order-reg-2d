"""Isolate Cartesian consistency errors using smooth harmonic/biharmonic data."""
import json
from pathlib import Path
import numpy as np
from experiments.allgrid import LaplaceProblem, StokesProblem
from experiments.matched_allgrid import run
from experiments.swapped_low_order import LAPLACE_KITE, STOKES_STARFISH

K = 8.0
def harmonic(points):
    x, y = np.asarray(points)[..., 0], np.asarray(points)[..., 1]
    ex = np.exp(K * (x - .895))
    return ex * np.cos(K*y), ex * np.sin(K*y)

def lexact(points):
    a, _ = harmonic(points)
    return np.where(LAPLACE_KITE.inside(points), a, 0.)

def sexact(points):
    a, b = harmonic(points)
    y = np.asarray(points)[..., 1]
    u = np.stack((a/K-y*b, -y*a), axis=-1)
    inside = STOKES_STARFISH.inside(points)
    return np.where(inside[...,None],u,0.),np.where(inside,-2*a,0.)

def sgradient(points):
    a, b = harmonic(points)
    g = np.stack((-2*K*a,2*K*b),axis=-1)
    return np.where(STOKES_STARFISH.inside(points)[...,None],g,0.)

laplace = LaplaceProblem(name=LAPLACE_KITE.name, geometry=LAPLACE_KITE.geometry,
    inside=LAPLACE_KITE.inside, boundary_densities=LAPLACE_KITE.boundary_densities,
    exact=lexact, box=LAPLACE_KITE.box)
stokes = StokesProblem(name=STOKES_STARFISH.name, geometry=STOKES_STARFISH.geometry,
    inside=STOKES_STARFISH.inside, boundary_data=STOKES_STARFISH.boundary_data,
    exact=sexact, exact_gradient=sgradient, box=STOKES_STARFISH.box)
rows=[]
for m in (64,128,256,512,768,1024):
    for label, problem, mode in (("laplace",laplace,"exact_direct"),
                                  ("stokes",stokes,"exact_direct"),
                                  ("stokes_exact_gradient",stokes,"exact_gradient")):
        report,_ = run(problem,m,mode=mode)
        report["diagnostic_label"]=label
        report["manufactured_scale"]=K
        rows.append(report)
        print(label,m,{key:report[key] for key in (("laplace",) if label=="laplace" else ("pressure","velocity"))},flush=True)
out=Path("results/paper_revision/exact_data_reconstruction.json")
out.write_text(json.dumps(dict(k=K,
    fields="A=exp(k(x-.895))cos(ky), B=exp(k(x-.895))sin(ky); Laplace v=A; Stokes psi=y A/k, p=-2A; outside fields zero",
    source="All prescribed direct and outer values exact; original stencil masks and numerical pressure gradient retained",
    rows=rows),indent=2)+"\n")
