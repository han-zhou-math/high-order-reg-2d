"""Plot exact-data controls; avoid harmonic roundoff levels in fitted rates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/paper_revision")
data = json.loads((root/"exact_data_reconstruction.json").read_text())
groups = {name:[row for row in data["rows"] if row["diagnostic_label"]==name]
          for name in ("laplace","stokes","stokes_exact_gradient")}
curves = [("Laplace",groups["laplace"],"laplace","#2667b0","o"),
          ("Pressure",groups["stokes"],"pressure","#be4e33","s"),
          ("Velocity, numerical pressure gradient",groups["stokes"],"velocity","#26763c","o"),
          ("Velocity, exact pressure gradient",groups["stokes_exact_gradient"],"velocity","#8055ac","s")]
fig, axes = plt.subplots(1,2,figsize=(9.2,3.55),constrained_layout=True)
fits={}
for i,(label,rows,field,color,marker) in enumerate(curves):
    ax=axes[0 if i<2 else 1]
    h=np.array([r["h_c"] for r in rows]);e=np.array([r[field]["inf"] for r in rows])
    valid=np.array([r["m_grid"] in ((128,256,512) if i<2 else (512,768,1024)) for r in rows])
    rate=float(np.polyfit(np.log(h[valid]),np.log(e[valid]),1)[0])
    fits[label]=dict(rate=rate,levels=[r["m_grid"] for r,v in zip(rows,valid) if v],field=field)
    ax.loglog(h,e,color=color,lw=1.3,marker=marker,ms=4,
              label=label+f" (fit {rate:.2f})")
    if i<2:
        floor=np.array([r["m_grid"]>=768 for r in rows])
        ax.scatter(h[floor],e[floor],facecolors="white",edgecolors=color,marker=marker,zorder=3,s=24)
    if i in (0,2):
        order=6 if i==0 else 4
        hs=np.array([h[-1],h[1]])
        es=e[1]*(hs/h[1])**order*.4
        ax.loglog(hs,es,"k--",lw=1,label=f"slope {order}")
for ax,title in zip(axes,("(a) Harmonic fields","(b) Biharmonic Stokes velocity")):
    ax.set_title(title,fontsize=10)
    ax.set_xlabel(r"Cartesian spacing $h_c$")
    ax.set_ylabel("Maximum error")
    ax.grid(True,which="both",alpha=.22)
    ax.legend(fontsize=7.1,loc="best",frameon=False)
fig.savefig(root/"exact_data_reconstruction.pdf")
fig.savefig(root/"exact_data_reconstruction.png",dpi=180)
(root/"exact_data_reconstruction_fits.json").write_text(json.dumps(fits,indent=2)+"\n")
print(json.dumps(fits,indent=2))
