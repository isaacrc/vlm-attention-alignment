"""The four Kastner2015 stream masks on fsaverage5, one colour each (panel A)."""
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
from nilearn import plotting, datasets as nidatasets

HERE=Path(__file__).parent
PLOTS=Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/plots")
V=10242
SPEC=[("early","Early visual","#E69F00"),("ventral","Ventral","#009E73"),
      ("lateral","Lateral","#CC79A7"),("dorsal","Dorsal","#0072B2")]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--panel-letter",default=None)
    ap.add_argument("--width",type=float,default=23.0); a=ap.parse_args()
    m=np.load(HERE/"rois"/"kastner_stream_masks_fs5.npz")
    lab=np.zeros(2*V,dtype=np.int32)
    for i,(k,_,_) in enumerate(SPEC,start=1): lab[m[k].astype(bool)]=i
    fs5=nidatasets.fetch_surf_fsaverage("fsaverage5")
    sulc={"left":fs5.get("sulc_left",fs5.get("curv_left")),"right":fs5.get("sulc_right",fs5.get("curv_right"))}
    pial={"left":fs5["pial_left"],"right":fs5["pial_right"]}
    half={"left":lab[:V],"right":lab[V:]}
    cmap=ListedColormap(["#E8E6E1"]+[to_rgb(c) for _,_,c in SPEC])
    views=[("left","lateral"),("left","medial"),("right","lateral"),("right","medial")]
    fig,axes=plt.subplots(1,4,figsize=(a.width,6*a.width/22),subplot_kw={"projection":"3d"})
    fig.subplots_adjust(left=0.01,right=0.99,top=1.02,bottom=0.14,wspace=0.01)
    for ax,(hemi,view) in zip(axes,views):
        plotting.plot_surf_roi(surf_mesh=pial[hemi],roi_map=half[hemi],hemi=hemi,view=view,
                               bg_map=sulc[hemi],cmap=cmap,vmin=0,vmax=len(SPEC),
                               avg_method="median",axes=ax,colorbar=False)
        ax.set_title(f"{hemi[0].upper()}H {view}",fontsize=22,fontweight="bold",y=0.02)
    n={k:int(m[k].sum()) for k,_,_ in SPEC}
    fig.legend(handles=[Patch(facecolor=c,edgecolor="black",label=f"{lbl} ({n[k]})") for k,lbl,c in SPEC],
               loc="lower center",ncol=4,fontsize=20,frameon=False,bbox_to_anchor=(0.5,-0.01))
    if a.panel_letter:
        import sys; sys.path.insert(0,str(HERE.parent/"shared")); import figstyle as FST
        FST.panel_letter(fig,a.panel_letter)
    out=PLOTS/"nsd_kastner_stream_atlas.png"
    fig.savefig(out,dpi=200,bbox_inches="tight",facecolor="white"); print("saved",out)

if __name__=="__main__": main()
