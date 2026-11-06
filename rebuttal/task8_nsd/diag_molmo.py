"""Why is Molmo's Describe band negative over CLIP? Test whether the attention map is
redundant with CLIP: cross-validated R^2 of predicting each attention band FROM CLIP."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
HERE=Path('.')
X_clip=np.load(HERE/"features"/"X_clip_nsd.npy")
stim=pd.read_csv(HERE/"stimuli_index.csv"); ids=stim["nsdId"].to_numpy()
lut={int(i):k for k,i in enumerate(ids)}

def band(model,task,rule):
    if model=="Molmo" and rule=="cvbehav":
        d=np.load(HERE/"attn"/f"Molmo_{task}_cvbehav.npz"); return d["nsdId"],d["alphas"]
    d=np.load(HERE/"attn"/f"{model}_{task}.npz",allow_pickle=True)
    if model=="Molmo": return d["nsdId"],d["heads"][:,26,:]      # L10H27
    return d["nsdId"],d["alphas"]

def redundancy(nsd,feats):
    rows=[lut[int(i)] for i in nsd if int(i) in lut]
    keep=[k for k,i in enumerate(nsd) if int(i) in lut]
    C=X_clip[rows]; A=feats[keep]
    C=(C-C.mean(0))/(C.std(0)+1e-8); A=(A-A.mean(0))/(A.std(0)+1e-8)
    kf=KFold(5,shuffle=True,random_state=0); sc=[]
    for tr,te in kf.split(C):
        m=RidgeCV(alphas=np.logspace(0,6,7)).fit(C[tr],A[tr])
        p=m.predict(C[te])
        sc.append(np.mean([np.corrcoef(p[:,j],A[te][:,j])[0,1]**2 for j in range(A.shape[1])]))
    return float(np.mean(sc))

print("How much of each attention band is predictable from CLIP (CV R^2)?\n")
for lab,(m,t,r) in {"Molmo describe L8H7  (CV head)":("Molmo","describe","cvbehav"),
                    "Molmo describe L10H27 (argmax)":("Molmo","describe","best"),
                    "Molmo social  L10H24":("Molmo","social","best"),
                    "CNN-LSTM describe":("CNN_LSTM","describe","x"),
                    "CNN-LSTM social":("CNN_LSTM","social","x")}.items():
    nsd,f=band(m,t,r)
    print(f"  {lab:32s} {redundancy(nsd,f):.3f}")
