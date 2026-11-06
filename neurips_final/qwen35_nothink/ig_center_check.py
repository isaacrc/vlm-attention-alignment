import sys, os, numpy as np
R='/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal'
sys.path.insert(0, R+'/task4_metric_suite'); sys.path.insert(0, R+'/shared'); sys.path.insert(0, R+'/task3_baselines')
import run_metric_suite_cv as M, saliency_metrics as SM, alignment_core as A
imgs=A.target_images()
yy,xx=np.mgrid[:256,:256]; d=np.hypot(yy-127.5,xx-127.5)
central=d<=np.sqrt(0.25*256*256/np.pi)  # central disc covering 25% of image
for task in ['social','describe']:
    sm=M.subject_maps(task,imgs); subs=sorted(sm)
    keys=[im for im in imgs if any(im in sm[s] for s in subs)]
    H=M.mean_maps(sm,subs,keys); prior=np.mean([H[k] for k in keys],0)
    fb={k:SM.fixation_binary(H[k]) for k in keys}
    other=np.clip(np.sum([fb[k] for k in keys],0),0,1)
    cc=[SM.cc(prior,H[k]) for k in keys]
    cen=[(fb[k]>0)[central].sum()/(fb[k]>0).sum() for k in keys]
    # leave-one-out prior: IG of full prior over LOO prior at fixations
    n=len(keys); ig_loo=[SM.info_gain(prior, fb[k], (prior*n-H[k])/(n-1)) for k in keys]
    pp=SM._prob(prior); top=np.sort(pp.ravel())[::-1]
    print(f'{task}: n_sub={len(subs)} prior-vs-image CC mean={np.mean(cc):.3f}; '
          f'share of top-5% pixels inside central 25% disc={np.mean(cen):.3f}; '
          f'union of top-5% masks covers {other.mean():.3f} of image; '
          f'prior leak (IG full vs LOO prior)={np.mean(ig_loo):.4f} bits; '
          f'prior mass in its own top 25% pixels={top[:len(top)//4].sum():.3f}', flush=True)
