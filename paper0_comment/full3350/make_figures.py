"""Figures from complete3350 verified case summaries; no fitted parameters change."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
D=Path(os.environ.get('COMMENT_WORKDIR', str(Path(__file__).resolve().parent/'generated_run'))).resolve();O=D/'figures';O.mkdir(exist_ok=True)
S=[json.loads((D/(c+'_SUMMARY.json')).read_text(encoding='utf-8')) for c in 'ABC']
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
X=np.arange(3)
def save(fig,name):
 fig.savefig(O/(name+'.pdf'),bbox_inches='tight',pad_inches=.06);fig.savefig(O/(name+'.png'),bbox_inches='tight',pad_inches=.06);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8.8,3.1),layout='constrained')
for i,(k,label) in enumerate([('S_c',r'$S_c$: native d02'),('S_v',r'$S_v$: vector-mean d05'),('S_s',r'$S_s$: scalar-mean d05')]):
 axs[0].bar(X+(i-1)*.24,[s['states'][k]['mean'] for s in S],.24,label=label)
axs[0].set(xticks=X,xticklabels=['A','B','C'],ylabel='Period-mean wind speed (m s$^{-1}$)',title='a  Native and averaged winds');axs[0].legend(loc='upper left',fontsize=7);axs[0].set_ylim(0,5.2)
for i,b in enumerate([3,9,27]):
 vals=[100*s['scales'][str(b)]['within_fraction_mean'] for s in S];bars=axs[1].bar(X+(i-1)*.24,vals,.24,label=f'{b} × {b}')
 axs[1].bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
axs[1].set(xticks=X,xticklabels=['A','B','C'],ylim=(0,67),ylabel='Within-block variance / total (%)',title='b  Variance not represented by block means');axs[1].legend(ncols=3,loc='upper left',fontsize=7)
save(fig,'Fig1_variance_budget')
fig,axs=plt.subplots(1,3,figsize=(9.3,2.9),layout='constrained')
axs[0].bar([0,1],[47,43]);axs[0].set(xticks=[0,1],xticklabels=['(2, 2, 5)','(1, 4, 4)'],ylabel=r'$\overline{s^3}$ (m$^3$ s$^{-3}$)',ylim=(0,60),title='a  Equal mean and variance');axs[0].text(.5,54,'Mean = 3; variance = 2',ha='center',fontsize=8)
v=np.array([s['moments']['speed_variance_share'] for s in S])*100;axs[1].bar(X,v,label=r'$\sigma_s^2$');axs[1].bar(X,100-v,bottom=v,label=r'$2S_vJ+J^2$');axs[1].set(xticks=X,xticklabels=list('ABC'),ylim=(0,125),ylabel='Share of horizontal component variance (%)',title='b  Partition of component variance');axs[1].legend(ncols=2,loc='upper left',fontsize=7)
for i,k in enumerate(['L3','G3']):
 bars=axs[2].bar(X+(i-.5)*.3,[100*s['moments'][k] for s in S],.3,label=r'$L_3$' if i==0 else r'$G_3$');axs[2].bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
axs[2].set(xticks=X,xticklabels=list('ABC'),ylabel='Third-moment truncation error (%)',ylim=(0,18),title='c  Local and mean-response errors');axs[2].legend(ncols=2,loc='upper left')
save(fig,'Fig2_target_moments')
fig,axs=plt.subplots(1,3,figsize=(9.3,3),layout='constrained')
for i,mode in enumerate(['bilinear_M','bicubic_M']):
 axs[0].plot(X+(i-.5)*.12,[s['interpolation'][mode]['max_component_residual'] for s in S],marker='o' if i==0 else 's',linestyle='none',label='Bilinear-M' if i==0 else 'Bicubic-M')
axs[0].axhline(1e-6,linestyle='--',linewidth=1,label='Tolerance');axs[0].set(yscale='log',xticks=X,xticklabels=list('ABC'),ylabel='Max component residual (m s$^{-1}$)',ylim=(1e-16,1e-4),title='a  Mean-component constraint');axs[0].legend(loc='upper left',fontsize=7)
for i,(key,label) in enumerate([('J_ratio',r'$J$'),('SD_ratio',r'$\sigma_s$')]):
 for j,mode in enumerate(['bilinear_M','bicubic_M']):
  axs[1].plot(X+(j-.5)*.16,[s['interpolation'][mode][key] for s in S],linestyle='-' if j==0 else '--',marker='o' if i==0 else 's',label=label+(' bilinear' if j==0 else ' bicubic'))
axs[1].axhline(1,linestyle=':',linewidth=1);axs[1].set(xticks=X,xticklabels=list('ABC'),ylim=(0,1.35),ylabel='Predicted / reference period mean',title='b  Subgrid amplitude remains low');axs[1].legend(loc='upper left',fontsize=6.5,ncols=2)
for i,(key,label) in enumerate([('D2_x','Array columns'),('D2_y','Array rows')]):
 vals=[s['spatial'][key]['mean_amplification'] for s in S];bars=axs[2].bar(X+(i-.5)*.32,vals,.32,label=label);axs[2].bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
axs[2].set(xticks=X,xticklabels=list('ABC'),ylabel='Structure-function amplification',title='c  Fixed within-cell permutations');axs[2].legend(loc='upper left',fontsize=7);axs[2].set_ylim(0,max(axs[2].get_ylim()[1]*1.25,50))
save(fig,'Fig3_mean_and_structure')
fig,axs=plt.subplots(1,3,figsize=(9.3,3.1),layout='constrained')
mu=[s['transfer']['mu_prediction_mean']/s['transfer']['mu_reference_mean'] for s in S];sd=[s['transfer']['SD_amplitude_ratio'] for s in S]
for i,(values,label) in enumerate([(mu,r'Mean speed $S_s$'),(sd,r'Speed SD $\sigma_s$')]):
 bars=axs[0].bar(X+(i-.5)*.3,values,.3,label=label);axs[0].bar_label(bars,fmt='%.2f',padding=2,fontsize=7)
axs[0].axhline(1,linestyle=':',linewidth=1);axs[0].set(xticks=X,xticklabels=['A (fit)','B','C'],ylim=(0,1.4),ylabel='Predicted / reference period mean',title='a  Source-calibrated parameterization');axs[0].legend(loc='upper left',fontsize=7)
bars=axs[1].bar(X,[s['transfer']['CDF_D']['mean'] for s in S]);axs[1].bar_label(bars,fmt='%.3f',padding=2,fontsize=8);axs[1].set(xticks=X,xticklabels=['A (fit)','B','C'],ylim=(0,.26),ylabel=r'Integrated squared CDF distance (m s$^{-1}$)',title='b  Distributional error')
for i,(key,label) in enumerate([('var_parallel','Parallel'),('var_perp','Perpendicular')]):
 vals=[s['groups']['U4to6'][key]['mean'] for s in S[1:]];bars=axs[2].bar(np.arange(2)+(i-.5)*.3,vals,.3,label=label);axs[2].bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
axs[2].set(xticks=[0,1],xticklabels=['B','C'],ylim=(0,18),ylabel='Horizontal component variance (m$^2$ s$^{-2}$)',title=r'c  d05 variability at $4<S_c<6$');axs[2].legend(loc='upper left',fontsize=7)
save(fig,'Fig4_transfer_diagnostic')
fig,axs=plt.subplots(3,1,figsize=(8.5,6.4),sharex=True,layout='constrained')
for ax,case in zip(axs,'ABC'):
 rows=list(csv.DictReader((D/(case+'_hourly.csv')).open(encoding='utf-8')));x=[float(r['elapsed_hour']) for r in rows]
 ax.plot(x,[float(r['sigma_s']) for r in rows],label='d05 reference',linewidth=1.4);ax.plot(x,[float(r['SD_pred']) for r in rows],label='Frozen parameterization',linestyle='--',linewidth=1.3)
 ax.set(ylabel=r'Mean $\sigma_s$ (m s$^{-1}$)',title=f'{case}  '+({'A':'Yakou, September (source fit)','B':'Yakou, December','C':'Ebao, December'}[case]));ax.legend(loc='upper right',ncols=2,fontsize=7);ax.set_ylim(bottom=0)
axs[-1].set(xlabel='Hours since start of retained analysis (nominal one-minute sampling)',xlim=(0,3350/60));save(fig,'FigS1_temporal_variability')
print('FIGURES',[(p.name,p.stat().st_size) for p in O.glob('*.pdf')])
