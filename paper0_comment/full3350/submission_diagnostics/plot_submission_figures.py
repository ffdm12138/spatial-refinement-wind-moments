"""Plot the submission layouts from unchanged full-3350 statistical summaries.

Run: python plot_submission_figures.py --results FULL3350_RESULTS.json --out <new-directory>
Only renders Figures 1--4. No raw-data processing, fit, forecast or new statistics.
Figure S1 is retained from the full-period analysis deposition. The exact numerical
analysis is versioned at GitHub commit 40ac8ba02c84e8ada78a964c534a9e9b45787d69.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--results',type=Path,default=Path(__file__).resolve().with_name('FULL3350_RESULTS.json'))
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 names=['Fig1_variance_budget','Fig2_target_moments','Fig3_mean_and_structure','Fig4_transfer_diagnostic']
 if any((a.out/(n+'.pdf')).exists() for n in names):raise FileExistsError('Use a new output directory; existing figures will not be overwritten')
 data=json.loads(a.results.read_text(encoding='utf8'));S=data['cases']
 assert [s['case']for s in S]==list('ABC')
 assert all(s['window']==[970,4320] and s['times']==3350 for s in S)
 plt.rcParams.update({'font.size':10.2,'axes.labelsize':10.2,'axes.titlesize':10.4,'legend.fontsize':9.5,
 'xtick.labelsize':10.2,'ytick.labelsize':10.2,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 x=np.arange(3);numbers={}
 def save(fig,name,values):
  fig.savefig(a.out/(name+'.pdf'),bbox_inches='tight',pad_inches=.06,metadata={'Title':name,'Subject':'Unchanged full3350 data; readable submission layout'})
  fig.savefig(a.out/(name+'.png'),bbox_inches='tight',pad_inches=.06);plt.close(fig)
  numbers[name]=values
 fig,ax=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained')
 means={k:[s['states'][k]['mean']for s in S]for k in ['S_c','S_v','S_s']}
 for i,(key,label)in enumerate([('S_c',r'$S_c$: native d02'),('S_v',r'$S_v$: vector mean'),('S_s',r'$S_s$: scalar mean')]):
  ax[0].bar(x+(i-1)*.24,means[key],.24,label=label)
 ax[0].set(xticks=x,xticklabels=list('ABC'),ylabel='Period-mean wind speed\n(m s$^{-1}$)',title='a) Native and averaged winds',ylim=(0,5.5));ax[0].legend(loc='upper left')
 fractions={str(b):[100*s['scales'][str(b)]['within_fraction_mean']for s in S]for b in [3,9,27]}
 for i,b in enumerate([3,9,27]):
  bars=ax[1].bar(x+(i-1)*.24,fractions[str(b)],.24,label=f'{b} × {b}')
  ax[1].bar_label(bars,fmt='%.1f',fontsize=9.2,padding=2)
 ax[1].set(xticks=x,xticklabels=list('ABC'),ylabel='Within-block / total variance (%)',title='b) Variance below averaging scale',ylim=(0,70));ax[1].legend(ncols=3,loc='upper left',columnspacing=.8,handlelength=1.1)
 save(fig,names[0],{'mean_speeds':means,'variance_percent':fractions})
 fig,ax=plt.subplots(1,3,figsize=(7.2,3.4),layout='constrained')
 ax[0].bar([0,1],[47,43]);ax[0].set(xticks=[0,1],xticklabels=['(2,2,5)','(1,4,4)'],ylabel=r'$\overline{s^3}$ (m$^3$ s$^{-3}$)',title='a) Same first two\nmoments',ylim=(0,65))
 ax[0].text(.5,58,'Mean 3\nVariance 2',ha='center',va='top',fontsize=9.5)
 shares=[100*s['moments']['speed_variance_share']for s in S]
 ax[1].bar(x,shares,label=r'$\sigma_s^2$');ax[1].bar(x,100-np.array(shares),bottom=shares,label=r'$2S_vJ+J^2$')
 ax[1].set(xticks=x,xticklabels=list('ABC'),ylim=(0,136),ylabel='Share of $Q$ (%)',title='b) Component-variance\npartition');ax[1].legend(loc='upper left',ncols=1)
 errs={k:[100*s['moments'][k]for s in S]for k in ['L3','G3']}
 for i,k in enumerate(['L3','G3']):
  bars=ax[2].bar(x+(i-.5)*.3,errs[k],.3,label=r'$L_3$'if i==0 else r'$G_3$');ax[2].bar_label(bars,fmt='%.2f',fontsize=9.2,padding=2,rotation=90)
 ax[2].set(xticks=x,xticklabels=list('ABC'),ylim=(0,20.5),ylabel='Truncation error (%)',title='c) Local and averaged\nresponse errors');ax[2].legend(ncols=2,loc='upper left',columnspacing=.7,handlelength=1.1)
 save(fig,names[1],{'illustration_R3':[47,43],'speed_variance_percent':shares,'errors_percent':errs})
 fig,ax=plt.subplots(1,3,figsize=(7.2,3.6),layout='constrained',gridspec_kw={'width_ratios':[.92,1.1,1.]})
 residual={mode:[s['interpolation'][mode]['max_component_residual']for s in S]for mode in ['bilinear_M','bicubic_M']}
 for i,mode in enumerate(residual):
  ax[0].plot(x+(i-.5)*.12,residual[mode],marker='o'if i==0 else 's',linestyle='none',label='Bilinear-M'if i==0 else 'Bicubic-M')
 ax[0].axhline(1e-6,linestyle='--',linewidth=1,label='Tolerance')
 ax[0].set(yscale='log',xticks=x,xticklabels=list('ABC'),ylim=(1e-16,1e-2),ylabel='Maximum component\nresidual (m s$^{-1}$)',title='a) Mean constraint')
 ax[0].legend(loc='upper left',handlelength=1.1,labelspacing=.3,borderpad=.35)
 ratios={}
 for i,(key,label) in enumerate([('J_ratio',r'$J$'),('SD_ratio',r'$\sigma_s$')]):
  for j,mode in enumerate(['bilinear_M','bicubic_M']):
   vals=[s['interpolation'][mode][key]for s in S];ratios[key+'_'+mode]=vals
   ax[1].plot(x+(j-.5)*.16,vals,linestyle='-'if j==0 else '--',marker='o'if i==0 else 's',label=label+(' bilinear'if j==0 else ' bicubic'))
 ax[1].axhline(1,linestyle=':',linewidth=1)
 ax[1].set(xticks=x,xticklabels=list('ABC'),ylim=(0,1.55),ylabel='Predicted / reference\nperiod-mean amplitude',title='b) Subgrid statistics')
 ax[1].legend(loc='upper left',handlelength=1.4,labelspacing=.28,borderpad=.35)
 amps={k:[s['spatial'][k]['mean_amplification']for s in S]for k in ['D2_x','D2_y']}
 for i,(key,label) in enumerate([('D2_x','Columns'),('D2_y','Rows')]):
  bars=ax[2].bar(x+(i-.5)*.32,amps[key],.32,label=label);ax[2].bar_label(bars,fmt='%.1f',fontsize=9.5,padding=2,rotation=90)
 ax[2].set(xticks=x,xticklabels=list('ABC'),ylim=(0,62),ylabel='Structure-function\namplification',title='c) Spatial arrangement');ax[2].legend(loc='upper left',handlelength=1.1)
 save(fig,names[2],{'residual_mps':residual,'ratios':ratios,'structure_amplification':amps,'tolerance_mps':1e-6})
 fig,ax=plt.subplots(1,3,figsize=(7.2,3.6),layout='constrained')
 mu=[s['transfer']['mu_prediction_mean']/s['transfer']['mu_reference_mean']for s in S]
 sd=[s['transfer']['SD_amplitude_ratio']for s in S]
 for i,(vals,label)in enumerate([(mu,r'Mean $S_s$'),(sd,r'SD $\sigma_s$')]):
  bars=ax[0].bar(x+(i-.5)*.3,vals,.3,label=label);ax[0].bar_label(bars,fmt='%.2f',fontsize=9.5,padding=2,rotation=90)
 ax[0].axhline(1,linestyle=':',linewidth=1)
 ax[0].set(xticks=x,xticklabels=['A\n(fit)','B','C'],ylim=(0,1.65),ylabel='Predicted / reference\nperiod mean',title='a) Mean and spread');ax[0].legend(loc='upper left',handlelength=1.1,labelspacing=.25)
 dist=[s['transfer']['CDF_D']['mean']for s in S]
 bars=ax[1].bar(x,dist);ax[1].bar_label(bars,fmt='%.3f',fontsize=9.5,padding=3)
 ax[1].set(xticks=x,xticklabels=['A\n(fit)','B','C'],ylim=(0,.27),ylabel='Squared CDF distance\n(m s$^{-1}$)',title='b) Distribution error')
 components={k:[s['groups']['U4to6'][k]['mean']for s in S[1:]]for k in ['var_parallel','var_perp']}
 for i,(k,label)in enumerate([('var_parallel','Parallel'),('var_perp','Perpendicular')]):
  bars=ax[2].bar(np.arange(2)+(i-.5)*.3,components[k],.3,label=label);ax[2].bar_label(bars,fmt='%.2f',fontsize=9.5,padding=2,rotation=90)
 ax[2].set(xticks=[0,1],xticklabels=['B','C'],ylim=(0,20),ylabel='Component variance\n(m$^2$ s$^{-2}$)',title='c) Common coarse speed\n'+r'$4<S_c<6$ m s$^{-1}$')
 ax[2].legend(loc='upper left',handlelength=1.1,labelspacing=.25)
 save(fig,names[3],{'mean_speed_ratios':mu,'SD_ratios':sd,'CDF_D_mps':dist,'conditional_components':components})
 manifest={'source_sha256':hashlib.sha256(a.results.read_bytes()).hexdigest(),'source_window':[970,4320],
 'layout_only':True,'plotted_values':numbers,'numerical_analysis_commit':'40ac8ba02c84e8ada78a964c534a9e9b45787d69',
 'figures':{n+'.pdf':hashlib.sha256((a.out/(n+'.pdf')).read_bytes()).hexdigest()for n in names}}
 (a.out/'PLOTTED_VALUES.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
 print('FOUR_FIGURES_RENDERED',a.out,'NO_RECOMPUTATION')
if __name__=='__main__':main()
