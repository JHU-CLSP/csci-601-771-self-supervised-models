"""Reproduce the original toy-model SVG; requires NumPy and Matplotlib."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
p=np.linspace(.00001,.99999,2000)
kl=p*np.log(2*p)+(1-p)*np.log(2*(1-p))
fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
for beta,color,label in [(0,'#6c7480','Reward only (β = 0)'),(1,'#002d72','Reward − KL (β = 1)'),(2,'#008578','Reward − 2 KL (β = 2)')]:
 f=1+2*p-beta*kl
 ax.plot(p,f,color=color,lw=2.4,label=label)
 if beta:
  peak=1/(1+np.exp(-2/beta))
  kk=peak*np.log(2*peak)+(1-peak)*np.log(2*(1-peak))
  ax.plot(peak,1+2*peak-beta*kk,'o',color=color,ms=7)
  ax.annotate(f'P(A) ≈ {peak:.3f}',(peak,1+2*peak-beta*kk),xytext=(-99,17 if beta==1 else -36),textcoords='offset points',color=color,arrowprops={'arrowstyle':'-','color':color},fontsize=11)
ax.axvline(.5,color='#6c7480',ls=':',lw=1.4)
ax.text(.5,-.39,' Reference P(A) = 0.5',va='bottom',ha='left',fontsize=10,color='#505c68')
ax.set(xlim=(0,1),ylim=(-.48,3.18),xlabel='Current policy probability P(A)',ylabel='Expected reward minus KL cost')
ax.set_xticks(np.linspace(0,1,6));ax.grid(alpha=.18);ax.legend(loc='upper left',frameon=False,fontsize=11)
fig.suptitle('Same reward scores; different cost for leaving the reference',fontsize=14,fontweight='bold')
output = Path(__file__).with_name('19-kl-tradeoff.svg')
fig.savefig(output, bbox_inches='tight')
output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')
for beta in [1,2]:
 peak=1/(1+np.exp(-2/beta));print(beta,peak,1+2*peak-beta*(peak*np.log(2*peak)+(1-peak)*np.log(2*(1-peak))))
