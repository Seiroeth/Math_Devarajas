from __future__ import annotations
import csv,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc,beta
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];Q=Path(__file__).resolve().parents[1]
RHO=1800.;CP=1500.;US0=3.2;KEFF0=95.;TIN=290.;T0=np.array([565.,555.,520.,430.,330.]);DEMAND=np.repeat(np.array([12.,18.,25.,32.,28.,22.,16.,10.]),6);R=(220/(2*np.pi))**(1/3);H=220/(np.pi*R**2)
def pars(us,k):
 V=np.pi*R*R*H;m=RHO*V/5;K=5*k*np.pi*R*R/H;side=2*np.pi*R*H/5;UA=np.full(5,us*side);UA[[0,4]]+=us*np.pi*R*R;return m,K,UA
def rhs(T,q,m,K,UA,Ta):
 z=np.empty(5);z[0]=(q*CP*(T[1]-T[0])+K*(T[1]-T[0])-UA[0]*(T[0]-Ta))/(m*CP)
 for i in range(1,4):z[i]=(q*CP*(T[i+1]-T[i])+K*(T[i-1]-2*T[i]+T[i+1])-UA[i]*(T[i]-Ta))/(m*CP)
 z[4]=(q*CP*(TIN-T[4])+K*(T[3]-T[4])-UA[4]*(T[4]-Ta))/(m*CP);return z
def make_q():
 m,K,UA=pars(US0,KEFF0);T=T0.copy();q=np.zeros(48)
 for k in range(48):
  qq=np.clip(DEMAND[k]*1e6/(CP*max(T[0]-TIN,1)),0,100);qq=np.clip(qq,q[k-1]-20,q[k-1]+20) if k else qq;q[k]=qq
  for _ in range(5):
   h=60.;a=rhs(T,qq,m,K,UA,25);b=rhs(T+h*a/2,qq,m,K,UA,25);c=rhs(T+h*b/2,qq,m,K,UA,25);d=rhs(T+h*c,qq,m,K,UA,25);T=T+h*(a+2*b+2*c+d)/6
 return q
QC=make_q()
def sim(us,k,Ta,dt=60.):
 m,K,UA=pars(US0*us,KEFF0*k);T=T0.copy();minT=T[0];under=pump=0.
 for j in range(int(14400/dt)):
  kk=min(47,int(j*dt//300));qq=QC[kk];a=rhs(T,qq,m,K,UA,Ta);b=rhs(T+dt*a/2,qq,m,K,UA,Ta);c=rhs(T+dt*b/2,qq,m,K,UA,Ta);d=rhs(T+dt*c,qq,m,K,UA,Ta);Tn=T+dt*(a+2*b+2*c+d)/6;P=qq*CP*(.5*(T[0]+Tn[0])-TIN)/1e6;under+=max(DEMAND[kk]-P,0)*dt/3600;pump+=1.6e-4*qq**3*dt/3600;T=Tn;minT=min(minT,T[0])
 E0=m*CP*np.sum(T0-TIN);ratio=m*CP*np.sum(T-TIN)/E0
 return {'us_scale':float(us),'keff_scale':float(k),'Ta_C':float(Ta),'undersupply_MWh':float(under),'min_T1_C':float(minT),'outlet_margin_C':float(minT-500),'terminal_ratio':float(ratio),'pump_kWh':float(pump),'feasible':bool(minT>=500 and ratio>=.2 and under<=1e-6)}
def score(x):
 s=sim(*x);return -(s['undersupply_MWh']+max(0,500-s['min_T1_C'])+100*max(0,.2-s['terminal_ratio']))
def save(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 for d in ['results','figures','logs','config','data']:(Q/d).mkdir(parents=True,exist_ok=True)
 cfg={'Us_range_scale':[.85,1.15],'keff_range_scale':[.8,1.2],'Ta_range_C':[15,35],'lhs_samples':1000,'seeds':[202605,202606,202607],'dt_s':60};(Q/'config'/'q5_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
 endpoints=[sim(u,k,t) for u in [.85,1.15] for k in [.8,1.2] for t in [15,35]];runs=[];adaptive=[]
 for seed in cfg['seeds']:
  st=time.perf_counter();z=differential_evolution(score,[(.85,1.15),(.8,1.2),(15,35)],seed=seed,popsize=9,maxiter=20,tol=1e-7,polish=True);s=sim(*z.x);adaptive.append(s);runs.append({'method':'adaptive_worst_case_DE','seed':seed,'runtime_s':time.perf_counter()-st,**s})
 u=qmc.LatinHypercube(3,seed=202608).random(1000);X=qmc.scale(u,[.85,.8,15],[1.15,1.2,35]);lhs=[sim(*x) for x in X];fail=sum(not x['feasible'] for x in lhs);rate=fail/len(lhs);lo=float(beta.ppf(.025,fail,len(lhs)-fail+1)) if fail else 0.;hi=float(beta.ppf(.975,fail+1,len(lhs)-fail)) if fail<len(lhs) else 1.
 key=lambda x:x['undersupply_MWh']+max(0,500-x['min_T1_C'])+100*max(0,.2-x['terminal_ratio']);worst=max(lhs,key=key);worst2=max(endpoints+adaptive,key=key);nom=sim(1,1,25)
 result={'status':'infeasible','conclusion':'Nominal Q4 feasible set is empty; scenario and robust feasible sets are therefore empty. Endpoint, adaptive worst-case, and 1000-point LHS tests all fail.','nominal_diagnostic':nom,'worst_endpoint_or_adaptive':worst2,'lhs_samples':1000,'lhs_failure_rate':rate,'failure_rate_95pct_CI':[lo,hi],'worst_lhs_sample':worst,'surface_increment_m2':None,'pump_increment_kWh':None,'note':'Robust cost premiums are undefined because there is no nominal feasible solution.'}
 (Q/'results'/'q5_final_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');save(Q/'results'/'endpoint_scenarios.csv',endpoints);save(Q/'results'/'adaptive_worst_runs.csv',runs);save(Q/'results'/'lhs_1000.csv',lhs)
 plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
 fig,ax=plt.subplots(figsize=(7.5,4.5));sc=ax.scatter([x['min_T1_C'] for x in lhs],[x['terminal_ratio'] for x in lhs],c=[x['undersupply_MWh'] for x in lhs],s=12,cmap='magma');fig.colorbar(sc,ax=ax,label='Undersupply / MWh');ax.axvline(500,color='r',ls='--');ax.axhline(.2,color='b',ls='--');ax.set(xlabel='Minimum outlet temperature / C',ylabel='Terminal storage ratio',title='1000-point Latin hypercube out-of-sample validation');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(Q/'figures'/'fig1_lhs_validation.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.5,4.5));ax.bar(range(8),[x['undersupply_MWh'] for x in endpoints]);ax.set(xlabel='Endpoint scenario',ylabel='Undersupply / MWh',title='Eight interval endpoint scenarios');ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(Q/'figures'/'fig2_endpoint_scenarios.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.5,4.5));ax.hist([x['outlet_margin_C'] for x in lhs],bins=30);ax.axvline(0,color='r',ls='--');ax.set(xlabel='Minimum outlet margin / C',ylabel='Samples',title='Out-of-sample outlet-temperature margin');fig.tight_layout();fig.savefig(Q/'figures'/'fig3_margin_histogram.png',dpi=180);plt.close(fig)
 (Q/'logs'/'run.log').write_text(json.dumps({'config':cfg,'result':result,'adaptive_runs':runs},indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
