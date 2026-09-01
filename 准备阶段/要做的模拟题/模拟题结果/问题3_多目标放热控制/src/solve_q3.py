from __future__ import annotations

import csv, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
Q = Path(__file__).resolve().parents[1]
PARAM = ROOT / "模拟题结果" / "问题1_参数辨识与预测" / "results" / "identified_parameters.json"
CFG = Q / "config" / "q3_config.json"

T0=np.array([565.,555.,520.,430.,330.]); TIN=290.; TA=25.; CP=1500.; MASS=56548.66776461628
DEMAND=np.repeat(np.array([12.,18.,25.,32.,28.,22.,16.,10.]),6) # MW, 48*5 min
DT=60.; NSTEPS=240

def rhs(T,q,K,UA):
    z=np.empty(5)
    z[0]=(q*CP*(T[1]-T[0])+K*(T[1]-T[0])-UA*(T[0]-TA))/(MASS*CP)
    for i in range(1,4):
        z[i]=(q*CP*(T[i+1]-T[i])+K*(T[i-1]-2*T[i]+T[i+1])-UA*(T[i]-TA))/(MASS*CP)
    z[4]=(q*CP*(TIN-T[4])+K*(T[3]-T[4])-UA*(T[4]-TA))/(MASS*CP)
    return z

def schedule_sim(q,K,UA,dt=60.):
    n=int(round(14400/dt)); T=np.empty((n+1,5)); P=np.empty(n); PP=np.empty(n); T[0]=T0
    for j in range(n):
        k=min(47,int((j*dt)//300)); qq=float(q[k]); x=T[j]
        k1=rhs(x,qq,K,UA); k2=rhs(x+dt*k1/2,qq,K,UA); k3=rhs(x+dt*k2/2,qq,K,UA); k4=rhs(x+dt*k3,qq,K,UA)
        T[j+1]=x+dt*(k1+2*k2+2*k3+k4)/6
        P[j]=qq*CP*(0.5*(T[j,0]+T[j+1,0])-TIN)/1e6
        PP[j]=1.6e-4*qq**3
    tt=np.arange(n+1)*dt
    return tt,T,P,PP

def metrics(q,K,UA,dt=60.):
    tt,T,P,PP=schedule_sim(q,K,UA,dt); demand=np.repeat(DEMAND,int(round(300/dt)))
    err=P-demand
    return dict(rmse_mw=float(np.sqrt(np.mean(err**2))),max_abs_deviation_mw=float(np.max(abs(err))),
        undersupply_mwh=float(np.sum(np.maximum(-err,0))*dt/3600),pump_energy_kwh=float(np.sum(PP)*dt/3600),
        min_T1_c=float(T[:,0].min()),temperature_margin_c=float(T[:,0].min()-500),
        ramp_violation=float(max(0,np.max(np.abs(np.diff(q)))-20)),q=q,tt=tt,T=T,P=P,PP=PP)

def penalty_objective(q,K,UA,w=.7):
    m=metrics(q,K,UA); norm_e=m['rmse_mw']/25; norm_p=m['pump_energy_kwh']/1500
    viol=max(0,-m['temperature_margin_c'])/20 + m['ramp_violation']/20
    return w*norm_e+(1-w)*norm_p+1e3*viol**2

def heuristic(K,UA):
    q=np.zeros(48); T=T0.copy()
    for k in range(48):
        guess=np.clip(DEMAND[k]*1e6/(CP*max(T[0]-TIN,1)),0,100)
        if k: guess=np.clip(guess,q[k-1]-20,q[k-1]+20)
        q[k]=guess
        for _ in range(5):
            h=60.; a=rhs(T,guess,K,UA); b=rhs(T+h*a/2,guess,K,UA); c=rhs(T+h*b/2,guess,K,UA); d=rhs(T+h*c,guess,K,UA); T=T+h*(a+2*b+2*c+d)/6
    return q

def nondominated(points):
    keep=[]
    for i,p in enumerate(points):
        dominated=False
        for j,r in enumerate(points):
            if i!=j and r[0]<=p[0] and r[1]<=p[1] and (r[0]<p[0] or r[1]<p[1]): dominated=True;break
        if not dominated: keep.append(i)
    return keep

def evolutionary(K,UA,seed):
    rng=np.random.default_rng(seed); base=heuristic(K,UA); pop=[]
    for _ in range(24):
        x=np.clip(base+rng.normal(0,15,48),0,100)
        for k in range(1,48): x[k]=np.clip(x[k],x[k-1]-20,x[k-1]+20)
        pop.append(x)
    for _ in range(14):
        vals=[]
        for x in pop:
            m=metrics(x,K,UA); vals.append((m['rmse_mw']+100*max(0,-m['temperature_margin_c']),m['pump_energy_kwh']))
        nd=nondominated(vals); elite=[pop[i] for i in nd] or [pop[int(np.argmin([v[0] for v in vals]))]]
        new=elite[:8]
        while len(new)<24:
            p=elite[rng.integers(len(elite))].copy(); p+=rng.normal(0,8,48); p=np.clip(p,0,100)
            for k in range(1,48): p[k]=np.clip(p[k],p[k-1]-20,p[k-1]+20)
            new.append(p)
        pop=new
    raw=[]
    for x in pop:
        m=metrics(x,K,UA); raw.append((m['rmse_mw'],m['pump_energy_kwh'],m['temperature_margin_c'],x))
    feasible=[v for v in raw if v[2]>=0]
    pool=feasible if feasible else raw
    return min(pool,key=lambda v:v[0]+v[1]/1500+100*max(0,-v[2]))[3], raw

def save_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    for d in ['results','figures','logs','config','data']: (Q/d).mkdir(parents=True,exist_ok=True)
    p=json.loads(PARAM.read_text(encoding='utf-8')); K=float(p['K_W_per_K']); UA=float(p['UA_W_per_K'])
    cfg={'K_W_K':K,'UA_W_K':UA,'initial_C':T0.tolist(),'Tin_C':TIN,'Ta_C':TA,'control_intervals':48,'interval_s':300,'optimization_dt_s':60,'verification_dt_s':10,'seeds':[202603,202604,202605]}
    CFG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    q0=np.zeros(48); cert=metrics(q0,K,UA,10.)
    runs=[]; candidates=[('零流量可行性证书',q0)]
    base=heuristic(K,UA); candidates.append(('分段需求反演/MPC启发式',base))
    for seed in cfg['seeds']:
        t=time.perf_counter(); res=minimize(lambda x:penalty_objective(x,K,UA,.7),base,method='SLSQP',bounds=[(0,100)]*48,
            constraints={'type':'ineq','fun':lambda x:20-np.abs(np.diff(x))},options={'maxiter':90,'ftol':1e-8,'disp':False}); elapsed=time.perf_counter()-t
        m=metrics(res.x,K,UA); runs.append({'algorithm':'加权和-SLSQP','seed':seed,'runtime_s':elapsed,'success':bool(res.success),'rmse_mw':m['rmse_mw'],'pump_kwh':m['pump_energy_kwh'],'min_T1_c':m['min_T1_c'],'constraint_violation_c':max(0,-m['temperature_margin_c'])})
        candidates.append((f'加权和-SLSQP(seed={seed})',res.x))
        t=time.perf_counter(); x,raw=evolutionary(K,UA,seed);elapsed=time.perf_counter()-t;m=metrics(x,K,UA)
        runs.append({'algorithm':'多目标进化(NSGA-II结构)','seed':seed,'runtime_s':elapsed,'success':m['temperature_margin_c']>=0,'rmse_mw':m['rmse_mw'],'pump_kwh':m['pump_energy_kwh'],'min_T1_c':m['min_T1_c'],'constraint_violation_c':max(0,-m['temperature_margin_c'])});candidates.append((f'多目标进化(seed={seed})',x))
    diag=[]
    for name,x in candidates:
        m=metrics(x,K,UA,10.);diag.append({'方案':name,'RMSE_MW':m['rmse_mw'],'最大偏差_MW':m['max_abs_deviation_mw'],'欠供热_MWh':m['undersupply_mwh'],'泵耗_kWh':m['pump_energy_kwh'],'最低T1_C':m['min_T1_c'],'温度裕量_C':m['temperature_margin_c'],'最大爬坡违反_kg_s':m['ramp_violation']})
    best=min(candidates,key=lambda z:penalty_objective(z[1],K,UA,.7)); bm=metrics(best[1],K,UA,10.)
    save_csv(Q/'results'/'algorithm_runs.csv',runs,list(runs[0]));save_csv(Q/'results'/'diagnostic_solutions.csv',diag,list(diag[0]))
    traj=[]
    demand10=np.repeat(DEMAND,30)
    for i in range(len(bm['P'])): traj.append({'time_s':i*10,'time_h':i*10/3600,'demand_MW':demand10[i],'actual_MW':bm['P'][i],'q_kg_s':best[1][min(47,i//30)],'pump_kW':bm['PP'][i],**{f'T{j+1}_C':bm['T'][i,j] for j in range(5)}})
    save_csv(Q/'results'/'best_diagnostic_trajectory_10s.csv',traj,list(traj[0]))
    result={'status':'infeasible','conclusion':'q_k=0仍使T1跌破500℃，因此不存在满足连续温度约束的48段控制策略。','q0_min_T1_C':cert['min_T1_c'],'q0_margin_C':cert['temperature_margin_c'],'demand_energy_MWh':float(DEMAND.sum()*5/60),'initial_usable_energy_GJ':float(MASS*5*CP*(T0.mean()-TIN)/1e9),'best_diagnostic_name':best[0],'best_diagnostic':{k:v for k,v in bm.items() if k not in ['q','tt','T','P','PP']},'K_W_K':K,'UA_W_K':UA,'verification_dt_s':10}
    (Q/'results'/'q3_final_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(np.arange(1440)*10/3600,demand10,label='需求功率');ax.plot(np.arange(1440)*10/3600,bm['P'],label='诊断实际功率');ax.set(xlabel='时间/h',ylabel='功率/MW',title='需求与诊断方案实际供热功率');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(Q/'figures'/'fig1_power_tracking.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));ax.step(np.arange(48)*5/60,best[1],where='post');ax.set(xlabel='时间/h',ylabel='流量/(kg/s)',title='诊断控制流量');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(Q/'figures'/'fig2_flow.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));
    for j in range(5):ax.plot(bm['tt']/3600,bm['T'][:,j],label=f'T{j+1}')
    ax.axhline(500,color='red',ls='--',label='T1下限');ax.set(xlabel='时间/h',ylabel='温度/℃',title='五层温度与连续温度约束');ax.legend(ncol=3);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(Q/'figures'/'fig3_temperatures.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5));xs=[float(x['泵耗_kWh']) for x in diag];ys=[float(x['RMSE_MW']) for x in diag];cs=[float(x['温度裕量_C']) for x in diag];sc=ax.scatter(xs,ys,c=cs,cmap='coolwarm',s=45);fig.colorbar(sc,ax=ax,label='最低温度裕量/℃');ax.set(xlabel='泵耗/kWh',ylabel='跟踪RMSE/MW',title='诊断多目标解（全部违反温度约束）');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(Q/'figures'/'fig4_diagnostic_pareto.png',dpi=180);plt.close(fig)
    with (Q/'logs'/'run.log').open('w',encoding='utf-8') as f:f.write(json.dumps({'config':cfg,'result':result,'runs':runs},ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
