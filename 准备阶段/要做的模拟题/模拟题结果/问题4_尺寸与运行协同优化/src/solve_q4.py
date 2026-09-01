from __future__ import annotations
import csv,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution,minimize
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];Q=Path(__file__).resolve().parents[1]
RHO=1800.;CP=1500.;US=3.2;KEFF=95.;TIN=290.;TA=25.;T0=np.array([565.,555.,520.,430.,330.])
DEMAND=np.repeat(np.array([12.,18.,25.,32.,28.,22.,16.,10.]),6)

def geom(R,H):
    V=np.pi*R*R*H;m=RHO*V/5;S=2*np.pi*R*H+2*np.pi*R*R;K=5*KEFF*np.pi*R*R/H
    side=2*np.pi*R*H/5;UA=np.full(5,US*side);UA[[0,4]]+=US*np.pi*R*R
    return V,m,S,K,UA
def rhs(T,q,m,K,UA):
    z=np.empty(5);z[0]=(q*CP*(T[1]-T[0])+K*(T[1]-T[0])-UA[0]*(T[0]-TA))/(m*CP)
    for i in range(1,4):z[i]=(q*CP*(T[i+1]-T[i])+K*(T[i-1]-2*T[i]+T[i+1])-UA[i]*(T[i]-TA))/(m*CP)
    z[4]=(q*CP*(TIN-T[4])+K*(T[3]-T[4])-UA[4]*(T[4]-TA))/(m*CP);return z
def heuristic(R,H):
    V,m,S,K,UA=geom(R,H);q=np.zeros(48);T=T0.copy()
    for k in range(48):
        qq=np.clip(DEMAND[k]*1e6/(CP*max(T[0]-TIN,1)),0,100)
        if k:qq=np.clip(qq,q[k-1]-20,q[k-1]+20)
        q[k]=qq
        for _ in range(5):
            h=60.;a=rhs(T,qq,m,K,UA);b=rhs(T+h*a/2,qq,m,K,UA);c=rhs(T+h*b/2,qq,m,K,UA);d=rhs(T+h*c,qq,m,K,UA);T=T+h*(a+2*b+2*c+d)/6
    return q
def simulate(R,H,q,dt=60.):
    V,m,S,K,UA=geom(R,H);n=int(14400/dt);T=np.empty((n+1,5));T[0]=T0;P=np.empty(n)
    for j in range(n):
        k=min(47,int(j*dt//300));qq=q[k];x=T[j];a=rhs(x,qq,m,K,UA);b=rhs(x+dt*a/2,qq,m,K,UA);c=rhs(x+dt*b/2,qq,m,K,UA);d=rhs(x+dt*c,qq,m,K,UA);T[j+1]=x+dt*(a+2*b+2*c+d)/6;P[j]=qq*CP*(.5*(T[j,0]+T[j+1,0])-TIN)/1e6
    dem=np.repeat(DEMAND,int(round(300/dt)));E0=m*CP*np.sum(T0-TIN);Ef=m*CP*np.sum(T[-1]-TIN)
    return {'rmse':float(np.sqrt(np.mean((P-dem)**2))),'under':float(np.sum(np.maximum(dem-P,0))*dt/3600),'minT':float(T[:,0].min()),'ratio':float(Ef/E0),'T':T,'P':P,'time':np.arange(n+1)*dt,'q':q,'V':V,'m':m,'S':S,'K':K,'UA':UA}
def obj_geom(x):
    R,H=x;V,*_=geom(R,H)
    if R<2 or R>4 or H<5 or H>10:return 1e6+1e5*(max(0,2-R)+max(0,R-4)+max(0,5-H)+max(0,H-10))
    if V>220:return 1e5+(V-220)**2*1e3
    s=simulate(R,H,heuristic(R,H));return s['rmse']+20*max(0,500-s['minT'])+100*max(0,.2-s['ratio'])+s['S']/1000
def save(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    for d in ['results','figures','logs','config','data']:(Q/d).mkdir(parents=True,exist_ok=True)
    seeds=[202604,202605,202606];cfg={'rho_kg_m3':RHO,'cp_J_kgK':CP,'Us_W_m2K':US,'keff_W_mK':KEFF,'seeds':seeds,'optimization_dt_s':60,'verification_dt_s':10}
    (Q/'config'/'q4_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    demand_MWh=float(DEMAND.sum()*5/60);maxE=RHO*220*CP*np.mean(T0-TIN)/3.6e9;deliver=.8*maxE;deficit=demand_MWh-deliver
    runs=[];cands=[]
    # 嵌套网格：外层几何，内层需求反演控制
    t=time.perf_counter();best=None
    for R in np.linspace(2,4,25):
        for H in np.linspace(5,10,25):
            if np.pi*R*R*H<=220+1e-9:
                v=obj_geom((R,H))
                if best is None or v<best[0]:best=(v,R,H)
    elapsed=time.perf_counter()-t;cands.append(('嵌套网格+内层控制',best[1],best[2]));s=simulate(best[1],best[2],heuristic(best[1],best[2]));runs.append({'algorithm':'嵌套网格+内层控制','seed':'deterministic','R_m':best[1],'H_m':best[2],'volume_m3':s['V'],'surface_m2':s['S'],'rmse_MW':s['rmse'],'minT_C':s['minT'],'terminal_ratio':s['ratio'],'violation':max(0,500-s['minT'])+max(0,.2-s['ratio']),'runtime_s':elapsed})
    for seed in seeds:
        t=time.perf_counter();de=differential_evolution(obj_geom,[(2,4),(5,10)],seed=seed,popsize=8,maxiter=18,tol=1e-6,polish=False,workers=1);elapsed=time.perf_counter()-t;R,H=de.x;cands.append((f'差分进化({seed})',R,H));s=simulate(R,H,heuristic(R,H));runs.append({'algorithm':'差分进化-嵌套控制','seed':seed,'R_m':R,'H_m':H,'volume_m3':s['V'],'surface_m2':s['S'],'rmse_MW':s['rmse'],'minT_C':s['minT'],'terminal_ratio':s['ratio'],'violation':max(0,500-s['minT'])+max(0,.2-s['ratio']),'runtime_s':elapsed})
        t=time.perf_counter();loc=minimize(obj_geom,de.x,method='Nelder-Mead',options={'maxiter':100,'xatol':1e-7,'fatol':1e-7});elapsed=time.perf_counter()-t;R,H=loc.x;cands.append((f'全局+局部({seed})',R,H));s=simulate(R,H,heuristic(R,H));runs.append({'algorithm':'全局搜索+局部精修','seed':seed,'R_m':R,'H_m':H,'volume_m3':s['V'],'surface_m2':s['S'],'rmse_MW':s['rmse'],'minT_C':s['minT'],'terminal_ratio':s['ratio'],'violation':max(0,500-s['minT'])+max(0,.2-s['ratio']),'runtime_s':elapsed})
    # 最大体积下的最小面积解析诊断几何
    Ra=(220/(2*np.pi))**(1/3);Ha=220/(np.pi*Ra**2);qa=heuristic(Ra,Ha);sv=simulate(Ra,Ha,qa,10.)
    result={'status':'infeasible','conclusion':'即使V=220 m3、无散热且允许消耗80%初始显热，可供能量仍小于需求，尺寸-控制原问题可行域为空。','demand_energy_MWh':demand_MWh,'max_initial_usable_energy_MWh':maxE,'max_deliverable_with_20pct_terminal_MWh':deliver,'minimum_energy_deficit_MWh':deficit,'diagnostic_geometry':{'R_m':Ra,'H_m':Ha,'V_m3':sv['V'],'surface_m2':sv['S'],'total_mass_kg':sv['m']*5,'K_W_K':sv['K'],'UA_W_K':sv['UA'].tolist(),'minT_C':sv['minT'],'terminal_ratio':sv['ratio'],'tracking_RMSE_MW':sv['rmse']},'verification_dt_s':10}
    (Q/'results'/'q4_final_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');save(Q/'results'/'algorithm_comparison.csv',runs)
    # geometry sensitivity
    rows=[]
    for R in np.linspace(2,4,41):
        for H in np.linspace(5,10,41):
            V,m,S,K,UA=geom(R,H)
            if V<=220:rows.append({'R_m':R,'H_m':H,'V_m3':V,'surface_m2':S,'mass_kg':5*m,'K_W_K':K,'UA_top_W_K':UA[0],'energy_upper_MWh':5*m*CP*np.mean(T0-TIN)/3.6e9})
    save(Q/'results'/'geometry_sensitivity.csv',rows)
    traj=[]
    for i in range(len(sv['P'])):traj.append({'time_s':i*10,'demand_MW':np.repeat(DEMAND,30)[i],'actual_MW':sv['P'][i],'q_kg_s':qa[min(47,i//30)],**{f'T{j+1}_C':sv['T'][i,j] for j in range(5)}})
    save(Q/'results'/'diagnostic_trajectory_10s.csv',traj)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(7.5,4.5));x=np.array([z['V_m3'] for z in rows]);y=np.array([z['surface_m2'] for z in rows]);c=np.array([z['energy_upper_MWh'] for z in rows]);sc=ax.scatter(x,y,c=c,s=8,cmap='viridis');fig.colorbar(sc,ax=ax,label='初始显热上界/MWh');ax.axvline(220,color='r',ls='--');ax.set(xlabel='体积/m³',ylabel='总换热面积/m²',title='几何可行域、面积与能量上界');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(Q/'figures'/'fig1_geometry_space.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5));ax.bar(['需求','最大初始显热','保留20%后最多可供'],[demand_MWh,maxE,deliver],color=['#c44','#3978b5','#e2a52d']);ax.set(ylabel='能量/MWh',title='问题4能量可行性证书');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(Q/'figures'/'fig2_energy_certificate.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));
    for j in range(5):ax.plot(sv['time']/3600,sv['T'][:,j],label=f'T{j+1}')
    ax.axhline(500,color='r',ls='--');ax.set(xlabel='时间/h',ylabel='温度/℃',title='最大体积最小面积诊断几何：五层温度');ax.legend(ncol=3);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(Q/'figures'/'fig3_diagnostic_temperatures.png',dpi=180);plt.close(fig)
    with (Q/'logs'/'run.log').open('w',encoding='utf-8') as f:f.write(json.dumps({'config':cfg,'result':result,'runs':runs},ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
