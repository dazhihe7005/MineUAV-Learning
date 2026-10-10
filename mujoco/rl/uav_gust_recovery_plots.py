"""Publish plots of saved traces only: no simulation, policy execution or training."""
import argparse,ast,json,resource
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_bc_safety import atomic_bytes,atomic_json
from uav_gust_recovery_audit import (CONDITIONS,CONTROLLERS,OLD_PARTS,PARTS,REPORT,
    REPRESENTATIVES,sha,verified_trace)

FIGURES='uav_gust_recovery_audit_figures'
NAMES=('force_vs_time.png','velocity_vs_time.png','position_error_vs_time.png',
       'command_vs_actual_velocity.png','termination_timeline.png')
LABELS={'constant_medium':'Constant-Medium','gust_medium':'Gust-Medium','gust_high':'Gust-High',
        'scripted':'Scripted','original':'Original BC','yaw_augmented':'Yaw-Augmented BC'}
COLORS=('#2474ac','#d55e00','#009e73')
GRID=np.arange(376)*.04

def figure_names(signals):
    # PI telemetry is absent in this acquisition. Never manufacture an integral plot.
    if signals['pi_integral_state']!='not recorded':
        raise ValueError('different telemetry schema requires a separately verified PI analysis')
    return NAMES

def cohort(result,condition,controller):
    return sorted((r for r in result['episode_metrics'] if r['condition']==condition and r['controller']==controller),
        key=lambda r:r['index'])

def aligned_series(z,name,direction):
    """NaN beyond observed termination; actions[i] belong to boundary[i], not[i+1]."""
    o=np.asarray(z['observations'],float);v=o[:,3:6]
    if name=='distance':values=np.linalg.norm(o[:,:3],axis=1)
    elif name=='speed':values=np.linalg.norm(v,axis=1)
    elif name=='velocity_projection':values=v@np.asarray(direction)
    elif name=='command_projection':values=(z['actions'][:,:3]*[1.5,1.5,1.])@np.asarray(direction)
    else:raise ValueError(name)
    result=np.full(len(GRID),np.nan)
    times=z['policy_times'][:len(values)];indices=np.rint(times/.04).astype(int)
    complete=np.isclose(times,indices*.04,atol=1e-8,rtol=0)
    result[indices[complete]]=values[complete]
    return result

def verify_provenance(root,result):
    for group in ('input_sha256','raw_trace_sha256'):
        for path,h in result['provenance'][group].items():
            if sha(root/path)!=h:raise ValueError('historical input changed: '+path)

def decorate(ax,ylabel):
    ax.axvspan(2,4,color='#888888',alpha=.12)
    ax.axvline(4,color='#555555',lw=.8,ls=':')
    ax.set(xlabel='Episode time (s)',ylabel=ylabel,xlim=(0,15))
    ax.grid(alpha=.2);ax.legend(fontsize=8)

def band(ax,values,label,color):
    # Suppress all-NaN endpoint columns explicitly; no successful-flight continuation.
    count=np.isfinite(values).sum(axis=0);valid=count>0
    low,med,high=np.nanquantile(values[:,valid],[.25,.5,.75],axis=0)
    ax.plot(GRID[valid],med,label=label,color=color)
    ax.fill_between(GRID[valid],low,high,color=color,alpha=.13)
    return count

def render(root,result):
    root=Path(root);verify_provenance(root,result);names=figure_names(result['signal_availability'])
    ev=json.loads((root/'mujoco/reports'/OLD_PARTS/'evaluation.json').read_text())
    original={r['task_key']:r for r in ev['rows']};group={};eligibility={}
    # Read one NPZ at a time. At most one group's100x376 derived series per signal.
    for condition in CONDITIONS:
        for controller in CONTROLLERS:
            rows=cohort(result,condition,controller)
            group[condition,controller]={name:[] for name in ('distance','speed','velocity_projection','command_projection')}
            for row in rows:
                source=original[row['task_key']]
                with verified_trace(source) as z:
                    for name in group[condition,controller]:
                        group[condition,controller][name].append(aligned_series(z,name,source['direction_world']))
            for name in group[condition,controller]:group[condition,controller][name]=np.array(group[condition,controller][name])
            eligibility[condition+'/'+controller]=np.isfinite(group[condition,controller]['distance']).sum(axis=0).tolist()
    out=root/'mujoco/reports'/FIGURES;out.mkdir(parents=True,exist_ok=True)
    files={}
    def save(fig,name):
        fig.tight_layout(rect=(0,0,1,.96) if fig._suptitle else None)
        atomic_bytes(out/name,lambda f:fig.savefig(f,format='png',dpi=130,bbox_inches='tight'))
        plt.close(fig);files[str((out/name).relative_to(root))]=sha(out/name)
    fig,ax=plt.subplots(figsize=(10,4))
    for condition,color in zip(CONDITIONS,COLORS):
        source=original[cohort(result,condition,'scripted')[0]['task_key']]
        with verified_trace(source) as z:ax.step(z['physics_force_times'],np.linalg.norm(z['physics_forces'],axis=1),where='post',label=LABELS[condition],color=color)
    decorate(ax,'Recorded force magnitude (N)');ax.set_title('500 Hz recorded force; every gust tick at/after4 s is zero')
    save(fig,names[0])
    for name,ylabel,filename in [('speed','Actual translational speed (m/s)',names[1]),('distance','Target distance (m)',names[2])]:
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for condition,color in zip(CONDITIONS,COLORS):band(axes[0,0],group[condition,'scripted'][name],LABELS[condition],color)
        axes[0,0].set_title('Scripted: median and IQR,100episodes/condition')
        for ax,condition in [(axes[0,1],'gust_medium'),(axes[1,0],'gust_high')]:
            for controller,color in zip(CONTROLLERS,COLORS):band(ax,group[condition,controller][name],LABELS[controller],color)
            ax.set_title(LABELS[condition]+': same100targets, allcontrollers')
        values=group['gust_medium','scripted'][name]
        for index in REPRESENTATIVES:axes[1,1].plot(GRID,values[index],label=f'fixed index{index}')
        axes[1,1].set_title('Gust-Medium Scripted: prespecified representatives')
        for ax in axes.flat:
            ax.axhline(.15 if name=='speed' else .1,color='black',ls='--',lw=.8,label='success threshold')
            decorate(ax,ylabel)
        fig.suptitle('Observed samples only; no continuation after success. Late counts in report.',fontsize=11)
        save(fig,filename)
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for row,condition in enumerate(CONDITIONS):
        values=group[condition,'scripted'];ax=axes[row,0]
        band(ax,values['velocity_projection'],'actual, median/IQR',COLORS[0]);band(ax,values['command_projection'],'command, median/IQR',COLORS[1])
        ax.set_title(LABELS[condition]+' Scripted:\nprojection on each episode force axis',fontsize=11)
        ax=axes[row,1]
        for index,color in zip(REPRESENTATIVES,('#2474ac','#d55e00','#009e73','#a35cc6')):
            ax.plot(GRID,values['velocity_projection'][index],color=color,label=f'index{index} actual')
            ax.plot(GRID,values['command_projection'][index],color=color,ls='--',lw=.8,label=f'index{index} command')
        ax.set_title('Fixed paired representatives:\nsolid actual, dashed command',fontsize=11)
        for ax in axes[row]:decorate(ax,'Force-axis velocity (m/s)');ax.axhline(0,color='black',lw=.5)
    fig.suptitle('Opposite velocity/command can be intentional braking, not proof of wrong-sign PI',fontsize=11)
    save(fig,names[3])
    fig,(ax,low)=plt.subplots(2,1,figsize=(12,9),gridspec_kw={'height_ratios':[2,1]})
    categories=[]
    for i,(condition,controller) in enumerate((c,n) for c in CONDITIONS for n in CONTROLLERS):
        rows=[r for r in result['episode_metrics'] if r['condition']==condition and r['controller']==controller]
        categories.append(LABELS[condition]+' / '+LABELS[controller])
        for r in rows:ax.scatter(r['duration_s'],i+(r['index']-49.5)*.006,c='green' if r['success'] else 'red',s=8,alpha=.6)
    ax.set(yticks=range(9),yticklabels=categories,xlabel='Actual termination time (s)',xlim=(0,15.35))
    ax.axvspan(2,4,color='gray',alpha=.12);ax.axvline(15,color='black',ls='--',label='unchanged timeout15 s')
    ax.legend(loc='upper left');ax.set_title('All900episodes: green success / red timeout; no physical failures')
    rows=[r for r in result['episode_metrics'] if r['post_gust'] is not None]
    names2=['distance-only ever','speed-only ever','joint ever','joint5samples']
    counts=[sum(r['post_gust'][k] for r in rows) for k in ['distance_only_ever','speed_only_ever','joint_ever']]
    counts.append(sum(r['post_gust']['joint_recovery_confirmation_s'] is not None for r in rows))
    low.bar(names2,counts,color=['#2474ac','#d55e00','#555555','#555555'])
    for x,n in enumerate(counts):low.text(x,n+8,f'{n}/600',ha='center')
    low.set(ylim=(0,650),ylabel='Gust episodes, after4 s');low.set_title('Different-time individual thresholds ≠ joint recovery')
    save(fig,names[4]);verify_provenance(root,result)
    return dict(sha256=files,complete_boundary_eligibility=eligibility,
        pi_integral_diagnostics='not generated: PI time series not recorded')

def interpret(result):
    return [
        dict(grade='Confirmed',claim='All600gust episodes have full2s exposure, zero recorded force after4s, and11s observed recovery window; all end timeout, not physical/numerical failure.'),
        dict(grade='Confirmed',claim='None of600gust episodes meets simultaneous distance<.10m ANDspeed<.15m/s even once after4s. Five-sample dwell alone does not explain failure.'),
        dict(grade='Confirmed',claim='Recorded trajectories approach/cross the target again but retain motion; postremoval distance can increase briefly, then falls. This is not monotonic divergence or proof of permanent inability to recover.'),
        dict(grade='Confirmed',claim='All900analyzed episodes have zero recorded policy command-element and allocator saturation. PI integral/acceleration saturation is a separate, unrecorded signal.'),
        dict(grade='Confirmed',claim='PI implementation has integral-contribution clamps and conditional anti-windup, retains state within episodes and resets at episode reset. Only initial integral zero is saved, not its subsequent trajectory.'),
        dict(grade='Supported hypothesis',claim='A shared underdamped/transient velocity-tracking limitation in the common control/vehicle cascade is consistent with repeated target crossings and similar Scripted/BC dynamics. Lack of effective settling is observed, but its individual-module cause is not identified.'),
        dict(grade='Supported hypothesis',claim='Stored integral compensation after gust removal could contribute to transient motion; this is compatible with the PI implementation but not demonstrated by measured integral traces.'),
        dict(grade='Unresolved',claim='Actual PI windup, clamping, anti-windup activation, reversed compensation, attitude lag, and relative contributions of P/I/lower loops cannot be established without100Hz internal telemetry.'),
        dict(grade='Unresolved',claim='Whether unchanged controllers would eventually satisfy recovery after15s is right-censored. The existing11swindow is real, not an empty/reversed timing interval; permanent instability and a timing bug are not established.')]

def completed_resource_peak(records,self_peak):
    if not records or any(r['exit_code']!=0 or r['failure'] or r['oom_count_delta'] for r in records):
        raise ValueError('incomplete phase or OOM cannot be published as complete')
    return max([self_peak]+[r['process_tree_peak_hwm_bytes'] for r in records])

def finalize_resources(root):
    # The supervisor writes the complete publication HWM only AFTER that child exits.
    root=Path(root);path=root/'mujoco/reports'/REPORT;result=json.loads(path.read_text())
    verify_provenance(root,result)
    for group in [result['analysis_source_sha256'],result['figures']['sha256']]:
        if any(sha(root/p)!=h for p,h in group.items()):raise ValueError('publication source/figure changed')
    records=[json.loads((root/'mujoco/reports'/PARTS/f'resource_{name}.json').read_text())
        for name in ('analysis','tests','publication')]
    result['resources']['publication_completed']=records[-1]
    result['resources']['peak_rss_bytes']=completed_resource_peak(records,
        max(result['resources']['publication_process_peak_rss_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    result['resources']['guard_workers_field']='serial analysis child count; MuJoCo workers remain0'
    atomic_json(path,result)
    print('Completed-phase peak RSS MiB:',result['resources']['peak_rss_bytes']/1024**2,flush=True)

def publish(root):
    root=Path(root);result=json.loads((root/'mujoco/reports'/PARTS/'analysis.json').read_text())
    result['figures']=render(root,result);result['evidence']=interpret(result)
    prior=json.loads((root/'mujoco/reports/uav_bc_external_disturbance_seed0.json').read_text())
    result['models']=dict(checkpoint_hashes=prior['checkpoint_hashes_after'],
        parameter_hashes=prior['parameter_hashes_after'],
        parameter_hash_status='previous experiment attestation; checkpoint bytes verified identical before/after this audit; no model loaded or executed')
    result['resources']=dict(analysis=json.loads((root/'mujoco/reports'/PARTS/'resource_analysis.json').read_text()),
        publication_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        analysis_processes=1,mujoco_workers=0,new_episodes=0,training_steps=0)
    test_resource=json.loads((root/'mujoco/reports'/PARTS/'resource_tests.json').read_text())
    if test_resource['exit_code']!=0 or test_resource['failure'] or test_resource['oom_count_delta']:
        raise ValueError('pure audit tests/resource check not green')
    result['resources']['tests']=test_resource
    result['resources']['peak_rss_bytes']=max(result['resources']['publication_process_peak_rss_bytes'],
        result['resources']['analysis']['process_tree_peak_hwm_bytes'],test_resource['process_tree_peak_hwm_bytes'])
    result['analysis_source_sha256']={str((root/'mujoco/rl'/name).relative_to(root)):sha(root/'mujoco/rl'/name)
        for name in ['uav_gust_recovery_audit.py','uav_gust_recovery_plots.py','test_uav_gust_recovery_audit.py']}
    result['tests']=dict(command='.venv/bin/python -m unittest discover -s mujoco/rl -p test_uav_gust_recovery_audit.py -q',
        passed=len([n for n in ast.walk(ast.parse((root/'mujoco/rl/test_uav_gust_recovery_audit.py').read_text()))
            if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]),failures=0,errors=0,
        note='fresh guarded pure fixture suite passed; global tests intentionally not run because some simulate/train')
    result['next']='One minimal fixed-target Scripted Constant-Medium vs Gust-Medium telemetry validation, recording100Hz velocity error/P/I/limited acceleration, anti-windup/clamp events, attitude and allocator output with unchanged gains/criteria. Not executed.'
    result['limitations']=['existing100target cohort only','finite15s/right-censored recovery',
        '25Hz state/action sampling misses100Hz PI and500Hz state transients',
        'different force onset histories prevent attributing Constant/Gust difference solely to withdrawal',
        'success termination prevents equal15s exposure/indefinite holding comparison',
        'no measured PI integral/PI output/attitude/rotor trajectory','no causal controller intervention']
    verify_provenance(root,result);atomic_json(root/'mujoco/reports'/REPORT,result)
    print('Published read-only gust audit:',len(result['episode_metrics']),'episodes;',len(result['figures']['sha256']),'figures',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--finalize-resources',action='store_true');a=p.parse_args()
    (finalize_resources if a.finalize_resources else publish)(a.root)
