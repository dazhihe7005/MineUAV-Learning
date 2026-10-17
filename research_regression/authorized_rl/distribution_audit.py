"""R23 pure arithmetic on frozen Train/Validation identities, never a policy input."""
import math
import re
from collections import Counter

CLASSES=('success','collision','time_limit','physical_failure','sensor_failure')


def number(value, *, minimum=None):
    if type(value) not in (int,float) or not math.isfinite(value) or minimum is not None and value<minimum:
        raise ValueError('invalid finite audit scalar')
    return value


def yaw_distance(a,b):
    delta=number(a)-number(b)
    return abs(math.atan2(math.sin(delta),math.cos(delta)))


def task_map(rows,split):
    if split not in ('train','validation') or not isinstance(rows,list) or not rows:
        raise ValueError('explicit nonempty Train or Validation only')
    out={}
    for row in rows:
        identity=row['identity_sha256']
        if (not isinstance(identity,str) or not re.fullmatch('[0-9a-f]{64}',identity)
                or identity in out or row['split']!=split or row['layout'] not in ('empty','front_box','side_pair','detour','channel')):
            raise ValueError('wrong/duplicate task identity or split')
        number(row['start_yaw_rad']);out[identity]=row
    return out


def heading_coverage(train,validation):
    tr=task_map(train,'train');va=task_map(validation,'validation')
    if set(tr)&set(va):raise ValueError('Train/Validation identity overlap')
    output=[]
    for identity,row in va.items():
        matching=[t for t in tr.values() if t['layout']==row['layout']]
        if not matching:raise ValueError('missing same-layout Train comparator')
        any_gap=min(yaw_distance(row['start_yaw_rad'],t['start_yaw_rad']) for t in tr.values())
        same_gap=min(yaw_distance(row['start_yaw_rad'],t['start_yaw_rad']) for t in matching)
        output.append({'identity_sha256':identity,'layout':row['layout'],'yaw_rad':row['start_yaw_rad'],
            'yaw_deg':math.degrees(row['start_yaw_rad']),
            'nearest_any_train_yaw_gap_rad':any_gap,'nearest_same_layout_yaw_gap_rad':same_gap,
            'nearest_any_train_yaw_gap_deg':math.degrees(any_gap),
            'nearest_same_layout_yaw_gap_deg':math.degrees(same_gap),
            'same_layout_train_yaw_deg':[math.degrees(t['start_yaw_rad']) for t in matching]})
    return output


def counts(rows):
    c=Counter(v['terminal_class'] for v in rows)
    if set(c)-set(CLASSES):raise ValueError('unknown terminal class')
    return {k:c[k] for k in CLASSES}


def original_train_groups(rows,train):
    tasks=task_map(train,'train');groups={identity:[] for identity in tasks};last=0
    for row in rows:
        identity=row['base_scenario_identity_sha256'];n=row['episode_policy_steps'];end=row['policy_end_step']
        if (row['distance_stage']!='ORIGINAL' or identity not in tasks or row['scenario_identity_sha256']!=identity
                or type(n) is not int or n<=0 or type(end) is not int or not end>last or n>end-last
                or row['terminal_class'] not in CLASSES or row['outcome']!=row['terminal_class']):
            raise ValueError('invalid/foreign/mixed/contradictory completed Train episode')
        for key in ('minimum_goal_error_m','final_target_distance_m'):number(row[key],minimum=0)
        number(row['episode_return'])
        near=row['near_goal_steps'];joint=row['goal_stop_diagnostics']['max_consecutive_joint_samples']
        if (type(near) is not int or not 0<=near<=n or type(joint) is not int or not 0<=joint<=near
                or row['terminal_class']=='success' and joint<5):raise ValueError('inconsistent actual goal samples')
        last=end;groups[identity].append(row)
    def summary(selected):
        def average(key):return math.fsum(v[key] for v in selected)/len(selected) if selected else None
        return {'episodes':len(selected),'policy_steps':sum(v['episode_policy_steps'] for v in selected),
            'terminal_counts':counts(selected),'mean_return':average('episode_return'),
            'mean_minimum_goal_error_m':average('minimum_goal_error_m'),'mean_final_goal_error_m':average('final_target_distance_m'),
            'episodes_with_sampled_goal_entry':sum(v['near_goal_steps']>0 for v in selected),
            'joint_max_distribution':dict(sorted(Counter(v['goal_stop_diagnostics']['max_consecutive_joint_samples'] for v in selected).items()))}
    return {'by_identity':{k:summary(v) for k,v in groups.items()},
        'by_layout':{layout:summary([v for v in rows if tasks[v['base_scenario_identity_sha256']]['layout']==layout])
                     for layout in sorted({v['layout'] for v in tasks.values()})},
        'overall':summary(rows),'scope':'completed nonstationary stochastic ORIGINAL Train episodes,not final policy evaluation'}


def validation_groups(rows,validation):
    tasks=task_map(validation,'validation');found={}
    for row in rows:
        identity=row['scenario_identity_sha256']
        if (identity not in tasks or identity in found or row['layout']!=tasks[identity]['layout']
                or row['terminal_class'] not in CLASSES or row.get('outcome')!=row['terminal_class']
                or type(row['policy_steps']) is not int or row['policy_steps']<=0):
            raise ValueError('foreign/duplicate/invalid actual Validation record')
        number(row['final_target_error_m'],minimum=0)
        pairs=row['collision_contact_pairs']
        if (not isinstance(pairs,list) or any(not isinstance(p,list) or len(p)!=2
                or any(not isinstance(s,str) for s in p) for p in pairs)
                or row['terminal_class']!='collision' and pairs):raise ValueError('invalid actual contact pairs')
        found[identity]={'layout':row['layout'],'terminal_class':row['terminal_class'],
            'policy_steps':row['policy_steps'],'final_target_error_m':row['final_target_error_m'],
            'collision_contact_pairs':pairs}
    if set(found)!=set(tasks):raise ValueError('incomplete actual Validation identity coverage')
    return {'by_identity':found,'by_layout':{layout:{'episodes':sum(v['layout']==layout for v in rows),
        'terminal_counts':counts([v for v in rows if v['layout']==layout])} for layout in sorted({v['layout'] for v in tasks.values()})}}
