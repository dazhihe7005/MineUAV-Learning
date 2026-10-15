"""Figures from preserved measurements; no fabricated vehicle trajectories."""
from __future__ import annotations
import io
import struct
import zlib
from research_ops.storage import atomic_bytes,sha256
from research_ops_v2.publication import check_blob


def clean_png(data):
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('not PNG')
    out=[data[:8]];offset=8
    while offset<len(data):
        if offset+12>len(data):raise ValueError('truncated PNG')
        n=struct.unpack('>I',data[offset:offset+4])[0];end=offset+12+n
        if end>len(data):raise ValueError('truncated PNG chunk')
        kind=data[offset+4:offset+8]
        if zlib.crc32(data[offset+4:end-4])&0xffffffff!=struct.unpack('>I',data[end-4:end])[0]:raise ValueError('PNG CRC')
        if kind in (b'IHDR',b'PLTE',b'tRNS',b'IDAT',b'IEND'):out.append(data[offset:end])
        offset=end
    output=b''.join(out);check_blob('mujoco/reports/audit/figures/a.png',output)
    return output


def plot_all(root,result,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outputs=[]
    directory=root/'mujoco/reports/lidar_ppo_learning_diagnosis/figures'
    def save(fig,name):
        stream=io.BytesIO();fig.savefig(stream,format='png',dpi=110,bbox_inches='tight')
        path=directory/name;atomic_bytes(path,clean_png(stream.getvalue()),exclusive=True)
        plt.close(fig);outputs.append({'path':str(path.relative_to(root)),'sha256':sha256(path),
            'source':'preserved episode/validation/snapshot aggregates, not new rollouts'})
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    episodes=rows['L9'];blocks=result['L9']['step_blocks']
    axes[0,0].scatter([e['policy_end_step'] for e in episodes],[e['episode_return'] for e in episodes],s=5,alpha=.45)
    axes[0,0].set(xlabel='L9 terminal policy step',ylabel='Recorded episode return',title='706 completed episodes; interrupted run')
    x=list(range(4));base=[0.]*4
    for kind,color in [('collision','tomato'),('time_limit','steelblue'),('success','seagreen')]:
        values=[b['counts'][kind]/b['episodes'] if b['episodes'] else 0 for b in blocks]
        axes[0,1].bar(x,values,bottom=base,label=kind,color=color)
        base=[a+b for a,b in zip(base,values)]
    axes[0,1].set(xticks=x,xticklabels=['0-30k','30-60k','60-90k','90-119370'],ylabel='Fraction',title='Episode-end bins; not independent seeds')
    axes[0,1].legend(fontsize=8)
    axes[1,0].scatter([e['policy_end_step'] for e in episodes],[e['final_target_distance_m'] for e in episodes],s=5,alpha=.5)
    axes[1,0].axhline(.1,color='red',linestyle='--');axes[1,0].set(ylabel='Terminal goal error (m)',xlabel='Policy step',title='Not closest-approach error')
    axes[1,1].scatter([e['policy_end_step'] for e in episodes],[e['mean_action_norm'] for e in episodes],s=5,alpha=.5)
    axes[1,1].set(xlabel='Policy step',ylabel='Mean clipped normalized action norm',title='Not physical velocity or per-axis saturation')
    fig.tight_layout();save(fig,'l9_recorded_training.png')
    layouts=list(result['L9']['by_layout']);fig,axes=plt.subplots(1,3,figsize=(13,4))
    by=result['L9']['by_layout']
    axes[0].bar(layouts,[by[l]['counts']['collision'] for l in layouts],label='collision',color='tomato')
    axes[0].bar(layouts,[by[l]['counts']['time_limit'] for l in layouts],bottom=[by[l]['counts']['collision'] for l in layouts],label='timeout',color='steelblue')
    axes[0].set(title='L9 completed Train episodes',ylabel='Count');axes[0].legend()
    axes[1].bar(layouts,[by[l]['final_target_distance_m']['mean'] for l in layouts]);axes[1].set(title='Terminal error, not minimum error',ylabel='Mean (m)')
    axes[2].bar(layouts,[by[l]['trajectory_length_m']['mean'] for l in layouts]);axes[2].set(title='Movement does not imply directed exploration',ylabel='Mean path (m)')
    for ax in axes:ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();save(fig,'l9_layout_stratification.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,tag in enumerate(('L8','L9S','L9')):
        eps=result['validation'][tag]['episodes'];axes[0].bar([j+(i-1)*.24 for j in range(6)],
            [e['final_target_error_m'] for e in eps],width=.24,label=tag)
    axes[0].set(xticks=list(range(6)),xticklabels=[e['layout'] for e in result['validation']['L9']['episodes']],
        ylabel='Final goal error (m)',title='Saved deterministic Validation: all collisions')
    axes[0].tick_params(axis='x',rotation=30);axes[0].legend()
    for i,tag in enumerate(('L9S','L9')):
        metrics=result['ppo_snapshots'][tag]['metrics']
        axes[1].scatter(i,metrics['train/approx_kl'],s=50,label=tag)
        axes[1].annotate(f"std={metrics['train/std']:.4f}\nEV={metrics['train/explained_variance']:.3f}\nclip=0",(i,metrics['train/approx_kl']),xytext=(8,-15),textcoords='offset points')
    axes[1].set(xticks=[0,1],xticklabels=['L9S / 4096','L9 / 100096'],ylabel='Recorded approximate KL',
        title='Different attempts: two points are NOT a learning curve',xlim=(-.4,1.7))
    fig.tight_layout();save(fig,'validation_and_separate_ppo_snapshots.png')
    return outputs
