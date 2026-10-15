"""Public snapshots are evidence for humans, never authorization input."""
from __future__ import annotations

import json
from pathlib import Path

from .identity import safe_path
from .storage import OpsError, atomic_bytes, atomic_json, now

BEGIN, END = '<!-- RESEARCH_OPS_V1_BEGIN -->', '<!-- RESEARCH_OPS_V1_END -->'


def render_phase(phase, sections):
    titles = ('阶段目标', '技术背景', '系统架构', '代码实现', '算法原理', '实验协议',
              '实验结果', '问题分析', '局限性', '下一阶段', '给项目负责人的总结')
    if set(sections) != {str(i) for i in range(1, 12)} or not all(isinstance(v, str) and v.strip() for v in sections.values()):
        raise OpsError('all eleven evidence-bearing phase sections are required')
    return '# Phase ' + phase + '\n\n' + '\n\n'.join(
        f'## 【{i}】{title}\n\n{sections[str(i)].strip()}' for i, title in enumerate(titles, 1)) + '\n'


def update_block(path, text):
    path = Path(path)
    if path.is_symlink():
        raise OpsError('cannot update symlink document')
    old = path.read_text() if path.exists() else ''
    block = BEGIN + '\n' + text.rstrip() + '\n' + END
    if BEGIN in old or END in old:
        if old.count(BEGIN) != 1 or old.count(END) != 1 or old.index(BEGIN) >= old.index(END):
            raise OpsError('malformed generated document block')
        start, end = old.index(BEGIN), old.index(END) + len(END)
        value = old[:start] + block + old[end:]
    else:
        value = old.rstrip() + '\n\n' + block + '\n'
    atomic_bytes(path, value.encode())


def refresh(root, state):
    """No secret-bearing environment, raw paths or untracked filenames are exported."""
    public = {key: state[key] for key in (
        'project_id', 'phase_id', 'experiment_id', 'run_id', 'current_branch', 'base_commit',
        'latest_commit', 'protocol_sha256', 'stage_status', 'authorization_scope',
        'budget_remaining', 'last_verified_actual_timesteps', 'checkpoint_resume_class',
        'latest_successful_push', 'next_action', 'last_update_time', 'last_shutdown_status')}
    public['snapshot_note'] = 'Pre-commit snapshot; resolve current HEAD and delivery using researchctl status/verify. Never restores authority.'
    atomic_json(safe_path(root, 'docs/RESEARCH_STATUS.json'), public)
    summary = (f"阶段：{state['phase_id']}；状态：{state['stage_status']}。\n\n"
        f"分支：`{state['current_branch']}`。快照基于Commit：`{state['latest_commit']}`。\n\n"
        "当前Commit以 `git rev-parse HEAD` 为准；快照不能包含它自身尚未创建的Commit SHA。"
        "最近成功推送及实时完成状态见 `./researchctl status` 的核验收据。\n\n"
        "训练授权：NO；剩余timesteps/wall-time：0/0；optimizer更新：禁止。\n\n"
        "历史L9保持FAIL；L9R修复、L9S4096步链路Pilot已记录。L9S无成功导航，不能当作避障能力。\n\n"
        + ('安全下一步：先 `./researchctl sync` 重试同一待推送Commit。' if state['next_action']=='sync'
           else '安全下一步：在Codex中只读审计中断/失败证据；禁止重新start。' if state['next_action'].startswith('BLOCKED')
           else '安全下一步：`./researchctl doctor` 后运行 `./researchctl start --task ops-audit`，或在Codex中只读分析已有实验。')
        + ' 新的训练需要用户提供独立预算与协议。')
    update_block(safe_path(root, 'docs/PROJECT_PROGRESS.md'), '# 项目进度\n\n' + summary)
    update_block(safe_path(root, 'docs/CODEX_NEXT_ACTION.md'),
        '# 次日恢复入口\n\n先读 `docs/CODEX_MASTER_SPEC.md`。\n\n' + summary +
        f"\n\n项目目录：`{Path(root).resolve()}`。协议：`{state['protocol_sha256']}`。\n\n"
        "最新checkpoint：本地冻结L9S Pilot；其文件身份由doctor核验，不授权初始化正式模型。"
        "训练中断恢复策略RESTART_REQUIRED；未实现EXACT_RESUME或BOUNDARY_RESUME。\n\n"
        "早上：`./researchctl status` → `./researchctl recover` → `./researchctl doctor`。\n\n"
        "晚上：`./researchctl request-stop` → `./researchctl verify`；若SYNC_PENDING，运行 `./researchctl sync`。\n\n"
        "CLI只运行登记的维护任务；AI开发需用户启动Codex。无开机服务或无人值守训练。")
    update_block(safe_path(root, 'README.md'), '## 当前研究管理状态\n\n' + summary +
        '\n\n[每日操作协议](docs/RESEARCH_OPERATIONS_PROTOCOL.md) · '
        '[环境锁](docs/ENVIRONMENT_LOCK.md) · [OPS-01报告](docs/PHASE_OPS_01_PERSISTENCE.md)')
    daily = safe_path(root, 'docs/DAILY_PROGRESS.md')
    if daily.is_symlink():
        raise OpsError('cannot update symlink daily log')
    old = daily.read_text() if daily.exists() else '# 每日项目交接\n'
    entry_id = state['run_id'] + ':' + state['stage_status'] + ':' + state['last_update_time']
    marker = '<!-- ' + entry_id + ' -->'
    if marker not in old:
        entry = (f"\n## {state['last_update_time']} — {state['phase_id']}\n\n{marker}\n\n"
                 + summary + '\n\n未完成：新的训练预算、正式导航效果验证；'
                 '当前维护记录不是训练快照。完整测试/审查见阶段JSON及Markdown。\n')
        atomic_bytes(daily, (old + entry).encode())
