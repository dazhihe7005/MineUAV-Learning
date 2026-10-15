# Research Operations V2 使用与版本协议

本协议为 `RESEARCH_OPS_V2`。原 `researchctl`、`research_ops/`、V1本地协议、预算账本、attempt均保留；V2不是对原V1身份重新计算SHA。原V1在其冻结分支仍可使用；**当前V2分支不要用V1 doctor绕过分支身份检查**。

## 每日操作

```bash
cd /home/keyousuke/MineUAV-Learning-px4-sitl
./researchctl-v2 status
./researchctl-v2 recover
./researchctl-v2 doctor
./researchctl-v2 start --task audit
```

晚上先请求停止；若返回 `STOP_REQUESTED`，等待当前任务退出，再核验与同步：

```bash
./researchctl-v2 request-stop --no-sync
./researchctl-v2 recover
./researchctl-v2 verify
./researchctl-v2 sync
./researchctl-v2 status
```

不带 `--no-sync` 的停止命令在确认安全停止后尝试同步。`SYNC_PENDING` 表示本地提交保留、远端未核验；网络恢复后只重试 `sync`，不能修改已经封存的待推文件。不同步成功不能宣称阶段完成。

这些命令只执行注册的有限维护工作，不启动AI、后台训练或开机服务；电脑关闭后不会继续工作。`recover`只审计，不自动重放旧任务。不确定的进程/锁/状态立即BLOCKED，不杀用户进程。训练预算始终0，任何PPO optimizer入口不受支持。

## 版本存储与升级

原V1文件位于 `.research_local/`，V2只读核验它们。新目录为：

```text
.research_local/ops_v2/
  active.json                         # 最后一步原子切换
  versions/v2_<登记清单SHA>/
    protocol.json                     # 新身份与父协议
    source_archive/                   # 冻结源码
    v1_backup/                        # 原文件原位不动，另存副本
    budget_ledger.json                # 原零预算字节副本
    migration.attempt.json
    migration.complete.json
    state.json                        # 校验和封装；只更新当前版本
    runs/OPS2_<随机ID>/                # 唯一attempt、结果及故障证据
```

首次 `migrate`，之后 `register`。二者均要求清单SHA显式匹配、独立审查GO绑定清单与源码；验证V1身份、原始历史、预算、资源、当前工作分支与登记提交。后续登记允许在新分支使用新源码，但保留前版协议、源码归档及文件身份；不修改前版SHA。活动任务或待同步版本不能被替换。

候选目录独占创建；失败残留保留，不能删除attempt自动重试。只有全部写入并校验后才原子切换active指针。失败不激活候选，原V1不变。当前不提供自动回滚命令；若状态冲突，应保留证据进行Level1审计，而非手改指针、reset或重新init。

登记草稿由 `prepare` 生成，不授权执行、不自动激活。例如下一项离线审计可使用：

```bash
./researchctl-v2 prepare --phase AUDIT-01 --task-id audit \
  --task-type offline-audit --parameters '{"scope":"history"}' --wall-time 60 \
  --source-path research_ops_v2 \
  --sync-file docs/PHASE_AUDIT_01.md \
  --sync-file mujoco/reports/audit_01.json \
  --report-path mujoco/reports/audit_01.json \
  --output .research_local/audit_01.registration.json
```

实际阶段还需显式列出所有将提交的代码、公共快照及长期进度文件。管理代码、回归启动器、CLI、V2测试源码强制加入身份范围，不能传空源码清单逃避审查。清单经过只读审查后：

```bash
./researchctl-v2 register --manifest <清单路径> --review <审查证据路径> \
  --expected-sha <清单规范化SHA256>
```

清单和审查文件不是安全凭据、不是训练授权。拥有本机文件写权限者仍可能篡改程序；本机制防误操作、并发和证据串用，不宣称对恶意系统管理员提供隔离。

## 固定任务能力

| 类型 | 参数白名单 | 实际行为 |
|---|---|---|
| offline-audit | scope=history | 核验身份、历史、依赖及资源 |
| regression-check | suite=historical / operations | 固定无优化器历史套件 / V2标准库单测 |
| documentation-update | 空参数 | 生成不带权限的V2公共JSON快照 |
| experiment-analysis | experiment_id=L9 / L9S | 读取历史报告，保存其来源摘要；不重评模型 |

任务明确单worker、墙钟预算0–600秒（不含0）；管理准入核验在任务计时前执行。任务工作目录固定仓库，结果固定在当前版本唯一run目录。既不接收shell，也不接收脚本、模块名或任意路径作为执行目标。`experiment-analysis`不是自动算法诊断，`documentation-update`不会伪造阶段正文或实验结论。

共享Git common-dir中的V1 flock，互斥所有管理版本。worker先登记身份，再等待主管写入启动许可；主管丢失且未获许可则不执行。许可后有独立Linux守护进程监测主管、worker、资源和绝对截止时间（50ms轮询，不是实时操作系统保证）。资源不足、超时、停止请求终止整个任务组，保留诊断。单worker不是“只有一个操作系统进程”：还存在主管与轻量守护进程，没有额外训练worker。

恢复遇到仍存活的后代进程则BLOCKED，不自动接管。唯一且确认未发布启动意图的孤立attempt可以封口为中断；发布过启动意图但缺进程身份时，不推断“没有进程”，必须BLOCKED。用户另一个不遵守flock的编辑器不受内核锁强制限制，故启动和结束仍核验源码。

## Git同步保护

只接受登记后的显式文件名单：中文阶段Markdown、JSON汇总、源码/测试及小型PNG。文件≤2MiB；禁止权重、NPZ、长日志、凭据和路径逃逸。PNG要求受限结构、CRC及尺寸上限，不支持带任意附加元数据的图片。凭据检查是规则扫描，不能保证发现所有未知敏感内容，提交前仍需审阅。

V1历史保护唯一允许的覆盖是本轮17份**测试夹具源码**，逐个记录V1 before与V2 after身份并由审查覆盖；模型、原始日志、历史报告不能藉此更新。同步前检查所有待推提交及blob，起点是已核验的V1/前版Push锚点，不把新登记时HEAD当成“已经发布”。不接管用户暂存区，不使用 `git add -A`、merge main或force push。

测试证据必须绑定当前源码与当前测试SHA，真实测试数>0且零错误/失败/optimizer调用；审查证据绑定同一登记身份。阶段报告的 `PASS` 字样本身不能证明完成。提交后保存 `SYNC_PENDING` 收据；远端SHA与本地提交核验一致才写完成状态。公共文档只记录快照，实时完成和远端SHA以本地核验收据为准，避免文档包含自身commit的循环问题。

## 给项目负责人的总结

每天仍然只需检查、恢复、核验，再启动已登记的离线维护。以后新增普通研发阶段有独立清单和审查流程，不必破坏旧协议。没有新增任何训练权限；想开展Pilot或正式训练仍需明确预算授权。恢复不等于续训，当前仍是 `RESTART_REQUIRED`。
