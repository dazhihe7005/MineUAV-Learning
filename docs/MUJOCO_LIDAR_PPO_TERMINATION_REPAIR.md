# Phase 3E-L9R — PPO终止记账修复与失败证据加固

本轮仅完成离线修复与验证：`TERMINAL_ACCOUNTING_FIXED=YES`、`FAILURE_CAPTURE_READY=YES`、`FORMAL_RETRAIN_AUTHORIZED=NO`。没有执行PPO/其他optimizer更新，没有Pilot、正式训练、旧checkpoint续训、PX4、FAST-LIO2或Test episode。原L9实验继续保持 **FAIL**，不能把本次测试通过回填成原训练成功。

基础提交 `c1c8d4f288a75a539b8721d4f991c613b830605f`；分支 `fix/lidar-ppo-terminal-accounting`。完整字段与身份见 [审计报告](../mujoco/reports/mujoco_lidar_ppo_termination_repair.json)。

## 根因及最小修复

**Confirmed：消费者契约错误。** 原 [NavigationEnv](../mujoco/lidar_integration/navigation_env.py) 按物理原始原因产生 `physical_failure=True` 和既有物理失败Reward，但保留 `termination_reason=z_below_zero/outside_flight_area/excessive_tilt/nonfinite_state`。旧TrainingMetricsEnv要求这个字符串本身属于五种规范类别，合法失败因此抛出 `invalid or nonexclusive training episode outcome`。原失败并不证明出现多个互斥标志。

先用真实MuJoCo环境在reset之后受控设置越界状态，获得唯一 `physical_failure=True`，复现旧异常（RED）。随后才修改记账层。**Unresolved：原L9具体物理触发原因。** 当时失败transition未持久化，现有数据仍不能确定是哪一种原因，不能据新故障注入倒推它。

新增 [terminal_accounting.py](../mujoco/lidar_integration/terminal_accounting.py) 的 `classify_terminal`，共享于训练记账、Health与Validation。仅修改三个既有消费者：`ppo_training_orchestration.py`、`ppo_formal_seed0_health.py`、`ppo_formal_validation.py`。没有修改NavigationEnv、Reward、Observation、终止条件、terminated/truncated语义、碰撞优先级、传感器失效机制、PI、动力学、CNN或PPO超参数。

| 原始原因 | terminal_class | 必须匹配的标志 | terminated / truncated |
| --- | --- | --- | --- |
| success | success | success | True / False |
| collision | collision | collision | True / False |
| time_limit | time_limit | timeout | False / True |
| sensor_failure | sensor_failure | sensor_failure | True / False |
| physical_failure或四种明确物理原因 | physical_failure | physical_failure | True / False |
| None（运行中） | None | 全部False | False / False |

两个新诊断字段为 `terminal_class` 和 `raw_termination_reason`；原 `termination_reason` 不被覆盖。检查所有五个显式布尔标志，未知原因、缺字段、多个标志、非布尔值、原始原因/标志不一致、terminated/truncated错误或已有元数据矛盾均fail-closed。未知字符串不会被猜成物理失败。校验通过之后才累计已返回步、物理tick、Reward及终止计数。

## 故障注入与RED→GREEN

新增57项测试中，受控MuJoCo用例覆盖success、collision、time_limit、sensor_failure及四种物理原因。逐个与未加记账包装的同种环境比较：Reward与每项breakdown完全一致，Observation逐元素一致，终止语义不变，原始原因保留，每次恰好计入一个类别。依次创建环境，定向测试不并行运行训练worker。

所有人工设置明确是 **FORCED_TEST_STATE**，不是自然训练失败，也不是导航成功率评价。负高度用例检查积分前不变量，保留reset的无接触集合；另一个调用mj_forward的真实地面接触用例验证collision优先级不变。time_limit注入末次step计数，并非新完整15秒训练episode。NaN分别注入ctrl、位置与四元数；非法四元数可能使底层包装直接抛异常，不得假称已返回完整transition。

新增严格分类器、记录器、Health集成、新协议禁止训练、全局序列化预算、Validation非有限位置与记录、Health最终距离、只读历史审计以及测试更新tripwire均先出现对应RED，再得到GREEN。原Bug的RED调用堆栈和阶段日志保存于独立ignored目录：

`mujoco/reports/mujoco_lidar_ppo_termination_repair_parts/run_01/`

修复后的训练Health/Validation在真实位置非有限时，轨迹长度、最终距离等相关诊断为 `NOT_AVAILABLE`，不是0，也不是把环境内部回退距离当成测量。聚合只使用可用数值并记录可用样本数量。Reward计算不变。

## 异常现场持久化

新增 [failure_evidence.py](../mujoco/lidar_integration/failure_evidence.py)，训练Driver为独立run/attempt创建记录器；Validation复用同一契约、可接收同一记录器。异常导致当前训练包装实例锁存，随后step/reset不能自动清除故障。

记录run_id、attempt_id、唯一event_id、调用编号、最后完成step、场景SHA、仿真时间/时钟域、两个终止字段、terminated/truncated及五个标志、最后动作、当前qpos/qvel/ctrl、LiDAR状态、物理tick、最后已知Reward、异常类型/消息/traceback、当前RSS和MemAvailable，以及上下文类型/来源。

- **RETURNED_TRANSITION**：底层确实返回五元组时记录其reward/info；分类失败时规范类别仍为NOT_AVAILABLE，原始原因照实保留。
- **LAST_KNOWN_CONTEXT**：底层step未返回时只记录能够直接读取的当前MjData和此前已知内容。info、返回Reward、原始原因等缺项写NOT_AVAILABLE。该状态不是完整transition，也不声称是故障发生前的快照。
- NaN和Inf以明确字符串保存，JSON使用 `allow_nan=False`。
- 上下文字段白名单，共享512节点/8192字符预算，深度6、映射64项、序列128项上限；有界迭代不复制整个映射。超限明确标记截断，不可挤掉独立异常元数据。异常文本、最多32帧traceback另有限长，最终文件不超过262144字节。
- 先写唯一临时文件并fsync，再以 `os.link` 原子发布，已有路径拒绝覆盖，随后fsync目录。同一个异常对象的外层重复捕获不再写第二份文件。
- 记录写入失败必须中止，并保留原异常为cause；不会因诊断失败继续运行。原L9目录明确禁止作为诊断/训练输出目录。

**覆盖边界：** 训练step、分类、后置Health记录及Validation episode内部异常已覆盖；不声称全面hook训练初始化/reset、PPO optimizer自身错误。SIGKILL、突然断电或存储不可用仍可能无法落盘。有限诊断不是完整MuJoCo/控制器/rollout恢复快照；仍为RESTART_REQUIRED，不能精确续训。

## 原L9证据只读审计

[analyze_termination_repair.py](../mujoco/lidar_integration/analyze_termination_repair.py)逐项核对原报告记录的11个raw文件SHA集合、历史报告/文档与基础Git对象、旧运行源码Git对象、七个历史checkpoint和7kg模型。21个受保护文件与修复前基线逐字节身份一致。没有补写formal.complete、没有覆盖JSONL/checkpoint、没有删除formal.attempt。原工作树87个untracked保留。

706条完整Train episode重新统计为 **656 collision、50 time_limit、0 success**，不含抛异常transition。平均return −29.389149，轨迹长度3.222638m，最终目标距离1.785219m，动作范数1.437426。12个已结束episode场景身份全部属于Train；Test episode为0。

唯一100096步里程碑：

| 指标 | 实际记录 |
| --- | --- |
| loss / policy gradient loss | 10.466864 / 0.000424378 |
| approx_kl / clip_fraction | 0.0000675833 / 0 |
| value_loss / explained_variance | 27.665215 / 0.662545 |
| mean action norm | 1.441488 |
| mean trajectory length | 3.160586m |
| mean final distance | 1.754031m |

仅一个诊断时点，不能充当连续学习曲线。0成功不证明PPO永远学不会，也不支持宣称已学会避障。下一次若获得独立授权，最值得跟踪的是完整失败现场/终止分型、无进展超时、动作幅度与近目标制动、轨迹/最终距离，以及跨里程碑Value拟合和KL/clip诊断；本轮不据此修改Reward或超参数。

## 新执行身份，不回填历史

旧L8/L9协议SHA仍为：

`fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6`

新增 [ppo_termination_repair_protocol.py](../mujoco/lidar_integration/ppo_termination_repair_protocol.py)，保留旧payload全部字段，在外加修复revision、终止/失败记录契约、9个源码SHA及 `formal_retrain_authorized=false` 后生成新协议：

`e267406ec4203353b4ee0c7f8349b57759ef64b31bd076c3db57b99b3682678f`

完整配置及逐文件SHA在报告中冻结。报告保留Python原始JSON数值类型（尤其ent_coef=0.0），落盘后重新加载并用原protocol_hash算法验证；独立结果审查曾发现生成器写成整数0导致哈希失配，已新增RED→GREEN成品回归，未改变哈希算法或训练配置。Reward、网络、物理、频率、24000射线、场景、动作尺度、PPO超参数、原预算、Final Checkpoint与RESTART_REQUIRED不变。当前Driver即使调用旧allow_formal_training=True，也在环境/模型/attempt创建之前拒绝训练；修改内存中的授权字段还会因协议SHA不匹配被拒绝。旧CLI保留旧身份/输出防线，不能把本次修复伪装成旧实验继续执行。未来新正式训练需另行授权及明确的新准入协议；本阶段没有执行新协议下训练。

## 回归范围、资源与审查

最终一次完整**选定范围**回归：**777 PASS = 720已有 + 57新增，0 FAIL、0 SKIP、22 deselected、2继承warning**，wall 102.132s。分组为Bridge514、LiDAR210（含新测）、MID360接口审计6、Shadow30、旧控制器/环境17。这不是“完整历史全部通过”或全仓测试；排除的22项及原因在报告逐项列出，包括会执行optimizer更新的测试、旧版本源码hash/授权契约测试及仍期望旧Bug的测试。没有删除旧测试，原失败版本可以在其历史提交审计；新版本由新契约测试验证。最终原始结果为独立local-only release_verification.json；先前775/776项完整运行的记录也保留，未冒充最终结果。

可复现的单线程验证入口 [verify_termination_repair.py](../mujoco/lidar_integration/verify_termination_repair.py) 对PPO.learn/train和torch optimizer.step设置更新前tripwire。完整回归没有意外调用；tripwire自身单测有一次被阻止的SGD.step，参数不变，**实际optimizer更新0**。没有运行新的Pilot、Smoke或正式PPO。相关历史测试内部的短物理/Scripted回归不是正式性能实验。

首次广泛回归为767 PASS、6 FAIL：临时外围内存监控线程触发既有Bridge“single-threaded runner”保护。代码证据已确认；只改变验证仪表化，改为setup/teardown单线程采样，保留旧保护后重新完整通过。该次失败记录保留，不算PASS或INCOMPLETE。

最终回归进程峰值RSS **1890.613MiB**，测试生命周期采样最低MemAvailable **8297.098MiB**；全部已测回归进程的最大RSS为2007.945MiB。一个worker，没有观察到OOM，未改swap。采样不是连续监控；早期独立RED进程不在这一测量窗口，整个任务所有进程的统一峰值NOT_AVAILABLE。序列化RED曾证实旧代码可构造约1GiB JSON；该失败路径随后以全局预算修复，不应重复旧压力路径。

软件MuJoCo3.14.0、NumPy2.2.6、SB3 2.9.0、PyTorch2.14.0+cpu、Gymnasium1.3.0。继承的Matplotlib Axes3D、Gymnasium未注册render警告保持披露。系统cv2/NumPy ABI信息仍由SB3可选Atari导入路径产生，测试使用CPU非Atari链路；不修复或升级无关系统依赖。

独立只读代码及最终结果/文档复审 **GO，0未解决Critical/Important**。最初指出全局序列化/元数据保持及Validation非有限指标两项Important，均RED→GREEN修复并再次复审；结果审查再发现JSON浮点类型身份问题，补充成品回归后关闭。最终审查独立核对777项原始结果、22排除、协议roundtrip、9项运行源码SHA、21项保护文件及文档资源数据。此GO仅认可离线修复交付，不授权重训或飞行；未以自己的测试代替独立认证。最终测试后没有修改运行时代码。

验证命令：

```bash
MINEUAV_LIDAR_UPSTREAM=/tmp/mineuav-mujoco-lidar-audit \
PYTHONPATH=/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco:/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco/rl:/tmp/mineuav-mujoco-lidar-audit/src:/home/keyousuke/MineUAV-Learning/.venv/lib/python3.10/site-packages \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -u -m lidar_integration.verify_termination_repair
```

本阶段到此停止；不自动重训seed0，不自动开始seed1/2，不运行Test，不接入PX4。
