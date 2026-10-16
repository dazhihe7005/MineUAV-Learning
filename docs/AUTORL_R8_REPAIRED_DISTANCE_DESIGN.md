# R8：修复后独立距离课程实验

基线05c7d9bd63b4e48b461a876407d7f4917e746fe6（R7完整1022项回归/独立GO/远端核验）。R6失败2f6e5c2永久保留，不重试、不加载其step0。新ID ARL_R8_REPAIRED_DISTANCE_SEED0、schema8、DISTANCE_CURRICULUM_R8_REPAIRED_V1，原授权账本不initialize/退款/扩额。

## 科学协议

独立新随机seed0/初始std.3继续可训练；65536实际步/3600s/CPU单worker。与历史R3 RANDOM std.3同初态db33ba4d…、同CNN/Reward/PPO/PI/动力学/24000rays，唯一研究变量为距离课程：原12Train起点/障碍/Yaw不变，32768前使用R5冻结0.20m任务，下一自然reset恢复原目标，不中途改目标。R6没有optimizer更新，不能作为学习对照；对照仍是固定R3 summary7651327c…，不重新执行。

Checkpoint0/32768/65536，完整rollout及对应optimizer后保存，0为未训练例外。关闭Train后独立short Validation6×3，原Validation在32k/final各6新执行，原step0六条复用同初态R3已发布证据明确标reused，实际新Validation30/Test0。Final选择，不用评价调本轮协议。

事前标准继承R6未检验假设：short final成功≥4/6、碰撞≤1/6、physical/sensor0；original final成功≥1/6且碰撞≤6/6；short final成功数>自身short step0分别判断。原成功distance<.10m/speed<.15m/s连续5policy step不变。单seed/少场景，不能外推避障泛化或学习LiDAR依赖。

## 实施及准入

固定新profile显式prepare-repaired-distance/start-repaired-distance，不接受checkpoint/脚本/seed/预算覆盖参数。复用原supervisor/flock/lifetime guard与原worker，schema8仅选R7适配器+R5 scheduler、旧std.3初始化和short evaluator；不复制PPO算法、不修改旧schema6。

固定pin R7 report ecbce36daebac0a561e88225430432679be85046f61101b6812989c511588efa、R7 adapter45dcc8f8…及其测试8472a947…；R5原可完成性proof/report、原shortmanifest和R3 control继续严格验证。训练前新source/registration/协议一致、profile和真实完整链/初始化测试RED→GREEN、完整无更新回归及独立GO、新登记/recover/doctor、实际余额/stop/资源核验后，唯一supervisor预留并执行。不使用R6旧GO，R7只认证离线修复，当前新运行入口需单独审查。

所有训练更新付费。当前余1606784步/64800s，预留后1541248步/61200s；MemAvailable≥1536MiB/磁盘≥2GiB，源码/证据/场景/NaN/未知终止异常即停止。单次失败保留唯一attempt、不重试、不退预留，RESTART_REQUIRED。原L9及R6始终FAIL，Test/PX4/FASTLIO/设备禁止。

## 收尾

实际结果分析、完整raw SHA/bytes/local-only清单、原模型/checkpoint/旧run保护、MD/JSON/五张真实证据图及长期文档、独立结果审查、Commit/OPS Push远端核验。若假设FAIL则明确记录；根据short32k/final及original迁移区别提出可反驳下一假设，不盲目重跑或增步。普通阶段不结束长期任务；先审计预算/stop/资源/会话能力再进入新安全工作。
