# AUTORL-R4：冻结策略反馈诊断设计与实施计划

> Native执行，TDD及独立只读审查。用户已授权范围内自主开展Level0/1研究，不另询问普通阶段确认；没有新增训练授权或预算。

## 目标与证据

R3减少初始std未改善最终导航。只回答：冻结policy在同一真实LiDAR背景下，如何响应位置误差、当前速度及目标Yaw误差？先检验已学到的反馈，而不直接重复训练。该探针不证明实际闭环稳定、物理减速度或避障能力。

## 数据／模型协议

复用原MultiScenarioTrainingEnv，只强制reset第一个冻结Train empty身份一次。保持真实MuJoCo raycast／AngularRangeEncoder及heading；保存这个独立快照。物理推进0、policy step0、episode结果0、Validation/Test0、optimizer0。生成后立即关闭MuJoCo，再逐个读取模型；不把qpos回放称飞行。

输入背景来自本次真实reset。受控替换7D state中的目标位置误差／世界速度／目标yaw误差，**OFFLINE_COUNTERFACTUAL_INPUT**；不修改观测源时间、不使用未来状态、障碍地图或Test。实际heading、LiDAR和age保持快照值。它们不是同一真实物理轨迹的多个新观测。

固定62个输入：X/Y轴各7个误差[-2,-.5,-.1,0,.1,.5,2]×3速度[-1,0,1]；Z轴5误差[-.5,-.1,0,.1,.5]×3速度[-.5,0,.5]；另5 yaw误差[-.5,-.1,0,.1,.5]，其余state为0。重复零输入用于不同轴的描述分组，不当新传感器样本。

只读取四个COMPLETE实验固定summary中的11份checkpoint：R1 UNIFORM的32768／65536，R2 PRIOR及R3 RANDOM／PRIOR的0／32768／65536。所有summary／complete／ZIP／sidecar／执行协议和policy权重身份须核验；不恢复环境、不调用learn，不重新执行旧episode。R1没有step0文件，不补造；R3随机step0与R1初始均值等价的既有张量证明仅作说明。

torch.inference_mode、deterministic=True。分别记录Gaussian原始均值、std、裁剪后归一化动作、原环境尺度换算的速度／Yaw-rate、Critic输出。验证clip(raw_mean)与SB3 predict一致、输入／输出有限、权重／optimizer状态不变。描述近零误差指令、误差方向响应、速度敏感性和横轴耦合；不将某个导数符号当作实际稳定性结论。原Scripted仅作为同输入的state-only参考，不是避障专家。

## 执行与保护

固定新ID `AUTORL_R4_FEEDBACK_ONLY`、独立`.research_local/autorl_r4_feedback/run_01`，使用原OPS新版本登记、同一flock、独立Linux lifetime guard及120s壁钟上限；显式专用Python模块，不开放任意脚本／模型路径。PPO.learn/train、Adam.step运行时tripwire，预算事件字节前后不变；MemAvailable>=1536MiB、磁盘>=2GiB，失败保留attempt、不得自动重放。原动力学／Reward／网络／超参数／checkpoint／旧实验不改。

## 文件与计划

新增`research_regression/authorized_rl/feedback_spec.py`（固定输入／证据验证）、`feedback_probe.py`（无更新模型推理）、`feedback_run.py`（唯一受限快照与执行）。新OPS／真实张量测试，不修改旧训练profile或训练器。

- [x] Task1：七项标准库测试先RED→GREEN，验证固定62输入、范围／NaN／额外特权字段拒绝、11 checkpoint覆盖、损坏／缺失证据拒绝、无任意模型入口及当前测试准入。
- [x] Task2：三项实际张量测试先RED→GREEN：原解析checkpoint目标方向／误差界、SB3 clip一致性；权重／optimizer／RNG不变、错误输入及Mask拒绝。新增实际观测NPZ与内存内容绑定、三文件运行时哈希封存／末尾复核，损坏或替换拒绝。测试本身没有环境step或optimizer。
- [ ] Task3：冻结源码和诊断协议，完整无更新相关回归，独立运行前只读审查GO、登记／doctor；唯一执行本诊断，保留raw输入／输出及source identity。
- [ ] Task4：真实结果图、11节中文报告／JSON、长期文档、独立结果审查、Commit/Push远端核验；再基于证据制定下一项单变量研究。

Task1输出固定case/checkpoint清单供Task2和Task3消费；Task2只允许确定性推理，Task3不引入训练新能力。Review Focus：反事实不冒充实测轨迹；均值／噪声／clip分开；固定checkpoint SHA在反序列化前验证；源时钟不续期；预算／旧证据不变且无PPO optimizer。

首次独立预审_01为NO_GO（C0/I1）：原始背景文件未在运行时绑定。执行前已修复并生成_02新身份，原_01候选／测试／NO_GO保留，不冒充已执行协议；_02需完整新测试和GO。没有先运行后补源码。
