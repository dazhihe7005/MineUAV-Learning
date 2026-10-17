# AUTORL-R23：训练与开发验证分布的只读核查

## 有界设计与自主执行依据

依据长期自主授权继续零训练研究；不是新预算授权。R22已完整发布55594504a1e906940c4f1b7c5668de79ee8291be，科学FAIL保留。本阶段复用既有只读算术审计流程，不加载checkpoint、执行推理、扫描、物理或优化器，不运行新Train/Validation/Test episode。

## 研究问题与备选路径

1. 推荐：量化冻结Train/Validation的Yaw与任务分布，并对完整R17/R22的实际结束episode按身份、场景及目标进入情况归组。无需训练，先检验“Train成功但独立初始条件失败”的证据。
2. 固定输入推理面板可另查反馈，但旧R15仅涉及R14，不能复用为R22结论；当前先不加载模型。
3. 不直接重跑长训练，也不改Reward/网络；没有因果证据不能宣布Yaw是唯一根因。

## 输入与统计契约

只读原L6预注册中的12Train、6Validation；封存Test仅保留整个原文件的既有SHA，不读取其episode结果。输入为完整R17、R22的summary、complete、episode_details、train_episodes及公开发布报告。R20中断前缀不混入完整对照。

Yaw距离使用atan2(sin(a-b),cos(a-b))的绝对值，输出弧度及度；同时报告离所有Train和同layout Train的最近角距，不能用线性max-min跨±π。位置/目标及Yaw是审计数据，不输入策略。

结束episode按base Train身份归组；只统计ORIGINAL，不将short目标冒充原目标。保留总episode与实际policy-step分母、五类终止、最近/最终目标误差、目标区进入及joint停稳采样。未知身份/重复身份/错split/错layout/非法数值/矛盾outcome立即拒绝；未结束尾部单列，不补造终止、Reward或轨迹。

Val仅使用历史每checkpoint的六个真实样本，必须身份齐全、无重复、零probe_cutoff；按layout明确列出实际接触对。不同训练时期的随机Train统计不能当成与final deterministic Validation同策略、同分布的泛化差值。

## 可证伪结论边界

CONFIRMED仅指数字、身份与实际分布差异。即使发现Yaw覆盖缺口和失败相关，也只支持SUSPECTED，不能证明控制符号错误、感知根因或课程因果。没有逐step轨迹时侧绕路径、碰撞前动作及实际advantage为NOT OBSERVABLE。不得据此修改冻结R22。

## 实现与验收顺序

增加纯标准库distribution_audit.py及六项RED→GREEN行为测试；复用无模型算术worker模式，唯一R23输出与原子收据、flock、父守护、有限墙钟和资源保护。冻结注册/协议/全部授权文件SHA，完成当前源码绑定的OPS、历史及无优化器Tensor回归，独立预审后才运行唯一算术任务。生成三图、中文阶段报告与JSON；独立结果审查、显式白名单Commit、正常OPS sync、远端核验后继续下一项安全研究。

预算：训练0步/0秒、新episode0、PPO更新0、PX4/设备0。保留余额1082496步/36000预留秒，所有旧授权文件逐字节不变。科学输出是诊断证据，不是导航性能提升；下一学习实验需另冻结协议、审查并扣账，禁止旧attempt重试/ZIP续训。
