# AUDIT-01 离线学习失败诊断计划

目标：只读分析L8/L9/L9R/L9S及必要的L5已保存控制证据，区分运行中断缺陷与尚未学会导航。训练、optimizer、Test episode、PX4、FAST-LIO2及设备运行均为0。

## 设计与执行

- [x] 核验6258dd4基线与远端；按原V2运行status/recover/doctor；复用独立worktree并创建研究分支。
- [x] 清点真实JSON/JSONL/源码，逐文件SHA；只读验证L9失败报告、L9S complete、L8 summary及冻结场景清单，保留截断和缺失信息。
- [x] 先写纯标准库失败测试：分场景/分时间统计、Train身份拒绝、末端距离不冒充最小距离、单PPO诊断不冒充趋势、数学Reward边界、输入损坏和未来LiDAR观测拒绝。
- [x] 实现 `research_regression/navigation_diagnosis.py`，不导入模型、优化器、训练入口或运行MuJoCo；重算episode、Validation、PPO快照与时效证据。图表只用真实已有统计，不画虚构物理轨迹。
- [x] 以V2 prepare生成草稿；使用同一公开登记API显式继承OPS-02的17项历史测试before/after覆盖，生成新的最终候选清单（保留草稿及各SHA，不改旧协议）。源码与交付名单经独立只读审查后register。
- [x] 真实运行已登记的只读审计准入任务；分析器另以Level1研发工具在同一flock和冻结AUDIT-01身份核验下执行，输出唯一新目录。V2固定experiment-analysis能力只做历史报告来源登记，不冒充完整诊断执行器。
- [x] 新增中文阶段MD、JSON与小型图，更新长期文档，运行标准库分析测试和适用的无optimizer相关回归，独立结果复审GO（Critical/Important均0）。
- [ ] 使用已登记白名单与V2同步保护Commit/Push、远端SHA核验；结束本阶段，不训练。

## 证据边界

原L9永久FAIL；119552只是最后资源样本。L8/L9S是独立seed0短实验，不当作多个训练seed。按布局和运行阶段分层，不能把episode当独立seed。原始episode无逐步Reward组成、动作、位置、速度；只有源码推导可用时标明数学边界，不能称为实际分量重算。没有记录轨迹时明确NOT OBSERVABLE。

## 验证重点

输入损坏或身份不符时停止；原数据前后SHA一致；新JSON无NaN/Inf；图不连接不同attempt为训练趋势；最多三个候选、单变量对照和预算只是建议，不是授权；原管理代码及模型/Reward/动力学不修改。
