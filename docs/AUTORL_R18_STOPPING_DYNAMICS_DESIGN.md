# AUTORL-R18：冻结策略停稳动态诊断

R17已完整发布07f2113b44e25f473dada365d1487b5b4df0f984。50% empty重放未保留最终原域成功；不能从Val末误差反推速度或振荡。旧bounded_traces仅最初四个Train episode，每5step一行，不包含final Val完整轨迹。本阶段只补足这一证据缺口，不训练、不重复任何训练attempt。

## 固定范围

新ID AUTORL_R18_STOPPING_DYNAMICS_ONLY。使用三个只读checkpoint：R17@32768（与R16@32768权重同一）、R16@65536、R17@65536。单一冻结Validation empty身份fe85ac443a0dc1bd2c676b6455f28ef0a4d15ea046287b692cc7eb3f6eefe06f。每模型一次、最多375策略step，合计最多1125步评价；训练步0、optimizer0、Test0，CPU1worker、wall上限180s。不能把新诊断episode伪装成旧Val原始轨迹或新的独立泛化评价。选择该Val场景用于开发诊断，非Test。

## 因果与实现

复用CheckpointManager、verify_centered_policy、原HeadingObservation/LidarNavigationEnv/build_ppo_scene/终止契约与真实500/100/25/10Hz/24000rays；不改任何上述模块。每步先记录当前采样的state7/heading/age/仿真时间及qpos/qvel，然后deterministic=True推理，记录原Gaussian均值、裁剪后的归一化动作；实际env.step后记录真实command、position、world velocity、distance、speed、goal_stop、接触与Reward。pre与post分开，不能把动作和未来状态混为同一采样。

记录PI积分、期望加速度（明确post-control记录），保留机体姿态以辅助诊断。每步有真实LiDAR源时间/年龄；不得给网络加入障碍真值。当前环境位置/速度遵循旧源码协议，不声称为SLAM/PX4估计。预测前后校验模型/optimizer/训练步数不变，learn/train/Adam.step硬拒绝。每次仅一个MuJoCo，前一环境关闭后下一个模型；旧grant所有文件/预算前后逐字节不变。

只读复现核对：这三次诊断的terminal、policy_steps、final_error及return应与对应历史同一场景的Val汇总一致（数值容差1e-8）；若不同停止、保留差异，不编造一致。新轨迹不回填旧实验。共同32k不加载两次来凑模型数。

## 预注册分析

按实际25Hz轨迹统计到达<.1m的样本、其中speed>=.15m/s数量、joint样本及连续joint最大长度、首次进入目标与首次满足5步时间。停稳阈值与旧环境完全相同，不改验收。画distance/speed、声明command与实际world velocity、位置轨迹/PI积分及采样因果关系。报告越过目标的逐轴误差符号变化，但不能用少量往返直接证明控制不稳定。记录actor均值与动作裁剪，不能混成电机饱和。

解释只限三策略/一场景。若final在near-goal高速且joint不足，可支持“本例停稳不足”；若远处静止或持续位置偏差，另列事实，不能预设原因。没有模型训练成功率或闭环能力新认证。下一算法改进须由真实诊断再提出，另冻结独立协议。

## 测试与门禁

先RED：固定模型/场景身份、Test排除/缺raw拒绝、因果时间、有限数值、真实2步物理/射线/command协议、模型不变、计数边界。GREEN后当前OPS/Tensor/885历史零optimizer回归，独立只读精确GO。新OPS register/recover/doctor、无活跃attempt、stop=false、MemAvailable>=1536MiB/磁盘>=2GiB后才唯一独立进程组执行，flock及原lifetime guard复用。失败不重试。

结果MD/JSON/图和共享进度、独立实际结果复审、Commit/正常OPS sync远端核验，再继续下一安全阶段。保护旧272+新R17授权文件、七历史checkpoint、7kg原模型及raw；PX4/FASTLIO/硬件/训练禁止。普通设计与实施按用户长期委托自主进行，不新增授权。
