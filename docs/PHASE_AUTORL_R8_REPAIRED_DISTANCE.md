# AUTORL-R8：修复后短距离课程学习对照

## 【1】目标

检验0.20m短目标优先的距离课程是否改善PPO到点/停稳。R6是未进入优化器的实现FAIL，不能当作算法结果；R7已离线修复/复审/推送05c7d9bd63b4e48b461a876407d7f4917e746fe6。本阶段新ID/新协议/新随机网络，不重试R6、不加载旧权重。

## 【2】背景

R5证明12个短目标在原动力学下可被state-only Scripted完成，但不代表PPO学会。历史R3 RANDOM/std.3的65536步最终Validation0成功6碰撞，作为固定历史对照不重放。R4有限反事实输入发现反馈偏置，不足以唯一归因Reward或学习率。本阶段只变目标距离课程。

## 【3】架构

独立累计grant→新固定schema8 prepare/start→原supervisor/flock/lifetime guard→原worker选R7任务记账adapter/R5 scheduler→原PPO/CNN/PI/真实MuJoCo和CPU射线→rollout更新后checkpoint→关闭Train→独立Validation。一个CPU训练worker、一个活跃MuJoCo；主管和轻量guard不是额外训练worker。

## 【4】实现

repaired_distance_profile.py固定新预算/ID、严格R7/R5/R3发布证据、当前回归准入；repaired_distance_training.py组合既有adapter和初始化，不复制算法。budget.py/runtime.py/__main__.py新增显式schema8固定分支；worker.py仅新环境/初始化分支和共享距离评价条件，不修改旧schema6行为。旧Navigation、Reward、PI、动力学、CNN源、R5和R7已验证helper均保持。初始化及bounded trace使用真实task ID，episode_details继续保留base、stage、target。

## 【5】算法

LiDAR[3,12,24]+state7+heading[sin,cos]+age1，SIM_GROUND_TRUTH_STATE；[-1,1]^4动作在原环境缩放[1.5,1.5,1.0]m/s和1rad/s。PPO128步rollout/64batch/2epochs/lr3e-4/gamma.99/GAE.95/clip.2/ent0/vf.5/maxgrad.5；std初始.3可学习，期望初始policy db33ba4d…。500/100/25/10Hz、24000 rays、NAV_V2_COLLISION_PREFLIGHT_V1不变。

## 【6】冻结协议

ARL_R8_REPAIRED_DISTANCE_SEED0，65536实际步/3600s/seed0/CPU1worker；REGa5c17d1466d52603580b903615b6957350c68c06e37b94f34714ee31bdd441e4，SOURCE62b3302122dc6d265b60112918082c2beb59239bc326eb6ab4d9c9c7de4a32a4，protocol2eeb62e185b5c1bfe9de5aa6a299dcc5e546248aaa7b88661667ae719a58efbe。

32768前均匀12short目标，其后下一自然reset恢复原目标；障碍/起点/Yaw和原算法不变，不中途改目标。checkpoint0/32768/65536，完整rollout和对应优化后保存，0仅未训练初始化。关闭Train后short6×3和original6×2共30实际新Validation；originalstep0六条复用同权重R3历史证据，明确reused、不计新episode。Final模型选择，Test0。事前分别评价shortfinal≥4成功/≤1碰撞/physicalsensor0、originalfinal≥1成功/碰撞≤6、shortfinal成功数>自身step0，原成功distance<.10m/speed<.15m/s连续5step。单seed不足以证明泛化/真正LiDAR依赖。

R7公开report SHAecbce36d…、R7源码/测试、R5 proof/report/manifest、R3 control均固定pin，不改旧协议。当前余额1606784步/64800s；如唯一启动预留后1541248/61200，全部原预留不退款。低于Mem1536MiB/磁盘2GiB、stop、源码/身份/NaN/未知终止异常即停止，RESTART_REQUIRED，不重试。

## 【7】结果

工程执行PASS，三项科学标准均FAIL。5项profile/CLI与1项实际初始化+完整记账链测试RED→GREEN，110OPS+33Tensor+885历史=1028项选择PASS，15排除/2既有warning。独立预审C0/I0、继承Minor1后新OPS登记/recover/doctor、资源/余额及prepare实际SHA全部核验，才唯一启动。本次真实65536步、512rollout、1024 PPO epoch update、2048 Adam调用，Actor/Critic均真实变化且有限；1308993实际物理tick，跨episode合计2617.986s仿真，不按每步固定20tick外推。

280个结束Train episode=180collision+100time_limit+0success/physical/sensor。SHORT128=76collision+52timeout，ORIGINAL152=104collision+48timeout；切换实际自然reset step33070，非假称32768恰好换目标。尾部122步未结束episode仍计预算，不混入成功率分母。12base访问18–29次、总281reset，全部24task属于冻结Train。原始23文件含3个checkpoint/完整计数与实际场景状态封存local-only；summary SHAa6760dc61ff03a3895eb451f99d61d77513c643a3e8d4f8e31a55f556475dce4，final ZIP54c95adb1211266c50d19c8d17b815804ab48c095b0739b3b189f8231d92bda6。Test0，没有重试。

| Checkpoint | 短Validation成功/碰撞/超时 | 原Validation成功/碰撞/超时 |
|---|---|---|
| 0 | 0 / 0 / 6 | 0 / 0 / 6（只读复用R3，非新episode） |
| 32768 | 0 / 6 / 0 | 0 / 6 / 0 |
| 65536 | 0 / 1 / 5 | 0 / 2 / 4 |

所有评价无probe_cutoff，最终physical/sensor0、真实新Validation30。短final平均误差1.07948m，较自身step0的0.20359m反而更大；原final1.82842m。相对R3原目标最终0成功6碰撞，R8减少4次碰撞但增加4次超时，成功仍0，不能宣称学会避障或导航。短候选≥4成功、原目标≥1成功、短final优于自身step0三项均FAIL；不降低标准。

短Train有21个episode曾进入0.10m球、265个近目标采样，最佳0.03409m；原Train只有1个/7采样/0.09555m。所有joint成功均未达到；缺完整逐步速度/停留轨迹，不能单独归因速度过高或保持时间不足。两阶段都没有符合既定motionless阈值的episode，但不能据此排除缓慢漂移。原32k目标阶段平均最小/最终误差与Reward均在JSON分期保存。

512条PPO诊断的approx_kl均值.0024946/最大.0177613，clip_fraction均值.0091248，raw动作元素裁剪均值.0023003；std最终.2963277。value_loss均值10.782/最大98.486、explained_variance均值.2051/范围−7.7056至.9956，最后.9339；图展示全部点，不能以末值证明整个训练Critic始终正常，也不能用小KL认定优化完全无问题。

主管wall720.338s、worker719.070s；训练health时钟wall656.281s/CPU646.050s，约99.86step/s。worker HWM1696.816MiB含checkpoint重载及deferred Validation；训练观察到HWM1642.180MiB，训练采样最低MemAvailable9614.242MiB；完整含Validation的全程最低值NOT_AVAILABLE。未观察OOM。envstep P50/P95/P99/Max=1.180/12.152/12.539/21.546ms，scan6.338/6.803/7.070/15.535ms，encode4.143/4.433/4.595/6.934ms；scan/encode26086条独立新增帧样本，不含reset/Validation，非重复每step复制。

三个checkpoint权重/optimizer/确定性预测重载验证通过；主管COMPLETE/账本第13RESERVE/14COMPLETE核验后才运行分析器。原grant及所有既有run文件字节未变（只允许新追加事件和head），SOURCE62b330…运行后仍一致。累计458752步/25200s保守预留，剩1,541,248步/61,200s，stop=false，不退款。

分析器审查最初NO_GO I2/M1（分箱时间因果、仅worker complete不足、内存标签），已在出图前修复，合成契约RED3→GREEN3，源审GO后主管完整才分析。五图来自完整真实raw；分箱结果只在右端可用，离散率用阶梯图。初版未发布图的长标签视觉问题另行重排，v1派生图/源/分析JSON保留本地，v2为最终派生版本；原训练raw不变。最终独立结果审查与Git远端收据见机器报告，本段不预先认证其结果。

## 【8】分析

CONFIRMED：R6仅身份集成错误，R8补齐后原PPO可完整执行；距离课程仍未产生严格成功反馈，连short32k评价都失败，因此不能把失败全归因后半段原目标遗忘。碰撞下降伴随超时上升，不等于导航提升。SUSPECTED：策略未形成稳定目标反馈/停稳，缺成功反馈和持续动作噪声可能有关。UNVERIFIED：唯一Reward/LR/Critic因果、完整近目标速度/保持过程；不因阴性结果自动增步。

## 【9】局限

小型固定Train/Validation、单seed/无Test；short目标可能在障碍前，成功不自动等于避障。冻结整帧LiDAR无真机逐点时间或FASTLIO；没有PX4/硬件。R7已知Minor为任意内部active_task结构破坏可能遮蔽故障context，正常冻结scheduler无此路径，不宣称任意内存损坏恢复。optional cv2/NumPy ABI与历史warning保持，无系统升级。不证明百万步内存稳定，无精确续训。

## 【10】下一阶段

先完成本次FAIL学习结果的独立审查/Commit/Push远端，旧ID不再启动。下一项候选R9是无optimizer的短任务解析Actor反馈预检：保留R8距离分布，只检验已有R2解析均值在12short真实物理下能否停稳，区别手写先验与学习，不重复R5 Scripted基线或R4旧checkpoint诊断。若先验可行，再独立冻结仅改变初始化、与R8随机初始网络比较的新PPO实验；short初态如果已成功，必须区分保留先验能力和真实学习增益，不能用手写成功充学习。仍须新协议/测试/预审/余额/stop/资源核验，不修改Reward/CNN/PPO，不加载旧checkpoint继续。

## 【11】负责人总结

这次完整训练跑通了，但把目标缩短到20cm仍没让PPO按原标准停稳。碰撞少了一些，超时却多了，不能说学会导航。下一步先免费做有限的非学习反馈预检，检验“正确到点反馈能否在同一短任务中工作”，再决定是否值得做仅改变初始化的新训练；不会重跑本次或偷偷加预算。余额约154万步/17小时保守wall。旧失败与原模型不动，普通研究继续自主推进，不需要你管理每个步骤。
