# AUTORL-R4：冻结LiDAR-PPO目标反馈与减速响应诊断

## 【1】阶段目标

在继续花训练预算前，检查R1/R2/R3冻结策略的确定性控制反馈。区分目标方向、近目标速度指令、速度敏感性、Yaw响应与Gaussian噪声；不是新的导航任务评价。

## 【2】技术背景

R3真实两臂各65536步完成，最终Validation均0成功6碰撞；随机组raw动作越界已降到.625%仍失败。不能仅归因噪声、Reward或学习率，更不能直接重跑百万步。R3已推送并远端核验`dafc0fa117913ad40a1bbb9057b7c6b7b4767e06`，旧结论不改。

## 【3】系统架构

原第一个Train empty环境一次reset→真实24000射线／原AngularRangeEncoder→保存独立快照→关闭MuJoCo→11份固定checkpoint逐个只读加载→同一62组受控输入的确定性推理→源数据／响应图／诊断。物理step、episode、Validation、Test及optimizer均0。

## 【4】实现

`feedback_spec.py`限定62输入及11份summary／complete／ZIP／sidecar／policy身份；`feedback_probe.py`检验输入／Mask、提取原始Gaussian均值／std、裁剪动作、Critic及原动作尺度的指令表示，并核验权重／optimizer／Torch RNG不变。`feedback_run.py`复用flock、现有独立Linux lifetime guard、资源保护和新OPS阶段登记；只允许独立新目录执行一次，无训练或任意模型路径入口。

运行时PPO.learn/train与Adam.step被tripwire禁止。唯一原始快照与所有输出local-only，预算字节前后核验；数据产生失败保留attempt，不重试、不捏造完整结果。没有修改原训练器、Reward、CNN、控制器、动力学或授权。

独立预审_01发现背景身份链Important，故未执行。_02补三文件写后SHA/size、实际内存观测dtype/shape/bytes digest与NPZ读回一致性，完成前复核；损坏/替换阻断complete。对应RED为两项真实失败，GREEN为三项张量测试通过；旧候选和NO_GO保留。

## 【5】算法与指标定义

输入仍为LiDAR[3,12,24]、state7、heading2及age1。背景LiDAR／heading／age来自真实reset。只替换state：X/Y各7误差×3速度，Z为5误差×3速度，另5 yaw误差，共62个case、59个不同state向量（零state用于四个分组）。标记`OFFLINE_COUNTERFACTUAL_INPUT`，不是62帧新观测或真实运动轨迹；时钟不重写，所有case引用同一源采样时刻。

事前描述指标：零误差零速度的速度指令范数；零速度下各轴非零目标误差与该轴动作方向同号的比例（X/Y各6，Z为4）；误差固定+.1m时正负速度输入的指令差；Yaw ±.5rad的输出差／1rad；确定性均值越界及裁剪触边比例；与原state-only Scripted的最大裁剪动作差。全部是固定输入响应，不是物理减速度、稳定性或避障成功率。Scripted不是避障专家。

原始均值、std与clip后的动作分别记录。换算[1.5,1.5,1.0]m/s、1rad/s只说明旧接口指令单位，未向控制器发送。Critic数值没有实际return真值配对，不据此宣布估值正确。批量推理时间含均值／Critic／predict，不是单次25Hz控制链实时验收。

## 【6】诊断协议

ID `AUTORL_R4_FEEDBACK_ONLY`，CPU单线程、一个快照环境，生成后关闭；单次壁钟guard120s。11份模型：R1 UNIFORM32768／65536；R2 PRIOR与R3 RANDOM／PRIOR各0／32768／65536。四个固定summary及checkpoint身份见预注册JSON，R1没有step0，不补造。全部seed0，不能当多seed。

执行目录`.research_local/autorl_r4_feedback/run_01`独占；无训练预算预留，当前用户累计余额1,672,320步／68,400s不得因本次诊断刷新。MemAvailable最低1536MiB、磁盘2GiB；历史／grant／journal／模型及源身份必须一致，任何缺失拒绝。不启动PX4、FAST-LIO2、设备、训练或Test episode。

最终冻结：登记`42dcf1b3b64a7e15d6a7bc46d8d5c272ba9dfcf0a3ad5f8d22922fe1f3c000a2`；源码`d789aa040da93c3883f8ac444c4827017c765a9c30065bd44d98e0d9ee8d2a27`；诊断协议`b437910773b28628d7d6c9a7cd9372fc8dfb0bfbce7c83e2911ed0f753aae29f`。_02独立预审GO后，91 OPS／16 Tensor／885历史全部终证据完成、准入核验PASS，再register／recover／doctor PASS，最后唯一执行；不复用_01准入。

## 【7】实际结果

工程诊断PASS。唯一result SHA `83d3766ebdba586282f23e2b3c39a4284cc03791a4017d963af2fa04cef7a673`；实际62case×11checkpoint=682组输入响应，只有一个真实背景快照，并非682帧或episode。所有权重／optimizer／Torch RNG不变，clip(raw_mean)与SB3 deterministic predict完全一致。预算grant/head/events字节、历史checkpoint及原物理模型保持；87个历史untracked不变。

以下均是固定背景反事实输出，不是实际速度／物理减速度。方向同号比例的X/Y分母6、Z分母4；不是导航成功率。

| 固定模型 | step | 零state速度指令范数 m/s | X/Y/Z方向同号比例 | Yaw输出差／1rad | 均值越界元素比例 |
|---|---:|---:|---|---:|---:|
| R1随机std1 | 65536 | .3650 | .500/.833/.500 | -.00762 | 0% |
| R2解析std1 | 0 | 0 | 1/1/1 | .99586 | 0% |
| R2解析std1 | 65536 | 2.3452 | .500/.500/.500 | 0 | 98.79% |
| R3随机std.3 | 0 | .00882 | .667/.667/.500 | .00225 | 0% |
| R3随机std.3 | 65536 | .33091 | .500/.833/.500 | .01269 | 0% |
| R3解析std.3 | 0 | 0 | 1/1/1 | .99586 | 0% |
| R3解析std.3 | 65536 | 1.56468 | .667/.500/.500 | .62151 | 31.05% |

R2/R3解析step0本探针最大归一化动作误差相对Scripted均.002073以内。训练后两者偏离；R3解析final在目标误差+.1m时，正速度输入相对负速度输入，X／Z速度指令反而增加.58833／.14286m/s，Y差0。R3随机final对应差为−.01661／−.01600／+.00249m/s。仅描述有限输入敏感性，不据此认定稳定/不稳定或实际制动效果。完整11checkpoint指标见JSON及四图。

运行内wall8.335s（不含启动前完整身份核验及后续绘图），worker ru_maxrss1306.43MiB含模型加载，最小采样MemAvailable9013.98MiB；批量62case的均值／Critic／predict计算2.913–4.275ms，不是单帧25Hz端到端预算。无新增训练与物理step，未观察OOM；未新测进程CPU利用率。历史选择回归峰值2051.50MiB是另一个测试进程，不能混作本诊断RSS。

当前源码992个唯一相关测试PASS=91 OPS+16无更新Tensor+885历史；15项逐项排除见原始receipt，非全仓。后者124.73s、2既有warning；cv2/NumPy可选ABI stderr与Axes3D问题继续披露，不全局升级依赖。测试/旧版本复测不当作额外训练或重复科学实验。最终独立结果复审与远端SHA收据为发布完成条件，见机器报告。

## 【8】问题分析

CONFIRMED：在同一受控背景下，解析先验训练后的零目标动作偏置和方向响应退化；随机std.3也没有学到覆盖正负目标误差的充分方向响应。低均值越界不代表正确反馈，与R3全碰撞结果兼容。SUSPECTED：原任务的目标到达/停稳信用分配较难，解析参数化对同PPO更新敏感；需分别控制验证。UNVERIFIED：哪个因素是唯一根因、近距离课程或降低学习率是否有效；本诊断不提供闭环因果证明。不能以偏置数值断言实际漂移，也不能用一个背景排除感知条件作用。

## 【9】局限

一个Train场景背景、有限反事实输入、单seed11checkpoint；这些输入不是新的真实闭环轨迹。固定LiDAR未覆盖所有感知条件，不能外推LiDAR因果依赖、矿井泛化、PX4或实机。观测源是SIM_GROUND_TRUTH_STATE，不是定位健康认证。旧cv2 ABI warning保留，不安装新依赖；没有EXACT_RESUME能力。

## 【10】下一阶段

当前诊断已唯一完成，不再运行。结果复审及Git同步后，优先设计近距离到点课程：先核验独立短距离Train任务可完成性，再单独冻结任务距离分布，保持Reward／CNN／PPO／动作／物理不变，对照原R3随机std.3。新场景身份/课程/标准须在训练前登记与审查；这不同于R1仅切换empty/混合但目标距离未缩短。未冻结前不训练，不同时修改学习率；Test仍封存。

## 【11】给项目负责人的总结

这次没有再烧训练预算，而是把旧网络放到完全相同的雷达背景，逐项问它目标在左/右、自己正在前进/后退时会给什么指令。结果表明：先前手写的基本反馈经过训练后明显偏离，随机策略也没有可靠学会到点停稳。我们确认了值得研究的症状，但还没证明唯一根因，更没学会避障。下一步优先把基础到点任务拆得更近、更容易，先确认可完成，再做独立对照。原授权余额仍足够，不需普通阶段再确认；PX4及实机不在授权中。

图：`mujoco/reports/autonomous_rl_feedback_r4/figures/`下position_feedback、velocity_response、zero_bias、yaw_response，均来自实际冻结模型推理，非人工控制轨迹。
