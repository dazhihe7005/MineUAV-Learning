# AUTORL-R19：既有轨迹的Reward信用与任务混合审计

R18及交付补充已完整发布4967f0abc53552730e660d8dbe5e2674709af138；其运行SOURCE保持，发布空白检查FAIL/未中止提交的Minor已公开独立处置，不能抹去。R18证明单个开发Val中final未进目标区，不能盲目加强制动。下一步用既有数据检验奖励排序与任务负反馈假设，不新增训练或物理采样。

## 固定数据及权限

新AUTORL_R19_REWARD_CREDIT_ONLY。只读取R18三条冻结模型的新诊断trace、R16/R17 ORIGINAL Train episode_details与summary、原L6清单；允许静态解析三份已验证ZIP的JSON元数据以核gamma=.99/GAE lambda=.95，不解pickle、不加载模型。train/optimizer/inference/physics/new episodes/Test均0，CPU1worker、wall120s；旧299授权文件及七raw/模型/物理身份前后全不变，预算1213568/43200不预留或退款。

计算核心仅标准库；既有OPS依赖探针可能在子进程导入Torch/MuJoCo以核版本，不等于加载策略或运行仿真。reset_exposure只解释已封存short/original身份序列，不宣称它是任意未知任务流的通用验证器。继承R7范围限制与R18已公开发布流程Minor分别保留，不能以本轮新Minor=0抹去历史。

## 预注册计算

逐条核实实际Reward等于实际reward_breakdown各项之和，保留progress/action/brake/success/collision/timeout_or_time_cost/physical_failure/sensor_failure，total不重复相加。对每条实际完整诊断轨迹计算G(gamma)=sum gamma^(step-1)*r_step，gamma固定[.99,.995,.999,1.0]。只有.99是已执行PPO参数，其余是离线反事实数值，不是新训练或相应策略表现。成功bonus有效贡献与未折扣原值同时报告，不能把GAE核或折扣回报当作实际未保存的PPO advantage。

冻结解释：若实际成功32k轨迹在gamma=.99下回报高于两个失败final，则“当前Reward偏好这两条已观察失败而非该成功”在此数据上被反驳；否则仅在三条轨迹上获得支持，不能外推所有策略。terminal success折扣保留比例<.10只命名为明显衰减，不是导航验收阈值。gamma与gamma*lambda的半衰期/有效几何衰减长度用25Hz换算，明确不是硬截断或严格因果学习范围。

读取两轮原域Train按布局重算结束episode数、终止类别、policy_steps、Reward各项。核sum结束steps+未结束tail等于65536减实际切换32850；尾episode按实际active_task base归属计入样本曝光，只加入步数、不伪造未结束Reward或终止。分别记录reset/episode比例与policy-step比例，不能将50% reset自动等同50%梯度样本。碰撞惩罚与成功bonus累计量只是奖励质量，不是梯度大小或优势方向。

## 判据与下一实验候选

数据完整性与所有恒等式通过才工程PASS；算法性能提升NOT TESTED。未知原因为NOT OBSERVABLE，不从单个Critic点推出整个趋势。最多推荐两个新单变量研究：后半程仅原距离empty的极端隔离（对照R17的.5，排除硬障碍负迁移；不是逐级搜索更多比例），或仅改变gamma的时间信用研究。必须由审计结果决定优先项，所有其他Reward/CNN/PPO/动力学/动作/时序保持对照，训练另冻结新ID/完整测试/独立GO/预算，不直接开跑。

## 实施和保护

先纯标准库RED→GREEN：手算折扣/total不双算、NaN/非法gamma/错奖励拒绝、场景身份和尾步记账、缺证据拒绝、Test与训练入口拒绝。复用原OPS登记/独占锁/lifetime/resource/stop及当前零optimizer回归，精确独立GO后唯一run。raw只读，结果新目录。完成图/中文11节MD/JSON/共享进度、独立实际复审、Commit/正常OPS sync远端，随后检查授权继续新研究。提交前检查需完整exit0后再单独commit，不能重复R18编排错误。
