# AUTORL-R13：状态归零锚定 Actor 设计与实施计划

## 设计意图与候选比较

R12固定背景诊断已发布于79befa26f46f3b8bedd55320eed3bb3869dc5518：低全局LR仍伴随累计均值偏置，不证明唯一训练根因。继续降低LR或探索std可能减慢学习且不直接约束该偏置；回归/BC约束会增加新的优化目标和预算解释。本阶段选择独立的Actor均值参数化预检，保留CNN、Critic、Reward、物理、PI、动作和传感器接口。用户明确要求自主设计/实施，无须逐个普通设计另行确认；独立代码与结果审查仍是硬门禁。

## 明确的结构

令共享特征x=[LiDAR64,state7,heading2,age1]，a(x)只将state7置零，其余使用同一当前观测特征。定义mu(x)=head(actor(x))−head(actor(a(x)))。两次Actor MLP共享参数、一次CNN计算；Critic仍只接收真实x。不是把虚构零state发送给环境或EKF，不读取未来/地图/障碍真值。

新StateCenteredPolicy继承SB3原MultiInputActorCriticPolicy；super构建原网络及Adam后，以不增加参数/不改变参数对象和顺序的包装器替换MLP及head。MLP输出两组64D Actor latent，head求差，所有SB3forward/get_distribution/evaluate_actions/predict路径统一。原LidarStateExtractor与74D拼接不变；Gaussian std仍原分布。初始seed0按既有手写先验赋值，再std=.3，学习率3e-5。初始均值和Critic应与R11初态数值等价；新类/键名的完整SHA必须不同，另提供规范键名参数SHA核对6312e99c…，不冒称同一策略函数。

## 范围与风险

R13训练step/optimizer/物理/新扫描/Validation/Test全0，不预留训练预算。不加载旧训练checkpoint初始化；仅序列化/重载本阶段新建并赋先验的策略，单测用临时目录，唯一预检run保留新初态ZIP于local-only目录并记录SHA。偏置/非有限/错误结构注入均为FORCED_TEST_STATE，不能冒充训练现象。零state均值为零是结构恒等式，不是学习成功；接近目标仍可能高速、随机探索、碰撞。结构限制也可能损害障碍附近动作表达，需要后续独立、已审查的真实PPO对照。无非空Adam状态恢复实测，仍RESTART_REQUIRED。

## 实施计划（native，先RED后GREEN）

### 1. 统一Actor实现与先验兼容

- [ ] 新research_regression/authorized_rl/centered_policy_tests.py：先写并运行缺失模块RED；断言任意当前背景零state均值为零、受控常量head偏置抵消、非零state反馈存在。
- [ ] 新state_centered_policy.py：CenteredMlp/CenteredHead/StateCenteredPolicy、固定make_centered_ppo、canonical_policy_sha、initialize_centered_prior与结构/参数身份核验；旧make_ppo/actor_prior代码不改。
- [ ] 初始规范SHA与冻结R11一致，所有62个既有反事实输入上的均值/std/value/log_prob与新建旧协议初态相符，单条/批量正确。

### 2. SB3梯度、序列化及防错

- [ ] RED→GREEN验证forward/get_distribution/evaluate_actions/predict一致；梯度有限且Actor/Critic均有非零梯度；不调用optimizer.step。
- [ ] 临时save/PPO.load保持策略类、参数顺序/数量、真实LR、std、输出一致；所有参数均在Adam组且无重复，包装不脱离优化器。
- [ ] 不兼容结构/已优化或非seed0初始化拒绝；NaN诊断不伪装有限结果；初态函数等价并不证明训练后性能。

### 3. 身份、准入与发布

- [ ] 新固定R13配置/OPS注册候选；完整当前operations/tensor/historical零更新回归，15历史排除明示；221授权文件及全部历史身份不变。
- [ ] 独立只读代码与结果审查。重要修复先RED→GREEN及受影响回归，禁止擅自开训。
- [ ] 中文11节报告/JSON/实际网络输出图，更新README与五份交接；显式提交/OPS sync/远端核验后才进入R14独立训练设计。

## 自审关注点与裁决

重点检查保存重载是否绕过包装、优化器引用是否失联、训练与推理是否使用不同均值、CNN是否重复算或混用未来背景、锚定是否被误称制动/学习成功。包装不添加参数，规范SHA只用于比较参数值，不代替新类的完整身份。中间提交不能绕过OPS白名单，本阶段统一在最终审查后提交；不删除技能工作区和历史收据。无模糊待定接口；后续训练单独冻结，不在R13执行。
