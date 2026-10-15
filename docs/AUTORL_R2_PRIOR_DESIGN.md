# AUTORL-R2-PREFLIGHT：解析Actor先验设计与实施计划

## 目标与约束

R1两臂最终均0/6成功，不能继续把课程排序当作已证实的解决方案。此阶段只验证原Tanh Actor能否近似已成功的state-only反馈；不训练、不调用optimizer，不改Reward、CNN、原物理环境或历史文件。此阶段本身不获得任何新训练预算，独立累计账本保持原131072步消耗。

候选比较：继续延长同一课程没有改善证据；改变Gaussian std可以研究探索但同时改变先验会混淆归因。先选择仅改变Actor初始均值映射，保持std=1、原结构和其他参数不变。解析赋值不是BC训练，也不是LiDAR避障能力。

## 数学与成功门

原Scripted归一化动作是clip([ex/4.5,ey/4.5,ez/3,yaw_error],−1,1)。原Actor两层64维Tanh，不换ReLU、不加skip。

令y为上述未clip反馈，alpha=.2、beta=.5。前四个神经元实现f(y)=10*tanh(.5*tanh(.2*y))；动作边界后clip(f(y))近似clip(y)。f单调且0≤f'(y)≤1，正半轴y−f(y)不减；区间[0,1]最大误差1−f(1)≈.016315，区间[1,+∞)clip后误差不超过同值。预注册全域动作最大绝对误差≤.02。不声称精确等价。

CNN特征64，随后state7、heading2、age1；只连接state位置误差和yaw误差。仅替换Actor的前4条特征路径及action head；其余Actor隐藏权重保持原随机，Critic/CNN/log_std完全不改。没有障碍真值或未来观测，初始均值不依赖LiDAR。首次赋值必须拒绝已更新模型与不支持的结构。

物理可行性门：三条冻结empty Train、每条最多375策略步，使用原环境原成功标准；每条最后5帧距离<.1m且速度<.15m/s。若失败，不以改成功标准补救。此阶段不运行Validation或Test，后续训练需独立协议及运行前GO。

## 实施计划（native执行；按用户自主授权，无需重复确认）

使用writing-plans、TDD和独立只读审查。

- [x] 先新增真实Torch/SB3测试并记录RED：缺失解析初始化；拒绝优化过模型、错误结构；Actor唯一变更；有限／全域误差；LiDAR及heading不改变初始均值；无optimizer调用。
- [x] 新增`research_regression/authorized_rl/actor_prior.py`的`initialize_actor_prior(model)`与只读公式配置；不修改旧训练worker。
- [x] 新增`prior_preflight.py`受限无学习检查：三条冻结Train物理轨迹，保存每步原观察／动作／距离／速度／LiDAR年龄、历史身份、模型赋值前后SHA；Adam.step、PPO.learn/train tripwire。
- [x] 为本新阶段prepare登记，独立只读审查源码／测试／方案；C/I未解决不运行物理probe。
- [x] register后执行真实测试和三个Train probe，原源码与数据身份不变、累计账本不变；统计真实wall/RSS，保存local-only轨迹和小型图。
- [ ] 形成中文MD／JSON／长期进度，独立结果复审，按白名单Commit/Push及远端核验，再决定是否开发R2独立PPO实验。

## 审查重点

Tanh而非ReLU；初始能力不冒充学习；未更改std/CNN/Critic；未通过未来帧、地图或Test取巧；原R1全部保持；任何PPO更新只能在新冻结执行协议与累计预算预留后。
