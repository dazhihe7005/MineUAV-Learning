# 从 MineUAV-Learning 实际源码理解 PPO

本文件审计当前仓库和本机安装的 Stable-Baselines3 **2.9.0**，不是用通用教程替代实际实现。下文的训练代码仅被阅读，**本阶段没有运行 learn/backward/optimizer.step**。

依赖源码根目录：`.venv/lib/python3.10/site-packages/stable_baselines3/`（下面记为 `SB3/`）。其六个关键源文件 SHA256 保存于本阶段 manifest/report。对应上游版本可见 [SB3 v2.9.0 PPO 源码](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.0/stable_baselines3/ppo/ppo.py)；本地文件和已加载 checkpoint 是本次结论的直接依据。

## 1. 本项目实际训练入口

`mujoco/rl/train_ppo_pi_lowstd.py::run_training` 调用 `ppo_pi_env.py::make_pi_training_envs` 建立八个 `DummyVecEnv` 环境，然后使用 `train_ppo_waypoint.py::make_ppo`。这是当前可比的 **7D PI-PPO seed0**，不是早期 P 速度环 PPO，也不是 10D PI-observable PPO。

固定配置：Adam lr=0.0003、gamma=0.99、GAE lambda=0.95、clip=0.2、entropy coefficient=0、value coefficient=0.5、gradient max norm=0.5。每环境256步、8环境，共2048样本/rollout；batch256，10个优化epoch/rollout；初始四维 log_std=-2，初始 std=exp(-2)=0.135335。

最终100k里程碑实际收集100,352 transitions，即49个完整rollouts。SB3的 `_n_updates=490` 计数的是优化epoch，不是单个Adam step；在无提前停止、每批256样本条件下共有49×10×8=3920个minibatch optimizer steps。不要把八个环境当作八个独立训练seed。

`PIMilestoneCallback::record_updated_state` 在完成更新后保存20k/50k/100k里程碑并进行历史固定评估。这不是独立Validation/Test模型选择流程。本阶段在生成新目标前固定使用最终100k checkpoint，没有按新Test选模型。

## 2. Observation 与 action

`mujoco/rl/mine_uav_env.py::MineUAVEnv._get_obs` 输出float32 `[7]`：

| 索引 | 实际语义 | 单位/坐标系 |
|---|---|---|
| 0:3 | target_position − qpos[:3] | m，world XYZ |
| 3:6 | qvel[:3] | m/s，world XYZ自由关节平移速度 |
| 6 | atan2(sin(target_yaw−actual_yaw),cos(...)) | rad，wrap到[-pi,pi] |

Actor输入批量tensor `[B,7]`；没有VecNormalize，没有读取PI integral或未来状态。当前PI的隐藏积分/姿态/角速度没有被完整观测，7D不是严格Markov全状态。BC和主对照PPO具有相同信息条件，不代表具备全部动力学状态。

`mujoco/control/velocity_command_controller.py::map_normalized_action` 将 `[4]` 归一化动作映射为 `[1.5*a0,1.5*a1,1.0*a2]` m/s world速度命令以及 `a3` rad/s yaw-rate。环境合法范围[-1,1]。控制频率25Hz；PI100Hz；MuJoCo500Hz。Policy不输出PWM。

10D额外参考：`ppo_pi_observable_env.py::MineUAVPIObservableEnv._get_obs` 在7D之后追加三个实际 `Ki*integral_error` world加速度项。它观察BC不可见的内部状态，不进入同观测公平主对照。

## 3. Actor Network

实际函数：`SB3/common/policies.py::ActorCriticPolicy._build_mlp_extractor/_build/_get_action_dist_from_latent`，`SB3/common/torch_layers.py::MlpExtractor.forward_actor`。

FlattenExtractor将 `[B,7]` 保持为 `[B,7]`，随后独立actor MLP：7→64 Tanh→64 Tanh。actor latent为 `[B,64]`，最后线性 `action_net` 输出均值mu `[B,4]`：

`h1=tanh(W1*o+b1); h2=tanh(W2*h1+b2); mu=Wmu*h2+bmu`。

没有输出Tanh squash。初始化为orthogonal，隐藏层gain sqrt(2)，均值head gain0.01；并不是BC的ReLU网络/默认初始化。actor含均值网络4932参数，加4个可学习log_std，共4936。

## 4. Critic Network 与 Value Function

实际函数：`MlpExtractor.forward_critic`、`ActorCriticPolicy.predict_values/forward/evaluate_actions`。另一个独立7→64 Tanh→64 Tanh→1网络，共4737参数。Critic latent `[B,64]`，value `[B,1]`；PPO.train中flatten为 `[B]`。actor+critic+log_std总9673参数。

`V_phi(o)` 近似当前策略在此观测下的未来折扣奖励期望：`E_pi[sum_l gamma^l r_(t+l)]`。它不是专家动作，也不是World Model预测。7D部分可观测条件会使这个估计面临隐藏状态混合。

## 5. Policy Distribution 与 Action Sampling

实际函数：`SB3/common/distributions.py::DiagGaussianDistribution.proba_distribution_net/proba_distribution/sample/mode/log_prob/entropy`；`ActorCriticPolicy.forward/_predict`。

可学习log_std `[4]` 对所有观测共享；std=exp(log_std)，广播到 `[B,4]`。分布为四维独立Normal：`pi(a|o)=product_j Normal(a_j;mu_j(o),sigma_j)`。

训练时 `sample()` 使用PyTorch `rsample()`：`a=mu+sigma*epsilon`，epsilon~N(0,I)，输出 `[B,4]`。`log_prob` 按四维求和输出 `[B]`。虽然内部采样用了rsample，PPO采集在 `torch.no_grad()` 下进行；优化依赖重新计算的log-prob ratio，不是穿过MuJoCo的可微控制。

`SB3/common/on_policy_algorithm.py::OnPolicyAlgorithm.collect_rollouts` 将原始Gaussian动作clip到env Box，再执行环境；buffer保留**原始未clip动作及其log_prob**。这是此SB3实现的行为，不应误写成Tanh-squashed policy或人为改checkpoint。

本阶段 `BasePolicy.predict` → `_predict` 使用 `deterministic=True`，得到Gaussian mode=mu，随后clip到[-1,1]，单状态返回 `[4]`。不是从Gaussian随机采样，不混入teacher action。

## 6. Reward

实际函数：`mujoco/rl/mine_uav_env.py::MineUAVEnv._compute_reward`，当前分支选 `reward_version='v2'`。输入前后距离scalar、归一化action `[4]`、success/failure flags；输出scalar reward和breakdown字典。

`w(d)=clip((0.5−d)/0.4,0,1)`。

`r=10*(d_previous−d_current)−0.005*||a||²−0.5*w(d_current)*||v_world||²+10*success−10*physical_failure`。

V2没有V3额外距离惩罚，没有V4切向速度惩罚，也没有显式每步时间罚。Success必须距离<0.10m且speed<0.15m/s连续5步；timeout15s不算成功。速度奖励的near权重与任务阈值不是完全相同的判定。Reward设计存在任务对齐的潜在限制，但本阶段不修改，不据此断言唯一失败原因。

八环境一步reward为numpy `[8]`；buffer存 `[256,8]`。连续状态发生物理失败时仍保留失败样本。

## 7. Return 与 Advantage

实际函数：`SB3/common/buffers.py::RolloutBuffer.compute_returns_and_advantage`，调用者 `OnPolicyAlgorithm.collect_rollouts`。

理想折扣return是 `G_t=sum_l gamma^l*r_(t+l)`；此实现训练target并非直接截断Monte-Carlo return，而是 **GAE/lambda-return** `returns_t=advantages_t+old_values_t`。

Advantage表示动作相对于该状态价值基线的收益增量。概念上 `A_pi(o,a)=Q_pi(o,a)−V_pi(o)`，但实现使用下面的估计而不是精确Q。

## 8. GAE

同一函数从rollout末尾反向迭代：

`delta_t=r_t+gamma*m_t*V_old(o_(t+1))−V_old(o_t)`

`A_t=delta_t+gamma*lambda*m_t*A_(t+1)`。

`m_t` 来自下个状态 `episode_starts`，最后一步来自 `dones`，阻断跨episode递推。最后一个未结束状态value来自criticbootstrap。`collect_rollouts` 对 `TimeLimit.truncated` 且有terminal_observation的timeout先在reward中加入gamma*V(terminal_obs)，再保留done边界；真实termination不采用同样的timeoutbootstrap。

保存的observations `[256,8,7]`、actions `[256,8,4]`、values/log_probs/advantages/returns `[256,8]`，`RolloutBuffer.get` 使用 `swap_and_flatten` 变成2048样本，随机permutation后每批256。传给 `evaluate_actions` 的obs `[256,7]`、actions `[256,4]`；value/log_prob/entropy/return/advantage最终 `[256]`。

`PPO.train` 在**每个minibatch**中做 `(A−mean(A))/(std(A)+1e−8)`，不是全buffer一次归一化。原始buffer advantage不因此被永久替换。

## 9. PPO Clipped Objective

实际函数：`SB3/ppo/ppo.py::PPO.train`；动作似然由 `ActorCriticPolicy.evaluate_actions` 返回。`old_log_prob` 是采集时冻结数值，`log_prob` 是当前策略对同一buffer action的计算。

`rho=exp(log_prob_new−log_prob_old)`，shape `[256]`。

`L_policy=−mean(min(rho*A,clip(rho,0.8,1.2)*A))`。

负号因为Adam最小化loss。clip的是概率比，不是action、梯度或PI输出。对负advantage仍使用同一个min公式，不能擅自写成只限制正收益动作。`clip_fraction=mean(abs(rho−1)>.2)` 是诊断值，不是成功率。

## 10. Entropy Bonus

实际函数：`DiagGaussianDistribution.entropy` 与 `PPO.train`。

每状态四维熵求和，shape `[256]`：`H=sum_j(log(sigma_j)+0.5*log(2*pi*e))`。

`L_entropy=−mean(H)`。总loss中系数 `ent_coef=0`，因此当前训练没有直接entropy bonus推动；仍可通过policy objective改变std。无gSDE，无外部探索噪声替换。

## 11. Value Loss

`PPO.train`：`L_value=mean((returns−V_new(o))²)`，target `[256]`，valueflatten `[256]`。当前 `clip_range_vf=None`，**不使用value clipping**；也没有额外乘0.5的内层MSE，0.5来自总loss的vf_coef。

`L_total=L_policy+0*L_entropy+0.5*L_value`。

没有训练reward model、world model或BC imitation辅助项。

## 12. Optimizer Step

实际创建：`ActorCriticPolicy._build`（Adam，默认此类Adam epsilon=1e−5）；实际更新：`PPO.train`。

每minibatch：zero_grad → loss.backward → clip_grad_norm_(all_policy_parameters,0.5) → optimizer.step。Actor/critic/log_std共同接受对应梯度。`target_kl=None` 所以本配置未启用KL early stop；`approx_kl=mean(exp(log_ratio)−1−log_ratio)` 仍被记录。

本阶段模型加载会重建SB3对象/optimizer用于兼容加载，但仅调用predict；参数requires_grad=False、eval mode、所有parameter/checkpoint hashes前后一致。构造optimizer对象不等于执行optimizer step。

## 13. 与 BC 的根本差别

实际BC源码：`mujoco/rl/uav_bc_policy.py::BCActor/normalize/normalized_loss/BCPolicy.predict`，`uav_bc_training.py::train/select_epoch`。

BC输入Train-stat标准化 `[B,7]`，MLP7→128ReLU→128ReLU→4输出 `[B,4]` normalized expert commands。标签来自真实Scripted的对应时刻动作；`L_BC=mean(((a_pred−a_expert)/Train_action_std)²)`。部署反标准化后clip；B=512、Adam lr.001、最多100epoch、Validation action MSE选择checkpoint。

PPO则从自己与环境交互采集奖励，依赖value/GAE/概率比更新，无专家动作标签；网络不同、训练交互预算不同。本次统一的是**闭环评估接口/任务/状态**，不是训练预算或网络规模，因此不能从最终success推导样本效率优劣。

## 14. 本次审计发现的局限

没有发现足以阻止7D公平评估的action/observation接口错误。现有PI7D仅seed0、final100352步、历史里程碑固定目标评估、原始checkpoint SHA缺失且未保存全部训练target流，限制了跨seed鲁棒性/严格全训练隔离证明。

历史95.6%属于旧P环五seed；PI10D历史100%含额外信息。不能混合这三类报告。7D PI隐藏状态、有限训练预算、RewardV2任务对齐、Gaussian边界处理均是需受控验证的因素，**不是本轮已经证明的唯一缺陷**。不自动修改或重新训练。
