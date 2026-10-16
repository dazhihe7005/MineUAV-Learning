# AUTORL-R16：中心化策略低初始噪声对照（工程PASS，科学总验收FAIL）

## 【1】阶段目标

检验仅降低初始探索std0.3→0.1，能否改善停稳经验与原距离导航学习。不是从R14权重继续训练，不重做旧R3未中心化std对照。

## 【2】技术背景

R14工程完整、科学总FAIL；R15固定输入显示已形成负速度反馈，但不代表物理制动已充分。短Train1867/2077近目标采样速度仍过大；低噪声假设需要真实PPO/物理验证，不能凭直觉认定。

## 【3】系统架构

原MuJoCo→24k射线/编码/异步输入→同中心化Actor和原Critic→Gaussian探索/动作clip→原PI/四电机物理。原监督/flock/预算/资源/源码/失败记录保护保留，只新增固定schema16入口与新初态赋值hook。

## 【4】代码实现

新centered_noise_profile.py冻结协议/来源/准入；centered_noise_training.py复用原初始化后仅原位修改log_std，再reset及保存本轮step0；新7纯和2真实tensor/保存重载/20tick一步测试。budget/runtime/__main__/worker增加显式16分支，复用原scheduler/算法/CheckpointManager，36全新Val不走旧初态复用。旧Actor核心文件、导航环境/Reward/PI/物理不改。

## 【5】算法原理

动作分布由N(mu_theta(x),diag(sigma²))定义；mu的共享MLP中心化结构不变。初始sigma减小但仍可优化，既改变实际探索经验也可能改变PPO梯度尺度，这是同一变量的影响。Gaussian指令标准差不是机体实际速度标准差；低std不自动保证安全或性能。

## 【6】冻结协议

ARL_R16_CENTERED_STD01_SEED0，65536actualstep/3600s保守预留，seed0/CPU1；新随机网络赋原手写均值先验及logstd.1，不加载旧ZIP。LR3e-5及128/64/2等PPO、Reward、课程、500/100/25/10Hz和24k射线同R14。0/32768/65536各6原+6短全新Val，共36，Test0，FinalCheckpoint选择。

REGbbf7dae8…/SOURCEf253400f…（112文件）/execution10e6b844…；新fullpolicy650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4，独立零更新校准仅log_std不同，同62输入mean/Value完全一致。五判据详细见design/prereg：短候选、own短保留、own原学习、相对R14原收益、短Train停稳经验比例机制；最后一项不能替代导航成功。原预算余1344640step/50400s、stop=false，实际预留后1279104/46800，不退款或重置。

## 【7】实际结果与测试

新7纯+2真实tensor/hook及4分析契约RED→GREEN，完整166OPS+55Tensor+885历史=1106选择PASS/15排除/0测试optimizer，非全仓。checks SHA f2264ede2f730ebc3cc8aae4b1b65b9df37315c6c7e55084a44428c2a67cc61c。独立C0/I0/新Minor0/继承Minor1精确GO后，新register/recover/doctor、prepare及即时门禁实际通过，唯一run主管exit0/ledger COMPLETE，无supervisor_failure。

实际65536step、512rollout、1024个PPO epoch、2048个Adam更新，1309123真实物理tick/2618.246s训练物理时间；部分终止step提前结束物理推进，不假定每step20tick。473条结束episode、474reset、自然32850步切课程，最后20步未结束episode另列。24个冻结Train任务及12基础几何访问可核，Test0。

| 域 | 结束episode | success | collision | timeout | physical/sensor |
|---|---:|---:|---:|---:|---:|
| 短Train | 251 | 223 | 18 | 10 | 0/0 |
| 原Train | 222 | 24 | 157 | 41 | 0/0 |

| checkpoint | 短Val 成功/碰撞/超时 | 原Val 成功/碰撞/超时 |
|---|---|---|
| 新初态0 | 6/0/0 | 1/3/2 |
| 32768 | 5/1/0 | 2/3/1 |
| final65536 | 5/1/0 | 0/3/3 |

36条全新Val，无旧初态复用；三个checkpoint参数/预测/optimizer实际重载核验，训练checkpoint有非空Adam。final是唯一正式选择，不挑32k较好结果。最终policy SHA25ff48c43dba75bb3cf19fb82ff8a5d9a916021a67a8512fc12111565eb5c259，final ZIPb3244a5622e23bc5d1216705b5f420acdc8c9217fa9428d6ccd5e7ab55c0468f；summary bf3814e86f0402ed02748483453a8279079aabca3e4df0c93264a82fcc71668c。23份raw local-only完整清单及五实际图在机器报告。

五冻结判据：短候选PASS；own短初态保留FAIL（5<6且碰撞1>0）；原目标严格学习FAIL（final0<初态1）；相对R14收益FAIL（0<1）；停稳经验机制PASS。短Train成功比例223/251=88.84%高于34/108=31.48%，近目标高速比例4205/5423=77.54%低于1867/2077=89.89%，但没有转化为最终独立导航收益。科学总验收FAIL，不改门槛或回选中间checkpoint。

全512记录点approx_kl均值0.000315262、max0.00321430，clip_fraction均值0.000404358；这是最后epoch minibatch诊断，不是累计策略KL。Value explained_variance均值0.06027、final0.02939，value_loss均值19.0046；记录低拟合证据，不据单点认定唯一根因。std final约0.0999022，确实保持可训练。

主管851.212s、wrapper863.251s，worker/child-tree HWM1738.480MiB；训练采样最低Mem6877.793MiB，含Val连续最低NOT AVAILABLE，无观察OOM。最后训练资源样本CPU793.459s/wall808.314s≈单核98.16%，不是整机CPU负载或包含Val的全程CPU。环境step p50/p95/p99/max=1.172/12.023/12.334/15.338ms；扫描6.347/6.779/7.011/9.699ms；编码4.077/4.260/4.525/6.460ms；PPO update18.597/19.171/20.504/26.553ms。

112运行源码保持，246旧不可变授权文件及R13/R15七raw不变，七历史checkpoint/原物理模型身份保持。真实累计预留720896step/39600s，余1279104step/46800s、stop=false，不退款/重置；22-event链与唯一RESERVE→COMPLETE可审计。实际结果独立复审、Commit及OPS远端完成以真实收据为准，未全部完成前不称PHASE_COMPLETE。

## 【8】问题分析

CONFIRMED：低噪声产生更多短目标停稳经验，但最终原目标Val0/6，没有达到学习提升。更有诊断价值的是：原Val两个empty在初态均超时，32k时均成功，进入原距离混合课程后final又均超时；front_box/detour/channel三个障碍场景三次均collision；side_pair初态成功后两次超时。final三个超时的末目标误差约0.101451/0.068340/0.098791m，不能仅以位置接近或终点距离判成功，还需速度和连续停稳条件。

SUSPECTED：第二阶段难混合任务负反馈/训练分布可能损害已获得的空场景能力，需要单变量检查保留与课程分布；Critic低解释方差也需持续记录。UNVERIFIED：是否灾难性遗忘的唯一机制、提高空场景重放比例能否帮助、LiDAR感知是否被有效利用及泛化。不用“短Train高成功”掩盖最终FAIL，也不把中间2/6当最终成绩。

## 【9】局限性

单seed/每域6小Val、手写初态及历史对照，不证明真实矿井/多seed能力。既有可选cv2/NumPy ABI与历史warning保留，CPU路径实际测试有效、不升级依赖。R7畸形active_task诊断Minor继续披露。恢复RESTART_REQUIRED，失败不重试本ID，不从checkpoint续训。

## 【10】下一阶段

完成本阶段实际结果复审、Commit及正常OPS sync远端后，新R17优先检验第二阶段原距离训练中empty任务的显式重放比例：只改课程采样分布，保留R16新初态/std.1/网络/Reward/LR/PPO/物理/目标清单，前半程仍原短课程。必须先冻结具体权重与评价标准，复用或核验已有非训练可完成性证据、测试、独立GO、余额/stop/资源才新attempt；不重复R16，不旧ZIP续训，不看到结果改Reward。普通科学FAIL不结束长期研究。

## 【11】给项目负责人的总结

小噪声确实让短任务训练更容易成功，但最终原距离验证反而0/6，所以不能说策略会导航了。32k时两个空场景曾成功，后半程混合训练后又失去，下一步值得只改空场景重放比例，检查能否保留基础能力，而不是盲加训练步数或改一堆参数。失败已如实保留，先审查和同步GitHub再自主继续。预算余1279104步/46800秒，PX4、实机及Test仍不运行。
