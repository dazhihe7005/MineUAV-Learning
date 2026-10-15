# AUTONOMOUS-RL-RESEARCH-V1：新增用户授权

这是项目负责人在AUDIT-01之后主动发出的独立授权，不回写MASTER_PROTOCOL_V1.0、L9R或OPS-V1/V2历史账本。恢复时优先读本文件及最新CODEX_NEXT_ACTION，再审计本地真实预算。

## 授权范围

累计最多2,000,000个训练environment timesteps、86400秒训练墙钟；单次最多1,000,064步。CPU、单worker，跨天／换实验ID不能刷新预算。允许经独立预注册的PPO Pilot、课程、对照及合理改进。当前实现只开放固定PPO对照入口，不执行免费optimizer测试。监督学习的预算折算尚未实现，不会擅自运行BC更新。

旧OPS工具显示NO和零预算是其历史维护权限，不是新授权的状态。新状态使用：

```bash
python3 -m research_regression.authorized_rl status
python3 -m research_regression.authorized_rl doctor
python3 -m research_regression.authorized_rl recover
```

新账本`.research_local/authorized_rl_v1/`包含唯一grant、哈希链事件及head。每个实验先预留全步数和全wall上限，完成也不自动退还wall；保守余额不是精确已花wall，另记录主管实测wall。失败／未知中断保留预留及唯一身份，不能重放、Warm Start或删除attempt重跑。账本损坏／孤立事件BLOCKED；本机可写用户并非恶意攻击者隔离边界。

## 停止与恢复

```bash
python3 -m research_regression.authorized_rl request-stop
```

停止请求永久锁存当前授权调度，主管停止当前进程组；此版本不提供静默清除停止。已停止／中断任务不假称完整结束，恢复为RESTART_REQUIRED。recover只审计、不启动训练；未决attempt先查实际进程和证据，不能绕过。电脑关机后Codex不工作，也没有开机服务；用户下次启动Codex后读本地状态继续安全研发。

## 实验纪律

先登记源码／协议／场景／Reward／网络／初始化／预算，独立审查GO才启动。固定协议运行时不改Reward、CNN、PPO、场景或动作／频率。每阶段MD、JSON、图、README及长期文档→相关测试→独立审查→Commit/Push→远端SHA核验。源码和报告发布仍由原OPS-V2显式白名单执行，没有关闭历史保护。

原L9永久FAIL，旧checkpoint不允许无损续训。初始研究从新随机网络开始，不使用SMOKE/PILOT权重。Train用于训练，Validation诊断，Test封存；不把高Reward、工程PASS或单seed成功外推为真实矿井能力。

资源最低MemAvailable1536MiB、磁盘2GiB；真实NaN/Inf、未知终止、源码或场景不符、证据损坏立即停。主管加独立Linux轻量守护进程约50ms轮询生命周期，非实时OS保证；只有一个PPO worker。禁止PX4、FAST-LIO2、实机／雷达通信、Gazebo／Isaac及系统依赖升级或swap修改。

## 给项目负责人的总结

你已批准研究预算，所以普通阶段与预算内经审查的新实验不再反复询问。此文档不会让失败训练复活，也不保证算法学会避障。首项是验证空场景能被非学习控制完成，再比较先学基础任务是否优于直接混合场景；每次真实结果和剩余额度都会记录并同步GitHub。
