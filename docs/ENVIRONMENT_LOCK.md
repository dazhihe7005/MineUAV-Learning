# MineUAV-Learning Environment Lock

机器可读清单：`research_ops/environment_lock.json`。管理CLI仅使用Python标准库，不导入SB3/PPO。原历史环境和全部历史实验身份保留。

| 项目 | 固定值 |
|---|---|
| 管理入口 | `/usr/bin/python3`，仓库内 `./researchctl` |
| 实验解释器 | `/usr/bin/python3.10`，3.10.12，SHA256 `a2f33a6e006989270f4340528eb61f8f97366e00a5d1b602ac8672ea44fc56ae` |
| 历史包目录 | `/home/keyousuke/MineUAV-Learning/.venv/lib/python3.10/site-packages` |
| MuJoCo | 3.14.0，必须实际存在MjModel API和包路径，不能是同名源码namespace |
| SB3 | 2.9.0 |
| PyTorch | 2.14.0+cpu |
| Gymnasium / NumPy | 1.3.0 / 2.2.6 |
| MuJoCo-LiDAR | `e6b879df4b96daaed30ef10fe02650d29048fa9b`，CPU后端 |
| mid360.npy SHA256 | `9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3` |
| 持久化上游 | `.research_local/dependencies/MuJoCo-LiDAR/`，Git忽略，非/tmp |

早上运行 `./researchctl doctor`。若仅持久化上游缺失，使用：

```bash
./researchctl doctor --restore-dependencies
```

它只获取固定Git源码并checkout指定提交，核对pattern和tracked工作树，不pip安装、不取PyPI latest、不启用GPU。已有不匹配目录不会覆盖。失败fetch目录保留诊断；历史/tmp副本保留不动。可用本机已验证旧副本作只读源：`--source /tmp/mineuav-mujoco-lidar-audit`；这只影响获取来源，最终commit和pattern要求不变。缺解释器、包/API或版本错误时BLOCKED；不自动修理系统Python或原venv。

后续获授权实验所需环境由 `research_ops.environment.runtime_environment` 构造：PYTHONPATH包含项目mujoco/、mujoco/rl/、持久上游src/及历史venv包目录；`MINEUAV_LIDAR_UPSTREAM`指向持久目录。OMP/MKL/OPENBLAS线程数均为1。不继承任意旧PYTHONPATH/PYTHONHOME，不注册开机服务。

历史 `.venv/bin/python`解释器symlink存在，但此前运行使用系统Python配合已验证的venv包路径。doctor记录真实sys.executable、module.__file__、metadata版本及MjModel API，而不是仅凭文件名或import成功宣称环境有效。

现有系统OpenCV基于NumPy1 ABI，而NumPy2.2.6环境存在 `_ARRAY_API` 导入警告。SB3可选Atari代码捕获ImportError；现有导航链路不调用Atari/OpenCV。OPS的probe直接检查MuJoCo/NumPy/Torch和SB3 metadata，不掩盖警告，也不触发任何optimizer。相关旧回归仍会打印已知ABI诊断。没有全局升级、swap修改或依赖安装。

环境清单及运行源码在首次init时冻结进本地protocol。改源码/配置后doctor将拒绝旧身份，必须在新管理协议版本中保留旧attempt并重新登记；V1不提供静默rebaseline。不能通过篡改protocol和摘要“修好”身份检查。

## 旧测试的路径兼容限制

OPS管理及doctor使用持久化依赖，已经实际通过。旧LiDAR导航/终止修复测试中仍有多处固定 `/tmp/mineuav-mujoco-lidar-audit` 路径，而其他测试读取环境变量。将整组旧回归仅改成持久路径，会触发scanner对“传入源目录”和“实际import来源”不一致的正确拒绝。

本轮记录了这次兼容性失败（812通过、64失败、9 setup错误、15排除），没有覆盖日志或修改冻结旧测试。旧回归随后使用仍保留的历史/tmp副本及一致的PYTHONPATH重跑；其实际结果见阶段JSON。两个副本均为相同固定Git提交和patternSHA。

若重启后/tmp副本消失，OPS自身仍可doctor/start/recover；**旧完整LiDAR回归的无/tmp兼容性UNSUPPORTED**，需单独的Level1测试可移植性修复，将源路径统一为受审计的配置，再保留旧行为回归。本轮不通过隐藏monkeypatch、替换已有目录或放宽source身份要求让旧测试通过。这个限制不会授予任何训练权限。
