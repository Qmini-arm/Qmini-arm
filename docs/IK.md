# `arm_ik` 运动学与逆解库

`arm_ik`从URDF读取模型，做正运动学、逆运动学、可达空间分析、轨迹插值和自碰撞检测，
并把关节角换算成CDS55xx舵机的tick值。

它不碰串口。硬件读写由`cds_arm`负责，两者的分工是：

```
arm_ik 算关节角  ->  ServoMap.to_ticks 换成 tick  ->  cds_arm.move() 下发
```

版本`0.3.0`，核心只依赖numpy和scipy。

## 安装

```bash
uv sync                  # 核心功能
uv sync --all-extras     # 含viser可视化
uv run pytest            # 46个测试
```

## 最小调用流程

```python
import numpy as np
from arm_ik import RobotModel
from arm_ik.servo import ServoMap

robot = RobotModel.from_urdf("description/arm.urdf")
servo = ServoMap.from_yaml("arm_ik/config/servo_calibration.yaml", robot.joint_names)

# 1. 把硬件安全限位折进模型。少了这一步，IK 会给出舵机执行不了的角度。
robot.tighten_limits(*servo.effective_limits(robot))

# 2. 逆解：给末端位置，得到6个关节角（弧度）
result = robot.ik(position=[0.18, 0.0, 0.20])
if not result.status.is_usable:
    raise RuntimeError(f"{result.status}, 差 {result.position_error * 1000:.1f} mm")
print(np.degrees(result.q))   # [-3.82, -1.10, 34.57, -20.66, -25.78, 3.08]

# 3. 换成舵机tick
ticks = servo.to_ticks(result.q)
print(ticks)                  # {1: 799, 2: 118, 3: 262, 4: 411, 5: 271, 6: 99}

# 4. 下发（由cds_arm负责，用法见 API.md）
# from cds_arm import connect
# with connect() as arm:
#     arm.takeover_current(speed=160)
#     arm.move(ticks, speed=160)
```

第1步和`is_usable`检查都不能省。理由见下面两节。

## 这台臂的关节与限位

`hand_palm`是默认末端。6个可动关节按从基座到末端的顺序：

| # | 关节名 | 舵机ID | URDF限位 (°) | 有效限位 (°) | 零位tick | 安全tick |
|---|--------|--------|--------------|--------------|----------|----------|
| 1 | `kd_base_side_to_kd_2` | 1 | −55.72 … +49.85 | −50.44 … +49.85 | 812 | 640…1000 |
| 2 | `kd_2_to_u3b_base` | 2 | −26.98 … +66.86 | −26.98 … +66.86 | 122 | 30…350 |
| 3 | `kd_3_to_u3b_lower` | 3 | −11.73 … +134.90 | −11.73 … +133.72 | 144 | 100…600 |
| 4 | `u3b_middle_to_kd_pair_front` | 4 | −43.99 … +35.19 | −43.99 … +34.90 | 481 | 330…600 |
| 5 | `kd_pair_to_u3b_upper` | 5 | −34.31 … +30.21 | −31.96 … +30.21 | 359 | 250…470 |
| 6 | `kd_4_to_palm` | 6 | −26.69 … +31.97 | −25.81 … +31.97 | 88 | 0…200 |

有效限位是URDF限位与舵机安全窗口的交集，两者不一致时取更紧的一侧，
所以较宽那一侧的行程用不到。

**`RobotModel`默认只知道URDF限位，不知道舵机窗口。** 它从URDF构造，
而安全窗口在`ServoMap`里，两者要靠`tighten_limits()`显式接起来：

```python
robot.tighten_limits(*servo.effective_limits(robot))
```

少了这一步，IK会给出超出舵机行程的解。实测400个可用解里有58个越界（14%），
这些解会被`to_ticks`静默夹取，**实机落点和IK说的不一样**。加上这一步后同样测试零越界。

`ServoMap.validate_against()`会列出超过2°的不一致，当前是关节1（5.28°）和关节5（2.35°）。

关节1那5.28°有确切来源：它恰好等于18 tick（`18 × 0.2933° = 5.279°`），
正是下面「已知限制」里`cds_arm.CENTER`与实测记录对舵机1零位的分歧值。
把零位从812换成830，窗口下限就变成`(640−830) × 0.2933 = −55.72°`，
与URDF限位精确相同——**说明这个差值是零位取错造成的假象，不是机械约束**，
URDF限位本来就是按830推出来的。

## 逆解的语义

这台臂行程窄，可达域是一层壳（离基座39…351 mm）而不是实心球，所以IK的返回值需要按状态区分处理，
不能当成布尔值。

### `IKStatus`

| 状态 | `is_usable` | 含义 |
|------|-------------|------|
| `CONVERGED` | `True` | 位置和姿态都达到容差 |
| `POSITION_ONLY` | `True` | 位置达到，姿态没达到 |
| `LIMIT_BLOCKED` | `False` | 被关节限位挡住 |
| `OUT_OF_REACH` | `False` | 目标在可达壳之外 |
| `MAX_ITER` | `False` | 迭代到上限仍未收敛 |
| `SINGULAR` | `False` | 雅可比接近奇异 |

**不可达时不抛异常，而是返回限位内最接近的位姿。** 因为对这台臂来说"最近可行解"通常比"失败"有用。
判断是否可用要看`result.status.is_usable`，或者直接检查`result.position_error`。

### 位置可达不等于该位置任意姿态可达

这是最容易踩的一点：

```python
robot.ik(position=[0.18, 0, 0.20])
# CONVERGED, 位置误差 0.000 mm

from arm_ik.model.transforms import rpy_to_matrix
robot.ik(position=[0.18, 0, 0.20], orientation=rpy_to_matrix([0, 0, 0]))
# LIMIT_BLOCKED, 位置误差 120.0 mm, 姿态误差 17.56°
```

同一个位置，只给位置时精确收敛，加上`rpy(0,0,0)`就差了120 mm。

原因是关节行程窄加上雅可比长期病态（条件数中位数155、90分位967），
6个关节要同时满足3个位置约束和3个姿态约束，多数姿态凑不出来。
**这是机械结构的限制，不是求解器的问题。**

**建议先用只给位置的解确认可达，再加姿态约束看是否还成立。** 库里没有"只约束某一个轴"
（比如只要求接近方向朝下、不管绕该轴的转角）的接口，`orientation`是全约束或不约束。
需要这种半约束时可以自己在`_solve_from`里做，或者调低`orientation_weight`让姿态变成软约束。

### 姿态的表示

`orientation`接受3x3旋转矩阵、四元数`(w,x,y,z)`或rpy三元组。

`rpy_to_matrix(rpy)`用**弧度**，约定与URDF/ROS一致：固定轴xyz顺序，即`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`。

要表达"手掌朝某个方向"需要知道`hand_palm`的哪个轴指向掌外。中位姿下它的三个轴在世界系的朝向：

```python
T = robot.fk(robot.mid_range)
T[:3, 0]   # local x -> [ 0.030,  0.953, -0.302]
T[:3, 1]   # local y -> [-0.361,  0.292,  0.886]
T[:3, 2]   # local z -> [ 0.932,  0.082,  0.353]
```

即中位姿下local z基本指向世界+x（臂伸出的方向），local x基本指向世界+y。
自己的目标姿态用哪个轴，照这个方式实测确认，不要凭猜。

## `RobotModel`

### `RobotModel.from_urdf()`

```python
RobotModel.from_urdf(path: str | Path, tip_link: str = "hand_palm") -> RobotModel
```

解析URDF并建立运动学链。自由度由URDF决定而非硬编码，
把某个`fixed`关节改成`revolute`后模型自动变成7自由度，不需要改代码。

属性：`dof`、`joint_names`、`lower`、`upper`、`velocity`、`mid_range`、`reach_bounds`。

`reach_bounds`返回`(最小半径, 最大半径)`，用于把"目标在可达域外"和"求解器卡住"区分开。
它是刻意放宽的估计（当前51.7…368.4 mm），比后面`analyze_workspace`实测的采样半径
（38.9…350.9 mm）更宽——因为采样覆盖不到真正的极值，宁可放宽也不要把可达的点误判为不可达。
要精确判断可达性用`sample_workspace`。

### `fk()`

```python
fk(q: ArrayLike) -> FloatArray      # 4x4齐次变换
```

末端在基座坐标系下的位姿。已与`yourdfpy`在100组随机构型上交叉验证，位置差1e-16 m量级。

```python
robot.fk(np.zeros(6))[:3, 3]      # [0.0323, 0.0152, 0.3451] 零位
robot.fk(robot.mid_range)[:3, 3]  # [0.2140, 0.0451, 0.1776] 中位姿
```

`link_poses(q)`返回链上每个link的位姿，`chain_state(q)`返回中间结果。

### `ik()`

```python
ik(
    position: ArrayLike | None = None,
    orientation: ArrayLike | None = None,
    *,
    target: ArrayLike | None = None,
    seed: ArrayLike | None = None,
    solver: str = "dls",
    config: SolverConfig | None = None,
) -> IKResult
```

`position`+`orientation`和`target`（4x4矩阵）二选一。`orientation`可以是3x3矩阵、
四元数或rpy三元组。省略`orientation`就只解位置。

`seed`是初始猜测，省略时用`mid_range`——**不要用零位当种子**，
零位是接近奇异的完全伸展姿态，且正好卡在某个关节的行程边界上。

连续求解（比如跟踪轨迹）时把上一次的解传给`seed`，可以让关节解留在同一分支上不跳变。

### servo6 独立控制

这台机械臂的第6轴只改变手掌绕自身轴的姿态，不改变手掌原点位置。
`ik_position()`因此只优化前5个关节，并把servo6固定在调用者给出的角度（或seed中的角度）。
返回值仍是完整6维向量，可以直接交给`fk()`和`ServoMap.to_ticks()`：

```python
from arm_ik.servo import Servo6Controller, ServoMap

servo = ServoMap.from_yaml("arm_ik/config/servo_calibration.yaml", robot.joint_names)
servo6 = Servo6Controller(robot, servo, angle=np.radians(10.0))
result = robot.ik_position(
    position=[0.18, 0.0, 0.20],
    servo6=servo6.angle,
    seed=robot.mid_range[:5],
)
ticks = servo.to_ticks(result.q)
```

`robot.compose_arm_q(q_arm, servo6=angle)`可在不求IK时合成完整关节向量；
`Servo6Controller.set_degrees()`、`to_tick()`和`compose()`分别用于独立设角度、换tick和合并。
Viser的IK界面也采用相同语义：位置由前5轴IK求解，servo6使用独立滑块，不再尝试完整RPY逆解。

### `jacobian()` 与可操作度

```python
jacobian(q) -> FloatArray                    # 6xN 几何雅可比
numeric_jacobian(q, eps=1e-7) -> FloatArray  # 中心差分，用于交叉验证
manipulability(q) -> float                   # sqrt(det(J J^T))
condition_number(q, length_scale=0.35) -> float
```

解析雅可比与数值雅可比的最大差在2e-9量级（50组随机构型实测）。

这台臂条件数中位数155、90分位967，属于长期病态。这解释了为什么IK需要多次重启，
也解释了为什么姿态约束经常满足不了。

### 限位相关

```python
clamp(q) -> FloatArray                       # 夹到限位内
within_limits(q, tol=1e-9) -> bool           # 支持单个构型或 (N, dof) 批量
random_configuration(rng=None) -> FloatArray
tighten_limits(lower, upper) -> None         # 只能收紧，不能放宽
```

`tighten_limits`拒绝放宽，因为URDF限位还编码了舵机控制器不知道的机械干涉。
把硬件安全限位折进模型用它。

## `IKResult`

```python
status: IKStatus
q: FloatArray                     # 关节角，弧度
position_error: float             # 米
orientation_error: float          # 弧度
iterations: int
restarts_used: int
residual_history: tuple[float, ...]
```

另有一个property`success`，等价于`status.is_usable`。**它会把`POSITION_ONLY`算作成功**，
也就是位置到了、姿态没到。只要姿态重要，就检查`status is IKStatus.CONVERGED`，
而不是`success`。

`residual_history`可以看收敛过程，调参时有用。

## `SolverConfig`

```python
position_weight = 1.0          orientation_weight = 0.35
position_tolerance = 1e-06     orientation_tolerance = 1e-05
max_iterations = 200           restarts = 24
damping_min = 0.0001           damping_init = 0.01
damping_max = 100.0            damping_decrease = 0.4
damping_increase = 2.5         max_step_norm = 0.15
seed = 0                       # 随机重启的RNG种子，不是初始关节角
```

**注意有两个`seed`，含义不同**：`SolverConfig.seed`是随机重启用的RNG种子（整数），
`ik(seed=...)`是初始关节角猜测（长度6的数组）。传错不会报错，只是静默做了另一件事。

`restarts=24`是实测定下来的：12次时300组随机往返里有1组解不出，24次全过，
而且因为绝大多数情况第一个种子就够（重启次数中位数为0），耗时没有变化。

`orientation_weight=0.35`把姿态误差（弧度）压到和位置误差（米）可比的量级。

```python
from arm_ik import SolverConfig
robot.ik(position=[0.2, 0, 0.15], config=SolverConfig(restarts=40))
```

## 两个求解器

| 名字 | 300组随机往返 | 位置误差中位数 | 单次耗时 |
|------|---------------|----------------|----------|
| `dls`（默认） | 300/300 | 8.3e-08 m | 7.1 ms |
| `least_squares` | 300/300 | 1.6e-13 m | 4.1 ms |

`dls`是自带的阻尼最小二乘，不依赖scipy。`least_squares`包装`scipy.optimize`，
精度和速度都更好，但对初值更敏感。

### 注册自定义求解器

要实现的是`_solve_from`，不是`solve`。基类的`solve()`负责多次重启、状态判定和误差统计，
子类只需要从一个给定种子出发解一次：

```python
from arm_ik import register_solver
from arm_ik.solvers.base import BaseIKSolver

@register_solver("my_solver")
class MySolver(BaseIKSolver):
    name = "my_solver"

    def _solve_from(self, target, seed):
        # 返回 (关节角, 迭代次数, 残差历史)
        return q, iterations, residuals

robot.ik(position=[0.2, 0, 0.15], solver="my_solver")
```

## `arm_ik.servo` 舵机映射

CDS55xx用0…1023的整数tick表示300机械度，一个tick是`300/1023 ≈ 0.2933°`。

### `ServoMap`

```python
ServoMap.from_yaml(path, joint_names) -> ServoMap
ServoMap.derive(robot, center, safe_limits, servo_ids, directions=None) -> ServoMap

to_ticks(q) -> dict[int, int]        # 关节角 -> tick，按舵机ID
to_joints(ticks) -> FloatArray       # tick -> 关节角
to_degrees(q) -> dict[int, float]    # 关节角 -> 舵机角度，用于日志
effective_limits(robot) -> tuple[FloatArray, FloatArray]
validate_against(robot, tolerance_deg=2.0) -> list[str]
```

`to_ticks`会把结果夹到每个舵机的安全窗口内，所以输出总是可下发的。
发生夹取时会打`SEVERITY=HIGH`的warning，因为那说明IK给出的角度超出了硬件允许范围，
实际落点和IK说的不一样。

标定文件在`arm_ik/config/servo_calibration.yaml`，随包安装。
从仓库外定位它（`config`不是package，所以要从`arm_ik`往下走）：

```python
from importlib.resources import files
path = files("arm_ik") / "config" / "servo_calibration.yaml"
```

### 重新生成标定

换了实机、调过零位、或`replay`显示浏览器姿态与真臂不符时，重新生成：

```python
from cds_arm.core import CENTER, SAFE_LIMITS
from arm_ik.servo import ServoMap

servo = ServoMap.derive(
    robot,
    center=CENTER,                      # {servo_id: tick}，URDF零位对应的tick
    safe_limits=SAFE_LIMITS,            # {servo_id: (lower, upper)}
    servo_ids=[1, 2, 3, 4, 5, 6],       # 顺序对应 robot.joint_names
    directions=None,                    # None 表示由URDF限位与tick窗口拟合方向
)
servo.to_yaml("arm_ik/config/servo_calibration.yaml")
servo.validate_against(robot, tolerance_deg=2.0)   # 确认没有大的不一致
```

零位要用真机读，别照抄常量：

```python
from cds_arm import connect
with connect("/dev/ttyUSB0") as arm:
    print(arm.read_positions())   # 摆到URDF零位姿态后读，这就是 center
```

### `JointCalibration`

```python
joint_name: str        servo_id: int
center_tick: int       # URDF零位对应的tick
direction: int         # +1 表示tick增大则关节角增大
tick_lower: int        tick_upper: int
fit_residual_deg: float
```

**`direction`是实测的物理属性，默认全部为`+1`**（`MEASURED_DIRECTION`，
真机实测记录在`recent.txt`）。早期版本靠"哪个符号能更好复现URDF限位"来推断方向，
在舵机窗口被单独放宽后这个打分会选错，把关节2推成`-1`，实机会朝反方向转。
现在方向由`directions`参数传入，`fit_residual_deg`退化为纯诊断量。

### `fk_from_servo()`

```python
fk_from_servo(robot, servo_map, backend) -> tuple[FloatArray, FloatArray]
```

读实际舵机位置，返回`(关节角, 末端位姿)`。`backend`只要实现`read_positions()`即可，
`cds_arm.CDSArm`满足这个协议。

## `arm_ik.workspace` 可达空间

```python
from arm_ik.workspace import sample_workspace, analyze_workspace

ws = sample_workspace(robot, count=20000, seed=0)
ws.positions       # (N, 3)
ws.radii           # 离基座距离
ws.is_reachable([0.2, 0, 0.15], tol=1e-2) -> bool

rep = analyze_workspace(robot, ws)
```

`ReachabilityReport`字段：`bounds_lower`、`bounds_upper`、`min_radius`、`max_radius`、
`mean_radius`、`condition_number_median`、`condition_number_p90`、
`manipulability_median`、`orientation_cone_half_angle_deg`、`samples`。

6000点采样的实测结果：

```
半径      38.9 … 350.9 mm
x         −0.238 … +0.258 m
y         −0.195 … +0.267 m
z         −0.070 … +0.350 m
条件数    中位数 155，90分位 967
```

`orientation_cone_half_angle_deg`这个字段要当心，**它不是"腕部能覆盖的姿态锥"**。
实现量的是末端local z与参考轴（默认世界+z）的夹角，且夹角超过90°的样本被记作0
（[analysis.py:76-79](../arm_ik/workspace/analysis.py#L76-L79)）。这台臂34%的样本超过90°，
所以报出的83.6°是截断后分布的95分位，真实夹角跨1.3°…178.7°、中位71.4°。
把它当作"姿态覆盖受限"的证据是错的——判断姿态能不能解，直接拿目标位姿试`ik()`。

注意`is_reachable`判断的是**位置**落在采样壳内，不保证该位置上的任意姿态可达。

## `arm_ik.trajectory` 轨迹

### `interpolate_joint()`

```python
interpolate_joint(q0, q1, points=100) -> FloatArray   # (points, dof)
```

五次曲线，两端速度和加速度都为零。两个合法构型之间插值不会越界。

### `interpolate_cartesian()`

```python
interpolate_cartesian(robot, pose0, pose1, points=100) -> tuple[FloatArray, FloatArray]
```

位置走直线，姿态走SO(3)测地线（而不是在欧拉角上线性插值，那会让腕部转速不均匀）。
每个路点的IK都用上一个点的解作种子，所以关节解留在同一分支上。

**中间路点不可达时抛`ValueError`，不静默夹取。** 窄行程臂的直线笛卡尔路径经常会
穿过臂够不到的地方——比如零位姿态在中段位置就保持不住。遇到这种情况改走关节空间路径，
或者把笛卡尔路径拆成几段可达的。

### `time_parameterize()`

```python
time_parameterize(q_path, velocity_limits, dt=0.02, accel=None) -> TimedTrajectory
```

`TimedTrajectory`有`times`、`q`、`qd`、`duration`和`sample_at(t)`。

`accel`（rad/s²）**约束的是路点处的速度跳变除以相邻两段的平均时长**，
不是采样网格上`diff(qd)/dt`的加速度。每段以恒定速度走过，所以网格上每个路点必然有阶跃，
那个数会远高于设定值。它买到的是路点之间速度变化平缓，不是时间最优的梯形曲线；
尖角会被减速通过而不是圆滑处理。要更平滑就先把`q_path`加密，或者让舵机环在采样点之间插值。

## `arm_ik.collision` 自碰撞

```python
from arm_ik.collision import CollisionChecker

checker = CollisionChecker(robot, extra_ignored=(), margin=0.0)
checker.is_free(q) -> bool
checker.check(q) -> list[CollisionPair]     # link_a, link_b, depth
```

基于URDF里的box碰撞体做OBB相交，单次约0.7 ms。相邻link默认忽略。

## `arm_ik.viz` 可视化

需要`viz` extra。viser在浏览器里渲染，服务端是本地Python进程，
所以可以跑在无头机器上从笔记本看。

```python
from arm_ik.viz import launch_viewer, launch_ik_app, replay
from cds_arm import connect

launch_viewer(robot)                      # 每个关节一个滑条
launch_ik_app(robot)                      # 拖拽/滑条设定目标，驱动IK
replay(robot, servo_backend, period=0.05) # 读真实舵机驱动数字孪生

# 传入可写的 CDSArm 后，viewer/IK 面板会出现“驱动实际机械臂”开关。
# 开关启用时先无跳变接管当前反馈，再发送后续滑条/IK目标。
with connect("/dev/ttyUSB0") as arm:
    launch_viewer(robot, servo_backend=arm, speed=160)
    # 或：launch_ik_app(robot, servo_backend=arm, speed=160)
```

三个模式都有「显示可达空间」开关，勾选后画出8000点的可达壳，按离基座距离着色。
点云在首次勾选时才采样，不拖慢启动。

`launch_ik_app`的目标位置有两种控制方式并且双向同步：拖拽手柄，
或者3个位置滑条（米）。servo6另有独立角度滑条；位置由前5轴IK求解，
不会把完整末端RPY作为逆解约束。

`replay`不涉及IK，是排查标定错误最直接的工具：如果实机姿态和浏览器里的对不上，
基本就是`center_tick`或`direction`错了。

viewer 和 IK 启动时默认像`replay`一样自动连接唯一串口；也可以通过`--device`
（`--serial`同义）指定串口：

```bash
uv run arm-ik viz --mode viewer --device /dev/ttyUSB0 --speed 160
uv run arm-ik viz --mode ik --device /dev/ttyUSB0 --speed 120
```

浏览器中勾选“驱动实际机械臂”才会开始发送；关闭后只停止发送新目标，不会自动关闭舵机扭矩。
需要纯仿真且不打开串口时显式加`--sim`。

## 命令行

```bash
uv run arm-ik --urdf description/arm.urdf fk 0 0 0 0 0 0
uv run arm-ik --urdf description/arm.urdf ik --pos 0.18 0 0.2 --servo
uv run arm-ik workspace --count 20000
uv run arm-ik viz --mode ik --port 8090 --sim
uv run arm-ik viz --mode replay --device /dev/ttyUSB0
```

`--urdf`是全局选项，必须放在子命令**之前**。`ik`的`--rpy`可选，
不给就只解位置；`--servo`额外打印舵机tick。

## 已知限制

**指定完整6自由度位姿经常解不出来。** 关节行程窄（合计仅504°，是±π六轴臂关节空间的0.01%），
雅可比长期病态（条件数中位数155、90分位967）。这是机械结构决定的，不是求解器的问题。
可达域是一层壳（离基座38.9…350.9 mm）而非实心球。

**零位不适合当种子。** 零位是接近奇异的完全伸展姿态。默认种子是`mid_range`。

**URDF与舵机窗口尚未完全对齐。** 除关节2以外，其余5个关节的URDF限位和舵机安全窗口
仍有0.3…5.28°的差值，那部分行程用不到。`validate_against()`会列出超过2°的项。

**零位标定存在未解分歧。** `cds_arm.CENTER`与`recent.txt`的真机实测在
舵机1差18 tick、舵机5差8 tick。当前标定文件用的是`cds_arm`那份。
这个差值会直接变成末端偏差，建议用真机`read_positions()`读一次实际零位定下来。
