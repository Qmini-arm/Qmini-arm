# Qmini-arm

Qmini机械臂的URDF描述、网格资源和CDS55xx六轴基础运动Python库。

## 实验性 Demo

- [Qmini Avatar](demos/qmini_avatar/README.md)：使用普通电脑摄像头和 MediaPipe，将人手的
  平移、腕部旋转与五指开合实时映射到 Qmini 机械臂和 uHand。它拥有独立的安装环境，
  OpenCV、MediaPipe 等视觉依赖不会进入底层动作库。

## Python库

`cds_arm`只提供基础运动API，不包含预设演示姿态、动作序列、交互标定或演示CLI。

### 安装

```bash
python3 -m pip install .
```

Mac串口通常为`/dev/cu.usbserial-*`，香橙派/Linux通常为`/dev/ttyUSB0`或
`/dev/ttyACM0`。

### 运动到目标位置

目标必须完整包含ID 1～6，并位于各轴安全范围内。

```python
from cds_arm import connect

target = {
    1: 812,
    2: 122,
    3: 144,
    4: 481,
    5: 359,
    6: 88,
}

with connect("/dev/ttyUSB0", baud=115200) as arm:
    # 明确接管当前姿态；必要时会同步开启六轴扭矩。
    arm.takeover_current(speed=160)

    # 阻塞运动，到位或失败后返回。
    feedback = arm.move(
        target,
        speed=160,
        arrival_tolerance=5,
        trim_step=2,
        trim_max=48,
    )
    print(feedback)
```

主要接口：

- `connect()`：打开并自动关闭串口连接；本身不发送目标、不开启扭矩。
- `read_positions()`：读取六轴当前位置。
- `takeover_current()`：检查位置模式并无跳变接管当前姿态。
- `send_goals()`：同步发送六轴目标但不等待到位。
- `move()`：带反馈、安全范围和近目标补偿的阻塞运动。
- `diagnostics()`：读取负载、电压、温度、P增益等状态。
- `read_registers()` / `read_u16()`：底层只读寄存器接口。

完整方法签名和安全语义见[接口文档](docs/API.md)。

> 关闭串口不会自动关闭舵机扭矩。运行前应托住机械臂，并保留硬件断电手段。

## 机械模型

URDF和STL网格位于`description/`。

## 运动学与逆解（arm_ik）

`arm_ik`是一个纯 Python 的运动学/逆解库，核心只依赖 `numpy` + `scipy`。
它从 `description/arm.urdf` 读取真实关节限位，给出 6 个舵机应转到的角度。

```bash
uv sync --all-extras                       # 建环境并安装
uv run pytest                              # 跑测试
```

### 基本用法

```python
from arm_ik import RobotModel
import numpy as np

from arm_ik.servo import ServoMap

robot = RobotModel.from_urdf("description/arm.urdf")
servo = ServoMap.from_yaml("arm_ik/config/servo_calibration.yaml", robot.joint_names)

# 把舵机安全限位折进模型。少了这步，IK 会给出舵机执行不了的角度，
# 下发时被静默夹取，实机落点和 IK 说的不一样。
robot.tighten_limits(*servo.effective_limits(robot))

# 给定期望末端位姿（位置 + 可选姿态），得到 6 个关节角（弧度）。
result = robot.ik(position=[0.18, 0.0, 0.20])
if not result.status.is_usable:
    raise RuntimeError(f"{result.status}, 差 {result.position_error * 1000:.1f} mm")

ticks = servo.to_ticks(result.q)   # 交给 cds_arm 下发
print(ticks)   # {1: 799, 2: 118, 3: 262, 4: 411, 5: 271, 6: 99}
```

如果只需要位置、希望单独控制手掌旋转轴，使用`ik_position()`固定servo6：

```python
from arm_ik.servo import Servo6Controller

servo6 = Servo6Controller(robot, servo, angle=np.radians(10.0))
result = robot.ik_position([0.18, 0.0, 0.20], servo6=servo6.angle)
ticks = servo.to_ticks(result.q)
```

该接口只优化前5轴，返回值仍为完整6轴向量；Viser IK界面中的servo6滑块也是独立控制，
不会再把完整末端RPY送入逆解。

`result.status` 是 `IKStatus` 枚举，而非布尔：

| 状态                        | 含义                                   |
| --------------------------- | -------------------------------------- |
| `CONVERGED`               | 位置与姿态都达到容差                   |
| `POSITION_ONLY`           | 位置达到，姿态未达到                   |
| `OUT_OF_REACH`            | 目标位置在可达壳之外，返回最近可达位姿 |
| `LIMIT_BLOCKED`           | 被关节限位挡住                         |
| `MAX_ITER` / `SINGULAR` | 达到迭代上限 / 接近奇异                |

关键点：**不可达目标返回限位内最接近的可达位姿，而非报错**。这台臂的关节行程
很窄，可达域是一层壳（离基座 40～351 mm）而非实心球，所以"给出最近可行解"
比"失败"更有用。

需要特别注意位置与姿态的区别：**位置可达不代表该位置上任意姿态都可达**。
例如 `[0.18, 0, 0.20]` 只给位置时能精确收敛，但同时要求 `rpy(0,0,0)` 就解不出来
（会返回 `LIMIT_BLOCKED`，位置偏差约 120 mm）。所以先用只给位置的解确认可达性，
再逐步加姿态约束。

### 命令行

```bash
uv run arm-ik fk 0 0 0 0 0 0                # 正运动学
uv run arm-ik ik --pos 0.18 0 0.2 --servo   # 逆解并显示舵机 tick
uv run arm-ik workspace --count 20000       # 采样并分析可达空间
uv run arm-ik viz --mode viewer --sim       # viser 浏览器纯仿真
uv run arm-ik viz --mode ik --sim           # 拖拽目标姿态驱动 IK（纯仿真）
uv run arm-ik viz --mode viewer --device /dev/ttyUSB0 --speed 160
                                             # 注入实机后，在浏览器勾选驱动开关
```

### 与 cds_arm 的关系

`arm_ik` 的运动学核心只做计算，不碰串口。硬件读写由 `cds_arm` 负责：
用 `arm_ik` 算出关节角 → `ServoMap.to_ticks` 转成舵机 tick →
`cds_arm.CDSArm.send_goals()` 下发。viewer/IK 可通过注入 `CDSArm` 后端提供
显式关闭的实机驱动开关；启用时会先 `takeover_current()`，再发送新目标。本库把 `ServoMap` 与舵机标定文件
`arm_ik/config/servo_calibration.yaml` 打包，便于换到不同的实机标定。

### 可视化

`viz` extra 依赖 viser，在浏览器里渲染。`replay` 可直接读真实舵机角度驱动数字孪生，
是排查标定（零位/方向）错误的最直接工具。

viewer 和 IK 默认会像 replay 一样自动连接唯一串口；也可以启动时传入
`--device`（或 `--serial`）指定设备。
在浏览器的“实际机械臂”面板中勾选“驱动实际机械臂”。需要纯仿真且不打开串口时使用
`--sim`。关闭驱动开关只停止后续目标发送，舵机扭矩按
`cds_arm` 的语义保持不变。

```bash
uv run arm-ik viz --mode replay
```
