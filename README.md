# Qmini-arm

Qmini机械臂的URDF描述、网格资源和CDS55xx六轴基础运动Python库。

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
