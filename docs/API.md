# `cds_arm` Python库调用接口

## 安装

从源码目录安装：

```bash
git clone https://github.com/zgdllt/Qmini-arm.git
cd Qmini-arm
python3 -m pip install .
```

或者复制wheel到目标机器后安装：

```bash
python3 -m pip install cds_arm_control-0.2.0-py3-none-any.whl
```

库名为`cds_arm`，发行包名为`cds-arm-control`。

## 最小调用流程

```python
from cds_arm import CENTER, connect

with connect("/dev/ttyUSB0", baud=115200, timeout=0.08) as arm:
    # 1. 只读检查
    print(arm.verify_position_modes())
    print(arm.read_positions())

    # 2. 明确接管；这是可能开启扭矩的步骤
    arm.takeover_current(speed=160)

    # 3. 阻塞运动到目标
    feedback = arm.move(
        dict(CENTER),
        label="回正式中心",
        speed=160,
        arrival_tolerance=5,
        trim_step=2,
        trim_max=48,
    )
    print("到位反馈：", feedback)
```

`connect()`只打开串口，不开启扭矩、不发送运动目标。`takeover_current()`会把当前反馈同步写成
目标，并在六轴原本全部关闭扭矩时同步开启扭矩。上下文退出只关闭串口，不会自动卸载扭矩。

## 顶层接口

### `connect()`

```python
connect(
    port: str = "auto",
    *,
    baud: int = 115200,
    timeout: float = 0.08,
) -> ContextManager[CDSArm]
```

- `port`：Mac可用`/dev/cu.usbserial-*`，Linux/香橙派常用`/dev/ttyUSB0`或
  `/dev/ttyACM0`；`auto`会自动识别唯一候选。
- `baud`：当前机械臂为`115200`。
- `timeout`：单次应答等待时间。每次寄存器读取默认允许额外重试2次。
- 返回上下文中的`CDSArm`控制器。

### `validate_positions()`

```python
validate_positions(values: dict[int, int], context: str) -> None
```

检查字典是否完整包含ID 1～6，并逐轴检查软件安全范围。非法时抛出`SafetyError`。

## `CDSArm`高层方法

### `read_positions()`

```python
arm.read_positions() -> dict[int, int]
```

依次读取ID 1～6的`0x24`当前位置，校验反馈仍位于软件安全范围后返回。
这是只读操作。

### `sample_positions()`

```python
arm.sample_positions(rounds: int = 3) -> dict[int, list[int]]
```

连续采样六轴，返回每个ID的原始反馈列表。这是只读操作。

### `stable_sample()`

```python
arm.stable_sample(
    *,
    rounds: int = 3,
    max_drift: int = 12,
    allow_zero: bool = False,
) -> dict[int, int]
```

采样并验证机械臂在读取期间没有明显移动。默认拒绝某个舵机连续反馈0，以避免把已知零值盲区
误认为真实位置。返回每轴最新的稳定反馈。

### `verify_position_modes()`

```python
arm.verify_position_modes() -> dict[int, tuple[int, int]]
```

读取六轴内部角度限位，拒绝连续旋转模式，并确认内部限位能够覆盖软件安全范围。只读。

### `torque_states()`

```python
arm.torque_states() -> dict[int, int]
```

读取六轴扭矩开关。`0`表示关闭，`1`表示开启。只读。

### `takeover_current()`

```python
arm.takeover_current(
    *,
    speed: int = 160,
    max_drift: int = 12,
    allow_zero: bool = False,
) -> dict[int, int]
```

推荐的安全接管接口，依次执行：

1. 检查六轴位置模式和内部限位；
2. 读取扭矩状态；
3. 连续采样当前位置；
4. 把当前位置同步写为六轴目标；
5. 如果六轴扭矩原本全部关闭，则同步开启；
6. 如果扭矩状态有开有关，则拒绝接管。

返回接管位置。调用前必须人工托住机械臂并准备断电。

### `send_goals()`

```python
arm.send_goals(
    positions: dict[int, int],
    speed: int,
    *,
    verify: bool = True,
) -> bytes
```

把完整六轴位置和速度打包为一个`SYNC WRITE`目标帧。发送前检查全部软件范围，默认发送后回读
六轴目标寄存器。返回实际发送的字节帧。

这是底层运动写接口，不等待到位。一般优先使用`move()`。

### `move()`

```python
arm.move(
    goal: dict[int, int],
    *,
    label: str = "运动",
    speed: int = 160,
    arrival_tolerance: int = 5,
    settle_time: float = 0.4,
    monitor_period: float = 0.08,
    stall_timeout: float = 6.0,
    total_timeout: float = 20.0,
    trim_step: int = 2,
    trim_max: int = 48,
    trim_window: int = 30,
    trim_interval: float = 0.10,
    trim_start_delay: float = 0.25,
) -> dict[int, int]
```

阻塞式六轴安全运动接口：

- 起点、名义目标、所有补偿目标和每轮实际反馈都经过软件安全范围检查；
- 首次同步下发名义目标；
- 正常运动停止进展并进入`trim_window`后，启动连续反馈微调；
- 默认每次沿目标方向补偿`2 raw`，越过名义目标后该轴降为`1 raw`反向修正；
- 每轴实际误差进入`arrival_tolerance`并连续稳定`settle_time`后返回；
- 失联、越界、停滞或总超时会抛出异常；
- 返回到位时六轴实际反馈。

`move()`不会自行开启扭矩。正常流程应先调用`takeover_current()`。

### `diagnostics()`

```python
arm.diagnostics() -> dict[int, dict[str, int | float]]
```

实际返回类型为：

```python
dict[int, dict[str, int | float]]
```

只读返回六轴位置、运动标志、负载、电压、温度、最大扭矩、最小PWM、死区和双向P增益，
不在库内部打印，方便调用程序自行记录或展示。

### `close()`

```python
arm.close() -> None
```

关闭串口，不关闭舵机扭矩。使用`with connect(...)`时通常不需要手动调用。

## 底层寄存器接口

### `read_registers()`

```python
arm.read_registers(
    servo_id: int,
    address: int,
    size: int,
    retries: int = 2,
) -> bytes
```

发送`READ DATA`并解析状态帧。`retries=2`表示首次失败后最多再发2次，即连续3次失败才抛出
`RuntimeError`。舵机状态字非零时也会抛出异常。

### `read_u16()`

```python
arm.read_u16(servo_id: int, address: int) -> int
```

以小端序读取双字节寄存器。

### 协议辅助函数

```python
build_packet(servo_id, instruction, params=()) -> bytes
checksum_is_valid(frame: bytes) -> bool
decode_error(error: int) -> str
```

用于需要自行扩展CDS55xx指令的程序。

## 导出的配置常量

```python
from cds_arm import (
    CENTER,       # 正式反馈中心
    SAFE_LIMITS,  # 六轴软件安全范围
    SERVO_IDS,
    DEFAULT_BAUD,
    POSITION_MAX,
)
```

当前正式中心：

```python
{1: 812, 2: 122, 3: 144, 4: 481, 5: 359, 6: 88}
```

## 异常与安全语义

```python
from cds_arm import SafetyError, connect

try:
    with connect("/dev/ttyUSB0") as arm:
        arm.takeover_current()
        arm.move(...)
except SafetyError as exc:
    print("安全检查或运动失败：", exc)
except RuntimeError as exc:
    print("串口或舵机状态失败：", exc)
```

- `SafetyError`：位置模式、ID集合、安全范围、目标回读或运动安全条件不满足；
- `RuntimeError`：连续失联或舵机状态字报错；
- `ValueError`：调用参数非法。

发生异常时库不会自动发送新目标，也不会自动关闭扭矩。调用程序应提示操作者托住机械臂，
并保留硬件断电手段。
