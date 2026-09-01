# `uhand` Python API

## 连接

```python
connect(
    port: str = "auto",
    *,
    timeout: float = 0.0,
    write_timeout: float = 0.1,
    startup_delay: float = 2.0,
    calibration: FingerCalibration | None = None,
) -> UHand
```

- 支持 UNO Type-B USB 直连；
- 波特率固定为 `115200`，连接入口不提供覆盖参数；
- 打开端口后默认等待 2 秒，让 USB 串口触发复位的 UNO 完成启动；
- `port="auto"` 优先识别 `usbmodem/ttyACM/Arduino` 设备，无法唯一确定时拒绝猜测。

`connect()` 只建立 USB 连接，不发送任何手指姿态。`with` 退出或 `close()` 只关闭串口，不改变
五指目标，也不会操作任何机械臂关节。

## 角度与开合

所有五元素参数的固定顺序为：

```text
(thumb, index, middle, ring, pinky)
```

### `command_fingers()`

```python
hand.command_fingers(
    finger_angles: Sequence[float],
    *,
    wait: bool = False,
    timeout: float = 1.0,
) -> tuple[int, int, int, int, int]
```

向发送线程提交一个最新目标。尚未写出的旧目标会被覆盖，适合视觉或规划实时循环。

### `set_fingers()`

```python
hand.set_fingers(
    finger_angles: Sequence[float],
    *,
    duration: float = 0.0,
    rate_hz: float = 30.0,
) -> tuple[int, int, int, int, int]
```

阻塞到最终数据帧写入串口。`duration > 0` 时从上一次请求目标做线性插值。

### `set_finger()`

```python
hand.set_finger(
    finger: str,
    angle: float,
    *,
    duration: float = 0.0,
    rate_hz: float = 30.0,
) -> tuple[int, int, int, int, int]
```

`finger` 接受 `thumb/index/middle/ring/pinky` 或对应的 `H1/H2/H3/H4/H5`，并兼容
`little`。其他四指保持程序记录的上一次目标。这里的“保持”不是读取物理反馈。

### `set_closures()`

```python
hand.set_closures(
    closures: Sequence[float],
    *,
    duration: float = 0.0,
    rate_hz: float = 30.0,
) -> tuple[int, int, int, int, int]
```

每个值范围为 `[0, 1]`，`0` 映射到该指 `open_angles`，`1` 映射到 `closed_angles`。

## 手势与抓取

### `gesture()`

```python
hand.gesture(
    name: str,
    *,
    amount: float = 1.0,
    duration: float = 0.5,
    rate_hz: float = 30.0,
) -> tuple[int, int, int, int, int]
```

| 名称 | 归一化开合 `(thumb,index,middle,ring,pinky)` |
|---|---|
| `open` | `(0,0,0,0,0)` |
| `fist` / `power` | `(1,1,1,1,1)` |
| `point` | `(1,0,1,1,1)` |
| `victory` | `(1,0,0,1,1)` |
| `thumbs_up` | `(0,1,1,1,1)` |
| `pinch` | `(1,1,0,0,0)` |
| `tripod` | `(1,1,1,0,0)` |

`amount` 在张开姿态与完整手势之间缩放所有闭合分量。

### `set_grasp()`

```python
hand.set_grasp(
    grasp_type: str,
    *,
    closure: float = 1.0,
    duration: float = 0.5,
    rate_hz: float = 30.0,
) -> tuple[int, int, int, int, int]
```

只接受 `open/power/pinch/tripod`，是给上层任务规划使用的窄接口。

## 协议边界

```python
build_finger_packet(finger_angles: Sequence[float]) -> bytes
```

输入必须恰好是五个 `0..180` 角度。内部输出兼容旧固件：

```text
AA 77 01 06 H1 H2 H3 H4 H5 RESERVED CHECK
```

`RESERVED` 永远固定为 `90`。传入六个值会抛出 `FingerValueError`。它不是手腕目标，API 也没有
设置它的入口。

## 状态和异常

- `hand.last_target`：最后请求的五指目标，不是实机反馈；
- `hand.available_gestures`：可用预设名称；
- `FingerValueError`：角度、开合量、手指名称或手势名称非法；
- `UHandError`：串口发现、连接、后台发送或等待超时失败。
