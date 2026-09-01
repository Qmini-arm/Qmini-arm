# uHand 五指控制 API

这个包把已跑通的 `hand_mirror_demo` 串口发送与执行链路整理成可复用的 Python API。
它只控制五根手指：

```text
H1 thumb · H2 index · H3 middle · H4 ring · H5 pinky
```

API 中没有手腕、第六轴或旋转轴参数。为兼容已验证协议，固件仍接收六字节舵机帧，因此驱动会在内部追加固定
占位值 `90`；调用者不能读取或修改这个占位值，机械臂手腕 `W1` 也绝不能从这里发送。

## 烧录五指直连固件

先在 Arduino IDE 中打开并烧录：

```text
arduino/uHand_UNO_usb_serial/uHand_UNO_usb_serial.ino
```

固件通过 UNO Type-B USB 的硬件串口以 `115200` 接收命令，只初始化 H1～H5 对应的
`D7/D6/D5/D4/D3`。旧第六路对应的 `D2` 不初始化、不输出舵机脉冲；协议中的第六个兼容字节
会被读取但直接忽略。烧录后关闭 Arduino IDE 串口监视器，再运行 Python 程序。

## 安装 Python API

```bash
cd Qmini-arm/hand_serial_tools
python3 -m pip install .
```

## 最小示例

API 只支持 UNO Type-B USB 直连，波特率固定为 `115200`：

```python
from uhand import connect

with connect("/dev/cu.usbmodem1301") as hand:
    hand.gesture("open", duration=0.5)
    hand.gesture("victory", duration=0.6)
    hand.gesture("fist", amount=0.7, duration=0.8)
```

## 四种控制方式

五指顺序始终是 `thumb, index, middle, ring, pinky`。

```python
# 1. 五指逻辑角度，范围 0..180
hand.set_fingers([180, 90, 45, 20, 0], duration=0.7)

# 2. 单独修改一根手指，其他手指保持上一次目标
hand.set_finger("index", 180, duration=0.3)

# 3. 归一化开合：0=标定的张开端点，1=标定的闭合端点
hand.set_closures([0.0, 0.25, 0.5, 0.75, 1.0], duration=0.8)

# 4. 预设手势
hand.gesture("point", duration=0.5)
```

预设包括：`open`、`fist`、`power`、`point`、`victory`、`thumbs_up`、`pinch`、
`tripod`。`power` 是任务层的包络抓取名称，与 `fist` 使用相同的基础开合向量。

抓取任务也可以使用更窄的入口：

```python
hand.set_grasp("pinch", closure=0.65, duration=0.8)
```

`set_grasp()` 只接受 `open/power/pinch/tripod`。

## 标定端点

默认沿用已验证 Demo 的逻辑端点：张开为 `180`，闭合为 `0`。如果某根手指的实机端点不同，
应传入逐指标定值，而不是修改手势表：

```python
from uhand import FingerCalibration, connect

calibration = FingerCalibration(
    open_angles=(170, 175, 178, 176, 172),
    closed_angles=(15, 10, 12, 14, 18),
)

with connect("/dev/ttyACM0", calibration=calibration) as hand:
    hand.gesture("pinch", amount=0.6, duration=1.0)
```

角度或开合量越界时 API 会直接报错，不会静默截断。

## 实时控制

外部视觉或规划循环可以调用非阻塞的最新目标接口：

```python
hand.command_fingers(latest_five_angles)
```

发送线程只保留尚未发送的最新目标，不让过时姿态在串口队列中累积；这与
`hand_mirror_demo` 已验证的实时发送策略一致。`set_fingers()`、`set_closures()` 和
`gesture()` 则会以默认 `30 Hz` 生成阻塞式线性过渡。

`last_target` 只是程序最后请求的五指目标，不是舵机真实位置反馈。现有 uHand 舵机没有真实
位置或力反馈，抓取物体时应限制闭合程度并预留人工断电手段。

完整签名和异常语义见 [LIBRARY_API.md](./LIBRARY_API.md)。
