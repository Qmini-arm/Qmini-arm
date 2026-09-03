# Qmini Avatar 实验性视觉遥操作 Demo

这个独立 demo 使用普通电脑摄像头在本机运行 MediaPipe，把一只人手映射到
Qmini 机械臂和 uHand：

- 手靠近/远离摄像头 → 机械臂末端 `+X/-X`；
- 画面中左右移动 → 机械臂末端横向移动；
- 画面中上下移动 → 机械臂末端上下移动；
- 掌部旋转 → 独立的第六轴腕旋转；
- 五指弯曲 → uHand 五指开合。

这是相对控制。程序不会假设摄像头与机械臂之间存在精确外参，而是用按下 `C` 时的人手和
机械臂姿态作为共同中立点。

## 安全状态机

程序默认暂停，而且不传串口参数时不会扫描或打开任何硬件：

1. 将一只手稳定放在画面中央；
2. 按 `C` 捕获中立姿态；
3. 小幅移动手，观察目标坐标；
4. 按 `Space` 才开始提交目标；再次按下立即暂停；
5. 人手丢失超过 0.45 秒后自动暂停，必须人工再次按 `Space`；
6. 大关节变化会被逐帧限速；只有安全路径仍发生自碰撞时才拒绝该帧。

启动时程序会在收紧后的真实舵机限位内采样 6000 个构型，剔除自碰撞构型，再建立机械臂的
可达空间点集。人手产生的长方体坐标目标不会直接交给 IK：已知可达区域内保留连续运动，
落在薄壳空洞或壳外的目标会吸附到邻近的无碰撞可达点。跨 IK 分支的动作会按关节步长逐帧
逼近，而不是持续报 `limit_blocked` 或突然跳动。

暂停、退出或关闭串口不会关闭机械臂扭矩。实机运行时必须托住机械臂，并保留硬件断电手段。

## 安装

Avatar 有自己的 `pyproject.toml`，视觉依赖不会进入仓库根部的底层包。从仓库根目录执行：
需要 Python 3.10～3.12；当前 macOS 实机验证环境使用 Python 3.12 与 MediaPipe 0.10.21。

```bash
cd demos/qmini_avatar
uv sync
uv run qmini-avatar --self-test
uv run pytest
```

如果不使用 `uv`：

```bash
cd demos/qmini_avatar
python3 -m venv .venv
.venv/bin/python -m pip install -e ../.. -e ../../hand_serial_tools -e .
.venv/bin/qmini-avatar --self-test
```

只打开摄像头和运动学仿真，不扫描或连接串口：

```bash
uv run qmini-avatar
```

## 连接实机

完整模式只需要传 `--live`。程序会联合扫描两类设备，自动把
`usbserial/ttyUSB/CH340/CP210/FTDI/UP-Debugger` 分给机械臂，把
`usbmodem/ttyACM/Arduino UNO` 分给五指手，并保证同一端口不会被重复使用：

```bash
uv run qmini-avatar --live
```

查看扫描依据和每个设备的角色评分：

```bash
uv run qmini-avatar --list-ports
```

自动检测遇到多个同分候选时会拒绝猜测。这种情况下才需要手动覆盖。也可以先逐个连接：

```bash
# 只接机械臂，五指仍为画面模拟
uv run qmini-avatar \
  --arm-port /dev/cu.usbserial-XXXX

# 手动指定的完整 Avatar
uv run qmini-avatar \
  --arm-port /dev/cu.usbserial-XXXX \
  --hand-port /dev/cu.usbmodemXXXX
```

`--arm-port auto` 和 `--hand-port auto` 也受联合检测约束。默认不加 `--live` 或任何端口参数时
仍然是纯仿真，避免一次普通摄像头测试意外接管机械臂。

## 调整映射

默认映射范围相对保守：前后约 ±55 mm、横向/高度约 ±75 mm。灵敏度可以分别调整：

```bash
uv run qmini-avatar \
  --depth-gain 0.10 \
  --lateral-gain 0.20 \
  --vertical-gain 0.20 \
  --smoothing 0.35
```

如果腕旋转方向与预期相反，使用 `--roll-gain -1`。降低 `--smoothing` 会更稳定，增大则更灵敏。

五指链路逐指计算两个关节夹角之和，读取 `config/finger_calibration.json` 中实机采集的张开、
握拳分数和舵机端点，然后在角度域采用 30 Hz、EMA 0.65、单次最大 12° 的平滑策略。可以复制
并修改这份 JSON，或显式指定另一份兼容标定：

```bash
uv run qmini-avatar --live \
  --finger-calibration /path/to/finger_calibration.json
```

如需微调动态响应，可以使用 `--finger-smoothing`、`--finger-max-step`、`--hand-hz`；默认值与
Hand Mirror 一致，通常不需要修改。

可达空间精度和启动速度之间也可以调整：

```bash
uv run qmini-avatar --live \
  --workspace-samples 10000 \
  --workspace-tolerance-mm 10
```

## 当前 MVP 边界

- 单目摄像头的前后距离来自手掌视觉大小，不是精确深度；
- 机械臂使用位置 IK 加独立腕旋转，不追踪完整手掌 6D 姿态；
- uHand 没有真实手指位置或力反馈；
- 当前只有自碰撞检查，没有桌面、人体或其他外部障碍物模型；
- 实时控制使用已发送目标作为下一帧 IK 种子，尚未加入高频关节反馈融合。
