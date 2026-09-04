# Qmini 石头剪刀布 Demo

这个独立 Demo 使用电脑摄像头识别人手的石头、剪刀或布，并让 Qmini uHand 实时给出
能够获胜的手势：

| 人手 | 机械手 | uHand 预设 |
|---|---|---|
| 石头 `rock` | 布 `paper` | `open` |
| 布 `paper` | 剪刀 `scissors` | `victory` |
| 剪刀 `scissors` | 石头 `rock` | `fist` |

程序只控制 uHand 的五根手指，不连接、不接管也不移动机械臂六轴。

## 实现方式

Demo 直接复用相邻 `qmini_avatar` 中已经验证的 MediaPipe 关键点提取和逐指视觉标定，得到
`thumb, index, middle, ring, pinky` 五个 `0..1` 闭合量，再做 RPS 分类：

- 食指、中指、无名指、小指均闭合：石头；
- 四根长指均张开：布；
- 食指和中指张开、无名指和小指闭合：剪刀；
- 位于阈值中间的不确定姿态不猜测。

拇指不参与形状判定，因为拇指弯曲分数对手掌朝向更敏感，而且合法的石头和剪刀姿势中
拇指位置也可能不同。默认要求同一结果连续出现 5 帧才接受，避免过渡动作导致机械手抖动。

机械手输出复用现有 `uhand` 公共 API 和 Arduino 固件。发送的是 `open/victory/fist` 预设，
程序没有复制或重新实现串口协议。

## 安装与自检

从仓库根目录执行：

```bash
cd demos/qmini_rps
uv sync
uv run qmini-rps --self-test
uv run pytest
```

如果不使用 `uv`：

```bash
cd demos/qmini_rps
python3 -m venv .venv
.venv/bin/python -m pip install -e ../.. -e ../../hand_serial_tools -e ../qmini_avatar -e .
.venv/bin/qmini-rps --self-test
```

## 先运行纯相机模式

不带实机参数时只打开摄像头，不扫描或连接串口：

```bash
uv run qmini-rps
```

把手稳定放在画面内。界面中的 `seen` 是当前帧分类，`stable` 是连续多帧确认后的结果，
`robot` 是机械手应该展示的获胜手势。

程序默认使用 `MJPG 320x240`。实测当前 UVC 摄像头通过 WSL USB/IP 使用 `YUYV 640x480`
时会整帧发绿，使用 `MJPG 640x480` 时又会发生下半帧截断；`MJPG 320x240@30fps` 可以稳定
传输完整画面。在非 WSL 环境中遇到不支持 MJPG 的摄像头时，可以恢复 OpenCV 自动选择：

```bash
uv run qmini-rps --camera-format auto
```

直连 Linux 或更换传输稳定的摄像头后，可以手动尝试更高分辨率：

```bash
uv run qmini-rps --width 640 --height 480
```

## 连接机械手

确保已经给 Arduino UNO 烧录仓库中的五指直连固件，然后运行：

```bash
uv run qmini-rps --live
```

程序会使用 `uhand` 的规则自动寻找唯一的 UNO Type-B USB 端口。无法唯一识别时显式指定：

```bash
uv run qmini-rps --hand-port /dev/cu.usbmodem1301
# Linux 示例：--hand-port /dev/ttyACM0
```

即使使用 `--live`，程序启动后也保持暂停。操作按键：

- `Space`：开始或暂停机械手响应；
- `R`：暂停、清除识别状态并让机械手张开；
- `Q` 或 `Esc`：退出。

手部丢失超过 0.6 秒后，程序会让机械手张开并自动暂停，必须再次按 `Space` 才会继续。
退出只关闭串口；普通舵机可能继续保持最后目标。

## 调整识别

默认复用 Avatar 的实机采集标定：

```text
../qmini_avatar/config/finger_calibration.json
```

可以换成另一份兼容标定，并调整判定阈值或连续帧数：

```bash
uv run qmini-rps --live \
  --finger-calibration /path/to/finger_calibration.json \
  --open-threshold 0.35 \
  --closed-threshold 0.65 \
  --stable-frames 5
```

如果张开或握拳经常显示 `uncertain`，应优先重新采集视觉标定；只有过渡区过宽或过窄时才调整
阈值。必须保持 `open-threshold < closed-threshold`。
