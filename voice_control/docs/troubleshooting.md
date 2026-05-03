# 常见问题排查

## 硬件问题

### Feetech 舵机通信错误："Incorrect status packet"
- **原因**：固件版本不对、电压不对或 USB 连接不稳定
- **解决**：
  1. USB 线直连电脑，**不要用集线器**
  2. 尝试更换线缆插槽
  3. STS3215 舵机需要 v3.10+ 固件
  4. 用 Feetech FT SCServo Debug 工具（Windows）检查固件

### 电机初始化显示 "ID_MODEL None"
- 尝试更换线缆插槽
- 检查舵机 ID 是否冲突（每条总线上 ID 必须唯一）

### "JointOutOfRangeError" 或 "inf%" 错误
- 删除校准文件并重新校准：
  ```bash
  rm -rf ~/.cache/huggingface/lerobot/calibration/
  lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
  ```

### 端口权限问题
```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
# 或永久解决：
sudo usermod -aG dialout $USER
# 然后重新登录
```

---

## 训练问题

### NaN 损失 / 输出乱码（T4 上）
- **原因**：T4（Turing 架构）不支持 bfloat16
- **解决**：确保使用 `--policy.use_amp=true`（强制 float16）
- 检查代码中没有硬编码 `torch.bfloat16`

### Colab OOM（显存不足）
- 降低 `batch_size`（16 → 8）
- 切换到 LoRA：`--peft.method_type=LORA --peft.r=64`
- 清除 GPU 缓存：`torch.cuda.empty_cache()`
- 重启 Colab 运行时

### Colab 断开连接 / 进度丢失
- 挂载 Google Drive，`--output_dir` 指向 Drive 路径
- 使用 `--save_freq=2000`
- 浏览器控制台运行防断连：
  ```javascript
  setInterval(() => { document.querySelector("colab-connect-button")?.click() }, 60000)
  ```
- 断开后用 `--resume=true` 恢复

### torchcodec 错误（Colab）
```bash
pip install torchcodec --force-reinstall
```

---

## 推理问题

### CPU 推理太慢（< 0.3 Hz）
- 尝试 `torch.compile(model, mode="reduce-overhead")`
- 降低图像分辨率到 320x240
- 增大 `n_action_steps`（更大的动作块 = 更少的推理调用）
- 最终方案：切换到 Colab 云端推理（`cloud_server.py`）

### 动作输出无意义 / 机器人乱动
- **检查 `single_task` 字符串**：训练和推理必须完全一致
- **检查摄像头名称**：录制时的 `front` 必须和推理时一样
- **检查摄像头顺序**：多摄像头时顺序必须一致
- 确认模型检查点正确加载（不是中间检查点）

### 抓取不稳定
- 检查训练数据中夹爪动作是否一致
- 可能需要更多训练数据（增加到 70-100 回合）
- 调整动作执行频率

---

## 语音识别问题

### Vosk 模型找不到
```bash
cd ~/Desktop/ITR_LeRobot/voice_control/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

### 麦克风无输入
```bash
# 检查可用音频设备
python -c "import sounddevice; print(sounddevice.query_devices())"

# 测试录音
python -c "
import sounddevice as sd
import numpy as np
print('Recording 3 seconds...')
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='int16')
sd.wait()
print(f'Max amplitude: {np.abs(audio).max()}')
if np.abs(audio).max() < 100:
    print('WARNING: Very low audio level, check microphone!')
else:
    print('Microphone is working.')
"
```

### 指令识别不准
- 靠近麦克风说话
- 减少背景噪音
- 检查 Vosk 语法列表中的指令拼写
- 尝试更大的 Vosk 模型（但会更慢）

---

## 演示当天检查清单

- [ ] 两条臂都通电，USB 直连
- [ ] 摄像头固定在正确位置
- [ ] 麦克风连接并测试
- [ ] 光照条件与训练时一致
- [ ] 杯子放在训练范围内的位置
- [ ] 运行 `voice_detector.py` 确认语音识别正常
- [ ] ACT 后备模型已就绪
- [ ] 成功运行视频已录制作为 backup
