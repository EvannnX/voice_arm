# 语音控制 SO-101 机械臂：逐步实施指南

> **目标**：在 4 月 30 日前完成一个可演示的语音控制杯子递送系统  
> **核心技术栈**：SmolVLA (450M) + Vosk 语音识别 + LeRobot + SO-101  
> **后备方案**：ACT + YOLOv8 + 路径点回放

---

## 当前进度

| 阶段 | 状态 | 备注 |
|------|------|------|
| **第一阶段：硬件搭建与验证** | **进行中** | 环境安装已完成，机械臂校准已完成，摄像头待配置 |
| 第二阶段：数据采集 | 未开始 | |
| 第三阶段：模型训练 | 未开始 | |
| 第四阶段：推理测试 | 未开始 | |
| 第五阶段：语音集成 | 未开始 | |
| 第六阶段：鲁棒性与演示准备 | 未开始 | |

**最后更新：2026-04-17**

---

## 项目文件夹结构

```
voice_control/
├── STEP_BY_STEP_GUIDE.md    ← 你正在看的这个文件
├── scripts/
│   ├── voice_detector.py     ← Vosk 语音指令检测
│   ├── smolvla_infer.py      ← SmolVLA CPU 推理 + 机器人执行
│   ├── main_pipeline.py      ← 端到端集成：语音 → 模型 → 机器人
│   ├── record_dataset.sh     ← 数据采集脚本
│   ├── fallback_act.py       ← ACT 后备方案
│   └── cloud_server.py       ← Colab 云端推理服务（方案 B）
├── configs/
│   ├── smolvla_train.yaml    ← SmolVLA 训练配置
│   └── act_train.yaml        ← ACT 训练配置
├── models/                   ← 存放下载的模型文件（vosk 模型等）
├── data/                     ← 本地数据缓存
├── notebooks/
│   ├── train_smolvla.ipynb   ← Colab SmolVLA 训练笔记本（待创建）
│   └── train_act.ipynb       ← Colab ACT 训练笔记本（待创建）
└── docs/
    └── troubleshooting.md    ← 常见问题排查
```

---

## 第一阶段：硬件搭建与验证（第 1-3 天）

### Day 1：环境安装

1. **~~确认 LeRobot 已安装并添加 SmolVLA 依赖~~ ✅ 已完成**
   - LeRobot 0.5.1 + SmolVLA + peft 0.19.1 已安装
   ```bash
   cd ~/Desktop/ITR_LeRobot/lerobot
   pip install -e ".[feetech,smolvla,peft]"
   ```

2. **~~安装语音识别依赖~~ ✅ 已完成**
   - vosk 0.3.45 + sounddevice 0.5.5 已安装
   ```bash
   pip install vosk sounddevice
   ```

3. **~~下载 Vosk 小型英语模型~~ ✅ 已完成**
   - 模型已下载到 `voice_control/models/vosk-model-small-en-us-0.15/`
   ```bash
   cd ~/Desktop/ITR_LeRobot/voice_control/models
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip vosk-model-small-en-us-0.15.zip
   rm vosk-model-small-en-us-0.15.zip
   ```

4. **~~测试语音识别模型加载~~ ✅ 已完成**
   - Vosk 模型加载正常，语法约束识别已配置
   - 检测到可用音频设备：default (#15), USB Audio (#12)
   - **待做**：连接麦克风后运行实际语音测试
   ```bash
   cd ~/Desktop/ITR_LeRobot/voice_control
   python scripts/voice_detector.py
   ```
   对着麦克风说 "bring me water"，确认控制台输出正确的指令映射。

### Day 2：机械臂校准 ✅ 已完成

1. **~~查找端口~~ ✅**

2. **~~校准两条臂~~ ✅**

3. **~~测试遥操作~~ ✅**
   - 使用 `run_teleop.py` 验证主从臂遥操作正常

### Day 3：摄像头设置 ⏳ 待完成

1. **测试 USB 摄像头**
   ```bash
   # 查找摄像头设备
   ls /dev/video*
   
   # 用 Python 快速测试
   python3 -c "
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   print(f'Camera works: {ret}, Resolution: {frame.shape if ret else \"N/A\"}')
   cap.release()
   "
   ```

2. **确定摄像头索引和分辨率**
   - 记录可用的摄像头索引（0, 2, 4...）
   - 确认 640x480 分辨率可用
   - 固定摄像头位置（正前方或侧方，保持一致）

3. **端到端遥操作 + 摄像头测试**
   ```bash
   lerobot-teleoperate \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM0 \
     --robot.id=my_follower \
     --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
     --teleop.type=so101_leader \
     --teleop.port=/dev/ttyACM1 \
     --teleop.id=my_leader
   ```

> **检查点**：遥操作流畅 + 摄像头画面正常 = 第一阶段完成 ✓

---

## 第二阶段：数据采集（第 4-6 天）

### Day 4-5：录制 50 个回合

1. **准备录制环境**
   - 固定摄像头位置，标记位置以便复现
   - 确保光照一致（避免窗户直射光）
   - 准备带吸管的杯子

2. **规划 5 个杯子起始位置**
   - 位置 1：桌面正中
   - 位置 2：左前方
   - 位置 3：右前方
   - 位置 4：左侧
   - 位置 5：右侧
   - 每个位置录制 10 个回合

3. **开始录制**
   ```bash
   # 修改 scripts/record_dataset.sh 中的端口和 HF 用户名，然后运行：
   bash ~/Desktop/ITR_LeRobot/voice_control/scripts/record_dataset.sh
   ```
   或手动运行：
   ```bash
   lerobot-record \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM0 \
     --robot.id=my_follower \
     --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
     --teleop.type=so101_leader \
     --teleop.port=/dev/ttyACM1 \
     --teleop.id=my_leader \
     --dataset.repo_id=${HF_USER}/so101_cup_handover \
     --dataset.single_task="Pick up the water cup and bring it to the user" \
     --dataset.num_episodes=50
   ```

4. **录制要点**
   - 每个回合 5-15 秒
   - 动作要流畅、稳定，不要急
   - 抓取失败的回合立即丢弃重录
   - 保持一致的递送终点位置

### Day 6：数据验证与上传

1. **可视化检查数据集**
   ```bash
   lerobot-visualize-dataset --repo-id ${HF_USER}/so101_cup_handover
   ```

2. **逐个回合检查**
   - 确认所有 50 个回合的摄像头画面清晰
   - 确认动作轨迹合理
   - 删除并重录任何有问题的回合

3. **上传到 HuggingFace Hub**
   ```bash
   huggingface-cli login
   # 数据集会在录制时自动上传，确认 Hub 上可见
   ```

> **检查点**：50 个高质量回合已上传至 HF Hub ✓

---

## 第三阶段：模型训练（第 7-10 天）

### Day 7-8：ACT 基线训练（保底方案）

1. **在 Colab 上打开 ACT 训练笔记本**
   - 访问 `https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-act.ipynb`
   - 切换运行时为 T4 GPU

2. **运行 ACT 训练**
   ```bash
   python -m lerobot.scripts.train \
     --dataset.repo_id=${HF_USER}/so101_cup_handover \
     --policy.type=act \
     --output_dir=/content/drive/MyDrive/act_cup \
     --batch_size=8 \
     --steps=100000
   ```
   - T4 上大约 2-4 小时完成
   - 训练完成后下载模型到本地测试

3. **本地测试 ACT**
   ```bash
   lerobot-record \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM0 \
     --robot.id=my_follower \
     --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
     --policy.path=${HF_USER}/act_cup_handover \
     --policy.device=cpu \
     --dataset.single_task="Pick up the water cup and bring it to the user" \
     --dataset.repo_id=${HF_USER}/eval_act_cup
   ```

### Day 9-10：SmolVLA 微调

1. **在 Colab 上安装依赖**
   ```bash
   !git clone https://github.com/huggingface/lerobot.git
   %cd lerobot
   !pip install -e ".[smolvla,peft]"
   !pip install torchcodec --force-reinstall
   ```

2. **挂载 Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **开始 SmolVLA 完整微调**
   ```bash
   python -m lerobot.scripts.train \
     --policy.path=lerobot/smolvla_base \
     --dataset.repo_id=${HF_USER}/so101_cup_handover \
     --batch_size=16 \
     --steps=20000 \
     --save_freq=2000 \
     --policy.use_amp=true \
     --output_dir=/content/drive/MyDrive/smolvla_cup \
     --wandb.enable=false
   ```

4. **如果 OOM，切换到 LoRA**
   ```bash
   python -m lerobot.scripts.train \
     --policy.path=lerobot/smolvla_base \
     --dataset.repo_id=${HF_USER}/so101_cup_handover \
     --batch_size=32 \
     --steps=20000 \
     --peft.method_type=LORA \
     --peft.r=64 \
     --peft.lora_alpha=16 \
     --policy.optimizer_lr=1e-3 \
     --policy.use_amp=true \
     --output_dir=/content/drive/MyDrive/smolvla_cup_lora
   ```

5. **跨 Colab 会话恢复训练**
   - 在浏览器控制台运行防断连脚本：
     ```javascript
     setInterval(() => { document.querySelector("colab-connect-button")?.click() }, 60000)
     ```
   - 添加 `--resume=true` 从最新检查点继续

> **检查点**：ACT 模型可用（保底）+ SmolVLA 训练完成或接近完成 ✓

---

## 第四阶段：推理测试（第 11-13 天）

### Day 11：下载模型并测试

1. **从 Google Drive 下载训练好的 SmolVLA 模型**
   ```bash
   # 或者通过 HF Hub 下载（如果已上传）
   # 将模型放到 voice_control/models/ 目录
   ```

2. **CPU 推理测试**
   ```bash
   python ~/Desktop/ITR_LeRobot/voice_control/scripts/smolvla_infer.py
   ```

3. **测量推理速度**
   - 目标：>0.5 Hz（每次推理 <2 秒）
   - 如果太慢，尝试：
     - `torch.compile()` 加速
     - 降低图像分辨率到 320x240
     - 增大 `n_action_steps`（更大的动作块）

### Day 12：优化推理速度

如果 CPU 推理速度不够（<0.3 Hz）：

**选项 A - 继续优化 CPU**：
```python
import torch
model = torch.compile(model, mode="reduce-overhead")
```

**选项 B - Colab 云端推理**：
```bash
# 在 Colab 运行 cloud_server.py
# 本地通过 ngrok URL 调用
python scripts/cloud_server.py
```

### Day 13：机器人执行测试

1. **连接机器人测试完整推理→执行循环**
   ```bash
   lerobot-record \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM0 \
     --robot.id=my_follower \
     --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
     --policy.path=${HF_USER}/smolvla_cup_handover \
     --policy.device=cpu \
     --dataset.single_task="Pick up the water cup and bring it to the user" \
     --dataset.repo_id=${HF_USER}/eval_cup_handover
   ```

2. **观察并记录**
   - 机器人是否能定位杯子？
   - 抓取是否稳定？
   - 递送路径是否合理？
   - 记录成功率（目标：>50%）

> **检查点**：模型能在 CPU 上运行并控制机器人完成基本抓取 ✓

---

## 第五阶段：语音集成（第 14-16 天）

### Day 14：独立测试语音模块

1. **运行语音检测器**
   ```bash
   python scripts/voice_detector.py
   ```

2. **测试所有指令**
   - "bring me water" → 应输出 "Pick up the water cup and bring it to the user"
   - "pick up the cup" → 应输出 "Pick up the cup from the desk"
   - "stop" → 应触发紧急停止
   - 测试不同音量、距离、口音

### Day 15：端到端集成

1. **运行完整流水线**
   ```bash
   python scripts/main_pipeline.py
   ```

2. **测试流程**
   - 说 "bring me water"
   - 观察机器人是否开始动作
   - 说 "stop" 测试紧急停止
   - 重复 10 次，记录成功率

### Day 16：调试与优化

1. **常见问题排查**
   - 语音误识别 → 调整 Vosk 语法列表或麦克风位置
   - 机器人动作不连贯 → 增大动作块大小
   - 抓取不稳 → 检查数据集中夹爪动作是否一致

2. **录制 10 次成功的端到端运行**

> **检查点**：说 "bring me water" → 机器人完成杯子递送 ✓

---

## 第六阶段：鲁棒性与演示准备（第 17-20 天）

### Day 17-18：鲁棒性测试

1. **变化测试**
   - 不同杯子位置（训练范围内和边界位置）
   - 不同光照（开灯/关灯/自然光）
   - 不同说话人
   - 连续多次执行

2. **添加错误处理**
   - "stop" 指令的紧急停止
   - 抓取超时检测
   - 语音无响应时的重试提示

3. **如果 SmolVLA 不可靠，切换到 ACT 后备方案**
   ```bash
   python scripts/fallback_act.py
   ```

### Day 19-20：演示准备

1. **准备演示环境**
   - 固定桌面布局
   - 标记杯子放置位置（可用胶带）
   - 确保光照一致
   - 测试麦克风最佳位置

2. **练习演示流程（至少 10 次）**
   ```
   演示流程：
   1. 启动系统 → 显示 "Listening for commands..."
   2. 放置杯子到标记位置
   3. 说 "bring me water"
   4. 机器人抓取杯子并递送
   5. 说 "put it down" 或手动复位
   6. 重复展示
   ```

3. **准备备份**
   - 录制一段成功运行的视频
   - 准备 ACT 后备方案模型
   - 准备路径点回放的最终后备

4. **制作文档/幻灯片**
   - 项目概述
   - 技术架构图
   - 演示视频截图
   - 遇到的挑战和解决方案

---

## 关键注意事项清单

### 必须记住的事项

- [ ] `single_task` 字符串在训练和推理时**必须完全一致**
- [ ] 摄像头名称（如 `front`）在录制和推理时**必须一致**
- [ ] T4 不支持 bfloat16，必须用 `--policy.use_amp=true`（float16）
- [ ] 每次 Colab 会话开始时挂载 Google Drive
- [ ] 训练 `--save_freq=2000` 防止进度丢失
- [ ] USB 线直连，不要用集线器

### 时间分配建议

| 阶段 | 天数 | 占比 |
|------|------|------|
| 硬件搭建 | 3 天 | 15% |
| 数据采集 | 3 天 | 15% |
| 模型训练 | 4 天 | 20% |
| 推理测试 | 3 天 | 15% |
| 语音集成 | 3 天 | 15% |
| 鲁棒性+演示 | 4 天 | 20% |

### 决策节点

- **Day 8**：ACT 基线是否工作？→ 如果是，继续 SmolVLA；如果否，专注调试 ACT
- **Day 12**：SmolVLA CPU 推理速度可接受？→ 如果否，切换到云端推理或 ACT
- **Day 15**：端到端集成是否工作？→ 如果否，启用后备方案
- **Day 18**：演示可靠性 >70%？→ 如果否，降级到路径点回放

---

## 需要克隆的仓库

```bash
# 1. LeRobot（如果还没有）
git clone https://github.com/huggingface/lerobot.git

# 2. 官方笔记本
git clone https://github.com/huggingface/notebooks.git

# 3. 语音控制机械臂参考项目
git clone https://github.com/M4YH3M-DEV/6-DOF-Voice-Controlled-Robotic-Arm.git

# 4. YOLOv8（后备方案用）
pip install ultralytics
```
