# voice_arm 一周 Mock 实现计划

> 目标：7 天内，在纯 Mock 模式下交付一个可演示的语音控制机械臂 demo。
> 原则：每天都能跑起来 + 每天多一层可见性。

---

## 当前基线（已就绪）

- MockArm + ArmController ABC（8 个 primitive：move_to / move_relative / grasp / release / home / stop / set_speed / get_state）
- Gemini Live 会话 + 系统 prompt
- ToolDispatcher + FunctionDeclaration（已绑定到 8 个 primitive）
- Audio pipeline（MicStream / Speaker）
- 已有单测：dispatcher / mock / declarations / audio_queue

## 待补（本周要做）

- 事件总线（可观测性）
- 工作空间安全 / 结构化错误
- 虚拟场景（Mock World，有 cup 和 target）
- 复合技能（pickup / present / place_at）
- 可视化（demo 的胜负手）
- 无麦脚本模式（demo 兜底）
- Demo 录屏 + runbook

---

## Day 1（周一）· 基线校准 + 事件总线

**目标**：E2E 语音链路跑通；任何状态变化都有结构化事件。

### 任务
- [ ] 烟测 `voice-arm --arm mock`：说 "go home" / "move right 5" / "stop"
- [ ] 在 `arm/` 新增 `events.py`：定义 `ArmEvent` dataclass + `ArmEventBus`（封装 `asyncio.Queue`）
- [ ] MockArm 每次状态迁移 publish 事件
- [ ] 事件字段：`timestamp / action / args / before_state / after_state / ok / error`
- [ ] 加测：mock 动一次 → bus 收到 1 个事件

### 出口条件
- 能用嘴控制 Mock 动
- log 里能看到完整事件流
- `pytest` 全绿

---

## Day 2（周二）· 工作空间安全 + 结构化错误

**目标**：LLM 不能把机械臂"说"到物理不可能的位置；拒绝时用语音解释。

### 任务
- [ ] 在 `ToolDispatcher` 入口做 bounds 检查（引用 `WORKSPACE_*_MM` / `MAX_RELATIVE_STEP_MM`）
- [ ] 越界 → 返回 `ToolResult(ok=False, error="…")`，**不抛异常**
- [ ] 系统 prompt 补一条：收到 error 要用自然语言解释给用户
- [ ] 加测：越界拒绝 / 相对步长限幅 / stop 幂等

### 出口条件
- 对着麦说 "move to x 10 meters"，机械臂不动
- 语音回复："超出范围，最大 300 毫米"

---

## Day 3（周三）· 虚拟场景（Mock World）

**目标**：让 demo 有叙事——桌上有杯子，抓到了就吸附到 gripper。

### 任务
- [ ] 新建 `src/voice_arm/world/scene.py`
- [ ] 定义 `VirtualObject(name, x, y, z, graspable)` + `MockScene`
- [ ] MockArm 持有一个 `scene`：`grasp()` 时若夹爪在物体 ε 半径内 → attach；`release()` 时 detach 落到桌面
- [ ] `ArmState` 或 `get_state()` 输出增加 `holding: str | None`
- [ ] 预置场景：cup @ (150, 100, 0)，target zone @ (-150, 100, 0)
- [ ] 加测：抓取吸附 / 释放落回 / 空抓不报错

### 出口条件
- `grasp` 后 `get_state` 能看到 `holding: "cup"`
- cup 位置跟着夹爪移动

---

## Day 4（周四）· 复合技能（Skills）

**目标**：语音"帮我把杯子递过来"能一次成功。

### 任务
- [ ] 新建 `src/voice_arm/skills/`，base 抽象类（参考 robotics-arm 的 skills/base.py）
- [ ] 实现 `pickup(object_name)`：above → down → grasp → up
- [ ] 实现 `present()`：移到 offer 位姿 + 保持 2 秒
- [ ] 实现 `place_at(x, y)`：目标上方 → down → release → up
- [ ] 在 `declarations.py` 注册为新的 FunctionDeclaration
- [ ] 技能运行期间 `stop` 能立即打断（靠 `asyncio.CancelledError`）
- [ ] 加测：三个技能成功路径 + stop 打断

### 出口条件
- 一句 "pick up the cup and hand it to me" → 整套动作完成
- 动作进行中说 "stop" 立即停

---

## Day 5（周五）· 可视化（demo 胜负手）

**目标**：观众能**看见**机械臂动，不只是听日志。

### 任务
- [ ] 选型（二选一）：
  - 方案 A：matplotlib 动画（最低门槛，pip 装一下）
  - 方案 B：rich 终端 dashboard（更酷，零 GUI 依赖）
- [ ] 订阅 Day 1 的 `ArmEventBus`，每次事件重绘
- [ ] 顶视图：工作空间边框 + 机械臂 XY + cup XY + target zone
- [ ] 信息栏：Z 高度 / gripper 状态 / holding / speed / 最近一条指令
- [ ] 独立线程或进程跑，不阻塞 Gemini async loop

### 出口条件
- demo 时一屏可视化 + 一屏日志，声画同步
- 可视化窗口崩了不影响主链路

---

## Day 6（周六）· 无麦脚本模式 + E2E 测试

**目标**：不依赖麦克风也能完整跑 demo，且有自动化回归。

### 任务
- [ ] 加 `voice-arm --script demo/basic.txt` 参数
- [ ] 从脚本读行（如 `pickup cup` / `present` / `place_at -150 100`）直接进 dispatcher，绕过 Gemini
- [ ] 写 3 个 E2E 测试：
  - pickup → present → place 成功路径
  - 越界请求被拒绝
  - skill 执行中 stop 打断
- [ ] 写 `voice_arm/docs/mock_demo_runbook.md`

### 出口条件
- `pytest` 全绿
- `voice-arm --script demo/basic.txt` 无人值守跑完

---

## Day 7（周日）· Demo 录制 + README

**目标**：交付物。

### 任务
- [ ] 录屏：终端 + 可视化窗口 + 语音指令全程
- [ ] 更新 `voice_arm/README.md`：Quick Start、三个 demo 场景、架构图（ascii 或 mermaid）
- [ ] 补 `voice_arm/docs/`：architecture.md / mock_demo_runbook.md / troubleshooting.md

### 出口条件
- 一个 1-3 分钟的 demo 视频
- README 新同学照着能 5 分钟跑起来

---

## 排期风险矩阵

| 风险 | 放在第几天 | 原因 |
|---|---|---|
| Gemini Live 连接不稳 | Day 1 | 最高风险，早暴露早解决 |
| 可视化技术选型卡壳 | Day 5 | 前 4 天核心逻辑已稳，可视化塌了有日志兜底 |
| Skills 的中断语义 | Day 4 | 依赖 Day 1 事件总线 + Day 2 错误通道 |
| 麦克风/音频硬件炸了 | Day 6 | 用 `--script` 脱敏，demo 日不受环境影响 |

## 每晚兜底

每天收工前 commit 一个可运行版本——**永远有一个能跑的 voice_arm**，哪怕当天目标没完成。

## 参考

- 克隆的参考仓库：`robotics-arm-reference/robotics-arm/`
  - skills 基类：`src/robot_drink_demo/skills/base.py`
  - LeRobot 适配器（后续接真机时用）：`src/robot_drink_demo/robot/lerobot_adapter.py`
  - 语音链路拆分（VAD / ASR / TTS）：`src/robot_drink_demo/voice/`
  - 配置分层思路：`configs/*.yaml`

---

## 变更记录

- 2026-04-22 初版：7 天 Mock-only 计划
