"""Canned product copy shared by the CLI client and project middleware.

Single source of truth for the agent handoff prompt, so `botlab control
connection` and a middleware's browser-facing endpoint serve byte-identical
text. Standard-library only, like the protocol core, so third-party
middleware can import it freely. The pairing token is deliberately absent:
it is surfaced separately by `connection --show-token` and must never be
embedded in prompts, command lines or logs.
"""


def agent_prompt(project: str) -> str:
    return (
        "你通过 BotLab Control CLI 操控一个正在运行的浏览器仿真。botlab-python 已安装、项目中间件正在运行，"
        "无需安装任何东西；浏览器可能尚未配对，先按下面流程核实再行动。\n"
        "\n"
        f"工作目录：{project}\n"
        "在其他目录调用时，为每条命令加 --project 指向该目录。\n"
        "\n"
        "第一步，核实链路：\n"
        "botlab control --project . state --json\n"
        "- connected 与 external_enabled 均为 true、run_state 为 running、ready 为 true 时，链路已就绪。\n"
        "- 若浏览器尚未配对（connected 为 false 或 external_enabled 为 false），执行：\n"
        "  botlab control --project . connection --show-token --json\n"
        "  把返回的 websocket 地址与 token 转告用户（仅在此对话中回复，绝不写入文件、日志或提交），\n"
        "  请用户粘贴到浏览器「CLI 控制」面板并点击「连接并启用外部控制」，然后重新核实链路。\n"
        "\n"
        "控制规则：\n"
        "1. 每个动作：读 state → 发命令 → wait --id 直到 terminal:true → 核对物理反馈，再发下一条。\n"
        "2. 工具输出解析失败不等于命令未执行：先查 state 的 active_command_id，确认未接受后才可重发。\n"
        "3. 参数按 capabilities 声明的范围发送；定时移动不是精确导航，实际效果以位置与姿态反馈为准。\n"
        "4. 网络超时只表示结果未知：先 wait 原请求，必要时对同一 request-id 显式 retry；新的运动决定用新 ID。\n"
        "5. 页面隐藏或仿真暂停时 ready=false，动作会被拒绝；恢复后从当前位置继续，旧动作不会自动重放。\n"
        "6. 项目若有 CONTROL_EXPERIENCE.md 或 AGENTS.md 等控制经验/说明文件，先读再动。\n"
        "\n"
        "现在：读取状态并汇报当前姿态（若未配对则先引导配对），然后等待用户指令。"
    )
