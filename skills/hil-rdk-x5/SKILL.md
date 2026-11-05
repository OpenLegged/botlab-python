---
name: hil-rdk-x5
description: RDK X5（地瓜机器人 / D-Robotics，Sunrise OS）HIL 板卡部署补充——直连默认组网 192.168.127.x、系统 Python 3.10 需独立 CPython 3.12、时钟漂移与远端 pkill 自匹配等实测坑
metadata:
  hardware: D-Robotics RDK X5（RDK X5 / RDK X5 Module）
  os: Sunrise OS（Ubuntu 22.04 aarch64 基座）
  tested: botlab-python feat/control-protocol-core@d9b3bc1（0.3.0）+ websockets 17.3；2026-10-25 首次部署，10-26 升级，11-05 复验
---

# RDK X5 HIL 部署补充

本 skill 只写 **RDK X5 特有**的事实。通用流程（网关/agent 命令、令牌规则、SSH 隧道、
离线装骨架、验证标准）一律以仓库 [`docs/hil-assistant.md`](../../docs/hil-assistant.md)
第 4 步为准，不在此重复。

## 识别

```bash
cat /proc/device-tree/model                      # RDK X5 家族
. /etc/os-release && echo "$PRETTY_NAME"         # Sunrise OS（Ubuntu 22.04 基座，aarch64）
```

出厂默认账户 `sunrise`（密码 `sunrise`，D-Robotics 公开默认值，长期使用建议修改），
sudo 免密。

## 组网（网线直连）

- 板卡直连时默认静态 IP `192.168.127.10`（出厂默认，实测值；被改过则以实际为准），
  主机侧配同段静态 IP（示例 `192.168.127.100/24`）：
  - macOS：`sudo networksetup -setmanual "<网口服务名>" 192.168.127.100 255.255.255.0`
  - Linux：`sudo ip addr add 192.168.127.100/24 dev <网口>`
- 连通判据：`ping -c 2 192.168.127.10` 通、`ssh sunrise@192.168.127.10` 免密可达。
- 接路由器/局域网（板卡 DHCP）同样适用：把各命令里的地址换成实际 IP 即可，
  协议不依赖具体网段（`ws://` 只走板卡回环 + SSH 隧道）。

## 与通用流程的差异 / 实测坑

1. **系统 Python 只有 3.10 且缺 `python3-venv`**——不满足 `websockets>=17.1` 的
   ≥3.11 门槛，直接走 runbook §4 的独立 CPython 路线
   （python-build-standalone `cpython-3.12.*-aarch64-install_only.tar.gz`，自带
   ensurepip）。不要 apt 升级系统 Python，也不要降级 websockets（拨号代码传
   `legacy=True`，16.x 直接 TypeError）。
2. **系统时钟可能停在 2000 年**（板卡无外网时不自动对时）。装包/校验前先对时：
   主机 `date +%s` 取本机 epoch → 板卡 `sudo date -s @<该值>`。
3. **经 ssh 远程执行 `pkill -f botlab-…` 会匹配到承载命令的 bash 自身**，连接以
   255 退出。模式里加字符类规避：`pkill -f "botlab-[h]il-gateway"`。
4. wheel 依赖标签按 runbook §4 的 `pip download` 命令原样适用（aarch64 manylinux，
   实测装出 websockets 17.3 + onnxruntime，`[target]` extras 可用）。

## 实测环境

- RDK X5（Sunrise OS，Ubuntu 22.04 aarch64），USB 网卡直连 192.168.127.x
- botlab-python `feat/control-protocol-core@d9b3bc1`（0.3.0）、websockets 17.3
- python-build-standalone CPython 3.12 → `~/botlab-venv`，wheel 仓 `~/botlab-pkgs`
- 网关/agent nohup 常驻（日志 `~/botlab-gateway.log` / `~/botlab-agent.log`）；
  板卡重启后**不会自启**，按 runbook §4 带 env 令牌重新拉起
