# ArduPilot SITL + WSL Quickstart

This is a minimal copy/paste flow for running ArduPilot SITL in Ubuntu (WSL) and connecting from Windows.

## 1) Ubuntu (WSL): ArduPilot SITL on `udp:127.0.0.1:14550`

```bash
# In WSL Ubuntu
sudo apt update
sudo apt install -y git python3 python3-pip python3-venv

git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive

Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile

# Launch Copter SITL and stream MAVLink to UDP 14550
Tools/autotest/sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14550
```

## 2) Windows: Run bridge (`--model auto`)

```powershell
# In Windows PowerShell from D:\drone_thesis_clean
cd D:\drone_thesis_clean

# If needed: .venv\Scripts\activate
.\.venv\Scripts\python.exe -m scripts.ardupilot_bridge --model auto --connection udp:127.0.0.1:14550
```

## 3) Windows: QGC via GitHub CLI

```powershell
# In Windows PowerShell
gh auth login
gh auth status
gh release download --repo mavlink/qgroundcontrol --pattern "QGroundControl-installer.exe" --latest
.\QGroundControl-installer.exe
```

## Troubleshooting

### Missing prereqs

```bash
cd ~/ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
hash -r
```

### `MAVProxy` not found

```bash
python3 -m pip install --user MAVProxy
export PATH="$HOME/.local/bin:$PATH"
hash -r
```

### Map not showing

```bash
# Ensure you started with --map
Tools/autotest/sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14550

# If GUI forwarding is unavailable in WSL, run without map:
Tools/autotest/sim_vehicle.py -v ArduCopter --console --out=udp:127.0.0.1:14550
```

### Port conflicts (`14550` already in use)

```bash
# Check listener
ss -lunp | rg 14550 || true

# Or switch port and match it in Windows bridge
Tools/autotest/sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14551
```

```powershell
.\.venv\Scripts\python.exe -m scripts.ardupilot_bridge --model auto --connection udp:127.0.0.1:14551
```

### Reset to spawn (land/disarm + quick restart)

```text
# In MAVProxy console (same terminal where sim_vehicle.py is running)
mode land
arm disarm
```

```bash
# In WSL: quick SITL restart
pkill -f "sim_vehicle.py -v ArduCopter" || true
sleep 2
cd ~/ardupilot
Tools/autotest/sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14550
```

```bash
# Optional hard reset to default spawn/params (wipes persisted SITL state)
cd ~/ardupilot
Tools/autotest/sim_vehicle.py -w -v ArduCopter --console --map --out=udp:127.0.0.1:14550
```
