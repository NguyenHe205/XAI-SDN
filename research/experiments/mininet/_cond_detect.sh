#!/usr/bin/env bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONUNBUFFERED=1
cd /opt/xaisdn
rm -f /opt/xaisdn/ctl_detect.json /opt/xaisdn/mn_detect.json /opt/xaisdn/ctl_detect.log
mn -c >/dev/null 2>&1
pkill -f run_controller.py >/dev/null 2>&1
sleep 1
XAISDN_DETECT=1 XAISDN_EXPLAIN=0 XAISDN_OUT=/opt/xaisdn/ctl_detect.json nohup /opt/xaisdn/bin/python /opt/xaisdn/run_controller.py /opt/xaisdn/xaisdn_controller.py 6653 > /opt/xaisdn/ctl_detect.log 2>&1 &
CTL=$!
for i in $(seq 1 25); do
  if grep -q "listening on" /opt/xaisdn/ctl_detect.log 2>/dev/null; then break; fi
  sleep 1
done
echo "CONTROLLER_PID=$CTL"
echo "CONTROLLER_READY=$(grep -c 'listening on' /opt/xaisdn/ctl_detect.log 2>/dev/null || echo 0)"
python3 run_topology.py --mode detect --out /opt/xaisdn/mn_detect.json --attack-seconds 20
echo "TOPOLOGY_RC=$?"
kill -TERM $CTL 2>/dev/null
sleep 3
pkill -f run_controller.py >/dev/null 2>&1
mn -c >/dev/null 2>&1
