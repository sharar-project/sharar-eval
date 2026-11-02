#!/bin/bash
# SSD reads (MiB/s), GPU use (%), SM clock (MHz) and MemAvailable (GiB) every 2 s, to see whether decoding reads from
# the SSD (an engine whose n-gram table is memory-mapped slows down when it does). Usage:
#   [DISK=nvme0n1] dlog.sh OUT_FILE SECONDS [STOP_FILE]
# It ends after SECONDS or once STOP_FILE exists. DISK: the disk holding the checkpoint (default: the first disk).
# Columns: time, read_MiBs, gpu, sm_mhz, avail_GiB (fields 3, 5, 7 and 9).
d=${DISK:-$(lsblk -ndo NAME,TYPE | awk '$2=="disk"{print $1; exit}')}; out=$1; end=$(( $(date +%s) + $2 )); stop=${3:-}
a=$(awk -v d=$d '$3==d{print $6}' /proc/diskstats)
while [ $(date +%s) -lt $end ] && { [ -z "$stop" ] || [ ! -e "$stop" ]; }; do
  sleep 2; b=$(awk -v d=$d '$3==d{print $6}' /proc/diskstats)
  g=$(nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader,nounits | tr -d ' ')
  echo "$(date -u +%H:%M:%S) read_MiBs $(( (b-a)*512/2/1048576 )) gpu ${g%,*} sm_mhz ${g#*,} avail_GiB $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" >> "$out"
  a=$b
done
