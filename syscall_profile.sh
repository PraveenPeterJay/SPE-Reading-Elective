#!/usr/bin/env bash
# syscall_profile.sh
# Rendered by Ansible template — do not edit directly.
# Usage: bash syscall_profile.sh <lane> <port> <container> <out_dir>

set -uo pipefail

LANE="$1"
PORT="$2"
CONTAINER="$3"
OUT_DIR="$4"

CID=$(docker ps -qf "name=${CONTAINER}" 2>/dev/null || true)

if [ -z "$CID" ]; then
  echo "WARN: container ${CONTAINER} not found, skipping syscall profile"
  echo "{\"lane\":\"${LANE}\",\"unique_syscalls\":\"N/A\",\"syscall_list\":\"container_not_found\"}" \
    > "${OUT_DIR}/syscall-${LANE}.json"
  exit 0
fi

# Collect proc metadata
docker exec "$CID" sh -c \
  'cat /proc/1/status' \
  > "${OUT_DIR}/proc-info-${LANE}.txt" 2>/dev/null || true

# Get PID of container's init process
PID=$(docker inspect --format '{{.State.Pid}}' "$CID" 2>/dev/null || echo "")

if [ -z "$PID" ] || [ "$PID" = "0" ]; then
  echo "WARN: could not get PID for ${CONTAINER}, skipping strace"
  SYSCALLS="N/A"
  SYSCALL_LIST="pid_unavailable"
else
  # Warm the inference endpoint (5 requests, ignore failures)
  for i in $(seq 1 5); do
    curl -s -X POST "http://localhost:${PORT}/predict" \
      -H "Content-Type: application/json" \
      -d '{"features":[6,148,72,35,169,33.6,0.627,50]}' > /dev/null 2>&1 || true
  done

  STRACE_FILE="${OUT_DIR}/strace-${LANE}.txt"

  # Trace for 8 seconds while driving inference load
  timeout 8 nsenter -t "$PID" -m -u -i -n -p -- \
    strace -p 1 -f -e trace=all -o "${STRACE_FILE}" 2>/dev/null &
  STRACE_PID=$!

  for i in $(seq 1 20); do
    curl -s -X POST "http://localhost:${PORT}/predict" \
      -H "Content-Type: application/json" \
      -d '{"features":[6,148,72,35,169,33.6,0.627,50]}' > /dev/null 2>&1 || true
  done

  wait "$STRACE_PID" 2>/dev/null || true

  if [ -f "$STRACE_FILE" ] && [ -s "$STRACE_FILE" ]; then
    SYSCALLS=$(grep -oP '^[a-z_0-9]+(?=\()' "$STRACE_FILE" | sort -u | wc -l)
    SYSCALL_LIST=$(grep -oP '^[a-z_0-9]+(?=\()' "$STRACE_FILE" | sort -u | tr '\n' ',' | sed 's/,$//')
  else
    SYSCALLS="N/A"
    SYSCALL_LIST="strace_empty"
  fi
fi

echo "{\"lane\":\"${LANE}\",\"unique_syscalls\":\"${SYSCALLS}\",\"syscall_list\":\"${SYSCALL_LIST}\"}" \
  > "${OUT_DIR}/syscall-${LANE}.json"

echo "Syscall surface for ${LANE}: ${SYSCALLS} unique syscalls"