#!/usr/bin/env bash
# 一键在独立 Screen 会话中运行 MATLAB run_graph_experiment.m，并保存日志。

set -Eeuo pipefail

MATLAB_BIN='/home/ubuntu/xlj/bin/matlab'

PROJECT_DIR='/home/ubuntu/xlj/compare_methods/graph_partition'

ENTRY_FILE='run_graph_experiment.m'

LOG_DIR="${PROJECT_DIR}/graph_logs"

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"


if [[ ! -x "$MATLAB_BIN" ]]; then
    printf '错误：MATLAB 不存在或不可执行：%s\n' "$MATLAB_BIN" >&2
    exit 1
fi


if [[ ! -f "${PROJECT_DIR}/${ENTRY_FILE}" ]]; then
    printf '错误：找不到实验入口：%s\n' "${PROJECT_DIR}/${ENTRY_FILE}" >&2
    exit 1
fi


if ! command -v screen >/dev/null 2>&1; then
    printf '错误：未安装 screen。\n' >&2
    exit 1
fi


mkdir -p "$LOG_DIR"


# ===============================
# 外层：创建 screen
# ===============================

if [[ "${1:-}" != '--inside-screen' ]]; then

    stamp="$(date '+%Y%m%d_%H%M%S')"

    session="graph_exp_${stamp}_$$"

    logfile="${LOG_DIR}/${session}.log"

    statusfile="${LOG_DIR}/${session}.exitcode"


    screen -dmS "$session" \
        bash "$SCRIPT_PATH" \
        --inside-screen \
        "$session" \
        "$logfile" \
        "$statusfile"


    printf '实验已提交到 Screen 后台会话。\n'
    printf '会话名称：%s\n' "$session"
    printf '日志文件：%s\n' "$logfile"
    printf '退出码文件：%s\n' "$statusfile"

    printf '\n'
    printf '查看会话：screen -ls\n'
    printf '进入会话：screen -r %s\n' "$session"
    printf '实时日志：tail -f %s\n' "$logfile"


    exit 0
fi



# ===============================
# screen 内部执行
# ===============================

session="${2:?缺少会话名称}"
logfile="${3:?缺少日志路径}"
statusfile="${4:?缺少退出码路径}"


cd "$PROJECT_DIR"


printf '[开始] %s\n[会话] %s\n[工程] %s\n[MATLAB] %s\n[入口] %s\n' \
    "$(date '+%F %T %Z')" \
    "$session" \
    "$PROJECT_DIR" \
    "$MATLAB_BIN" \
    "$ENTRY_FILE" \
    | tee "$logfile"



set +e


"$MATLAB_BIN" -batch "${ENTRY_FILE%.m}" \
    2>&1 | tee -a "$logfile"


matlab_status=${PIPESTATUS[0]}


set -e


printf '%s\n' "$matlab_status" > "$statusfile"


printf '[结束] %s | MATLAB退出码：%s\n' \
    "$(date '+%F %T %Z')" \
    "$matlab_status" \
    | tee -a "$logfile"


exit "$matlab_status"