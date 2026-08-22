#!/usr/bin/env bash
set -euo pipefail

min_available_gib=64
interval_seconds=60
stable_checks=3
lock_file="/tmp/mmdd-wait-for-memory.lock"

usage() {
    cat <<'EOF'
Usage:
  wait_for_memory_then_run.sh [options] -- COMMAND [ARG ...]

Wait until Linux MemAvailable stays above a threshold, then replace the
monitor with COMMAND.

Options:
  --min-available-gib N   Required MemAvailable in GiB (default: 64)
  --interval-seconds N    Seconds between checks (default: 60)
  --stable-checks N       Consecutive passing checks required (default: 3)
  --lock-file PATH        Prevent duplicate monitors (default:
                          /tmp/mmdd-wait-for-memory.lock)
  -h, --help              Show this help
EOF
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

while (($#)); do
    case "$1" in
        --min-available-gib)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            min_available_gib="$2"
            shift 2
            ;;
        --interval-seconds)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            interval_seconds="$2"
            shift 2
            ;;
        --stable-checks)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            stable_checks="$2"
            shift 2
            ;;
        --lock-file)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            lock_file="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

is_positive_integer "$min_available_gib" || {
    echo "--min-available-gib must be a positive integer" >&2
    exit 2
}
is_positive_integer "$interval_seconds" || {
    echo "--interval-seconds must be a positive integer" >&2
    exit 2
}
is_positive_integer "$stable_checks" || {
    echo "--stable-checks must be a positive integer" >&2
    exit 2
}
(($# > 0)) || {
    echo "A command is required after --" >&2
    usage >&2
    exit 2
}

command -v flock >/dev/null 2>&1 || {
    echo "flock is required" >&2
    exit 1
}
[[ -r /proc/meminfo ]] || {
    echo "/proc/meminfo is not readable" >&2
    exit 1
}

exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another memory monitor already holds $lock_file" >&2
    exit 1
fi

read_memory_kib() {
    local key value unit
    mem_available_kib=0
    swap_free_kib=0
    while read -r key value unit; do
        case "$key" in
            MemAvailable:)
                mem_available_kib="$value"
                ;;
            SwapFree:)
                swap_free_kib="$value"
                ;;
        esac
    done < /proc/meminfo
    ((mem_available_kib > 0)) || {
        echo "MemAvailable is missing from /proc/meminfo" >&2
        return 1
    }
}

format_gib() {
    awk -v kib="$1" 'BEGIN { printf "%.1f", kib / 1048576 }'
}

threshold_kib=$((min_available_gib * 1024 * 1024))
passing_checks=0

printf '[%s] Waiting for MemAvailable >= %s GiB for %s consecutive checks; interval=%ss\n' \
    "$(date -Is)" "$min_available_gib" "$stable_checks" "$interval_seconds"
printf '[%s] Command:' "$(date -Is)"
printf ' %q' "$@"
printf '\n'

while true; do
    read_memory_kib
    if ((mem_available_kib >= threshold_kib)); then
        ((passing_checks += 1))
    else
        passing_checks=0
    fi

    printf '[%s] MemAvailable=%s GiB, SwapFree=%s GiB, ready=%s/%s\n' \
        "$(date -Is)" \
        "$(format_gib "$mem_available_kib")" \
        "$(format_gib "$swap_free_kib")" \
        "$passing_checks" \
        "$stable_checks"

    if ((passing_checks >= stable_checks)); then
        printf '[%s] Memory threshold is stable; starting command now.\n' "$(date -Is)"
        exec "$@"
    fi
    sleep "$interval_seconds"
done
