#!/usr/bin/env bash
set -u

ARM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$ARM_DIR/../../.." && pwd)
SOURCE_REPO="$ARM_DIR/source/PIMSimulator"
CONTAINER_IMAGE="$PROJECT_ROOT/runspace/experiments/pimsimulator_feasibility/dependencies/pimsimulator-ubuntu20.04.sif"
LOCKED_COMMIT=3703d1f19c8f027360cc33a3243eb271e3bb6898
LOCKED_SOURCE_SHA256=803b018642bd0858f56ead6c716ca6fba38c42c63459b0b402b2a54ad167f392
LOCKED_CONTAINER_SHA256=5e9455a5c0cfdf9e68c818473f66498bc1b7e911850f85e5b30a40091ec12db4
RUN_ID=${PIMSIM_DIAG_RUN_ID:-add_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$ARM_DIR/runs/$RUN_ID"
BUILD_DIR="$RUN_DIR/build/source"
RAW_DIR="$RUN_DIR/raw"

if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing diagnostic run: $RUN_DIR" >&2
    exit 2
fi

mkdir -p "$BUILD_DIR" "$RAW_DIR/build" "$RAW_DIR/configurations" \
    "$RAW_DIR/functional_add" "$RAW_DIR/performance_repeats" "$RAW_DIR/full_suite"

record_exit_code() {
    printf '%s\n' "$2" > "$1"
}

run_in_container() {
    apptainer exec --userns "$CONTAINER_IMAGE" "$@"
}

actual_commit=$(git -C "$SOURCE_REPO" rev-parse HEAD)
printf '%s\n' "$actual_commit" > "$RAW_DIR/source_commit.txt"
git -C "$SOURCE_REPO" remote -v > "$RAW_DIR/source_remotes.txt"
git -C "$SOURCE_REPO" status --porcelain=v1 --untracked-files=all > "$RAW_DIR/source_status.txt"
if [[ "$actual_commit" != "$LOCKED_COMMIT" ]]; then
    echo "LOCKED_COMMIT_MISMATCH" > "$RAW_DIR/overall_status.txt"
    exit 3
fi
if [[ -s "$RAW_DIR/source_status.txt" ]]; then
    echo "SOURCE_CHECKOUT_NOT_CLEAN" > "$RAW_DIR/overall_status.txt"
    exit 4
fi

actual_source_sha=$(git -C "$SOURCE_REPO" archive --format=tar HEAD | sha256sum | awk '{print $1}')
printf '%s\n' "$actual_source_sha" > "$RAW_DIR/source_tree_sha256.txt"
if [[ "$actual_source_sha" != "$LOCKED_SOURCE_SHA256" ]]; then
    echo "LOCKED_SOURCE_HASH_MISMATCH" > "$RAW_DIR/overall_status.txt"
    exit 5
fi

actual_container_sha=$(sha256sum "$CONTAINER_IMAGE" | awk '{print $1}')
printf '%s\n' "$actual_container_sha" > "$RAW_DIR/container_sha256.txt"
if [[ "$actual_container_sha" != "$LOCKED_CONTAINER_SHA256" ]]; then
    echo "LOCKED_CONTAINER_HASH_MISMATCH" > "$RAW_DIR/overall_status.txt"
    exit 6
fi

(
    cd "$SOURCE_REPO" || exit 125
    git archive --format=tar HEAD
) | tar -xf - -C "$BUILD_DIR"

cp "$BUILD_DIR/system_hbm.ini" "$RAW_DIR/configurations/"
cp "$BUILD_DIR/system_hbm_1ch.ini" "$RAW_DIR/configurations/"
cp "$BUILD_DIR/system_hbm_64ch.ini" "$RAW_DIR/configurations/"
cp "$BUILD_DIR/ini/HBM2_samsung_2M_16B_x64.ini" "$RAW_DIR/configurations/"
sha256sum "$RAW_DIR/configurations/"* > "$RAW_DIR/configurations/sha256sums.txt"

{
    echo "run_id=$RUN_ID"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host_uname=$(uname -a)"
    echo "source_commit=$actual_commit"
    echo "source_tree_sha256=$actual_source_sha"
    echo "container_image=$CONTAINER_IMAGE"
    echo "container_sha256=$actual_container_sha"
    echo "compiler=$(run_in_container g++ --version 2>/dev/null | head -1)"
    echo "scons=$(run_in_container scons --version 2>/dev/null | head -1)"
    run_in_container dpkg-query -W build-essential g++ gcc googletest libgtest-dev libc6-dev scons
} > "$RAW_DIR/environment.txt" 2> "$RAW_DIR/environment_stderr.txt"

printf '%s\n' "apptainer exec --userns $CONTAINER_IMAGE scons" > "$RAW_DIR/build/command.txt"
(
    cd "$BUILD_DIR" || exit 125
    run_in_container scons
) > "$RAW_DIR/build/stdout.txt" 2> "$RAW_DIR/build/stderr.txt"
build_exit=$?
record_exit_code "$RAW_DIR/build/exit_code.txt" "$build_exit"
if [[ "$build_exit" -ne 0 ]]; then
    echo "FAILED_BUILD" > "$RAW_DIR/overall_status.txt"
    exit "$build_exit"
fi

sha256sum "$BUILD_DIR/sim" > "$RAW_DIR/build/sim_sha256.txt"
grep -E '(^| )g\+\+ .* (-c|-o sim )' "$RAW_DIR/build/stdout.txt" \
    > "$RAW_DIR/build/compiler_commands.txt" || true

printf '%s\n' "./sim --gtest_filter=PIMKernelFixture.add" \
    > "$RAW_DIR/functional_add/command.txt"
(
    cd "$BUILD_DIR" || exit 125
    run_in_container ./sim --gtest_filter=PIMKernelFixture.add
) > "$RAW_DIR/functional_add/stdout.txt" 2> "$RAW_DIR/functional_add/stderr.txt"
functional_exit=$?
record_exit_code "$RAW_DIR/functional_add/exit_code.txt" "$functional_exit"
sha256sum "$RAW_DIR/functional_add/stdout.txt" "$RAW_DIR/functional_add/stderr.txt" \
    > "$RAW_DIR/functional_add/output_sha256s.txt"
if [[ "$functional_exit" -ne 0 ]]; then
    echo "FUNCTIONAL_ADD_FAILED" > "$RAW_DIR/overall_status.txt"
    exit "$functional_exit"
fi

printf '%s\n' 'repeat,non_pim_cycles,pim_cycles,speedup,exit_code,stdout_sha256,stderr_sha256,binary_sha256' \
    > "$RAW_DIR/performance_repeats/repeated_run_results.csv"
repeat_failure=0
for repeat in 1 2 3 4 5; do
    repeat_dir="$RAW_DIR/performance_repeats/repeat_$repeat"
    mkdir -p "$repeat_dir"
    printf '%s\n' "./sim --gtest_filter=PIMBenchFixture.add" > "$repeat_dir/command.txt"
    (
        cd "$BUILD_DIR" || exit 125
        run_in_container ./sim --gtest_filter=PIMBenchFixture.add
    ) > "$repeat_dir/stdout.txt" 2> "$repeat_dir/stderr.txt"
    repeat_exit=$?
    record_exit_code "$repeat_dir/exit_code.txt" "$repeat_exit"
    non_pim_cycles=$(awk '/> Cycle :/{n++; if (n == 1) print $4}' "$repeat_dir/stdout.txt")
    pim_cycles=$(awk '/> Cycle :/{n++; if (n == 2) print $4}' "$repeat_dir/stdout.txt")
    speedup=$(awk '/> Speed-up :/{print $4}' "$repeat_dir/stdout.txt")
    stdout_sha=$(sha256sum "$repeat_dir/stdout.txt" | awk '{print $1}')
    stderr_sha=$(sha256sum "$repeat_dir/stderr.txt" | awk '{print $1}')
    binary_sha=$(sha256sum "$BUILD_DIR/sim" | awk '{print $1}')
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$repeat" "$non_pim_cycles" "$pim_cycles" \
        "$speedup" "$repeat_exit" "$stdout_sha" "$stderr_sha" "$binary_sha" \
        >> "$RAW_DIR/performance_repeats/repeated_run_results.csv"
    if [[ "$repeat_exit" -ne 1 ]]; then
        repeat_failure=1
    fi
done

printf '%s\n' './sim' > "$RAW_DIR/full_suite/command.txt"
(
    cd "$BUILD_DIR" || exit 125
    run_in_container ./sim
) > "$RAW_DIR/full_suite/stdout.txt" 2> "$RAW_DIR/full_suite/stderr.txt"
suite_exit=$?
record_exit_code "$RAW_DIR/full_suite/exit_code.txt" "$suite_exit"
sha256sum "$RAW_DIR/full_suite/stdout.txt" "$RAW_DIR/full_suite/stderr.txt" \
    > "$RAW_DIR/full_suite/output_sha256s.txt"

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RAW_DIR/environment.txt"
(
    cd "$RUN_DIR" || exit 125
    find raw -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$RUN_DIR/raw_sha256sums.txt"

if [[ "$repeat_failure" -ne 0 ]]; then
    echo "UNEXPECTED_REPEAT_EXIT_CODE" > "$RAW_DIR/overall_status.txt"
    exit 7
fi

# The expected upstream performance assertion failure gives suite/repeat exit 1.
printf 'DIAGNOSTIC_CAPTURE_COMPLETE functional_exit=%s suite_exit=%s\n' \
    "$functional_exit" "$suite_exit" > "$RAW_DIR/overall_status.txt"
echo "Diagnostic capture complete: $RUN_DIR"
