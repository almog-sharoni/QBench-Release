#!/usr/bin/env bash
set -u

ARM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UPSTREAM_DIR="$ARM_DIR/upstream/PIMSimulator"
DEPENDENCY_DIR="$ARM_DIR/dependencies"
CONTAINER_IMAGE="$DEPENDENCY_DIR/pimsimulator-ubuntu20.04.sif"
RUN_ID=${PIMSIM_RUN_ID:-reference_validation_$(date -u +%Y%m%dT%H%M%SZ)}
RAW_DIR="$ARM_DIR/artifacts/raw/$RUN_ID"
SNAPSHOT_DIR="$ARM_DIR/artifacts/build/$RUN_ID/source"
LOCKED_COMMIT=3703d1f19c8f027360cc33a3243eb271e3bb6898
LOCKED_SOURCE_SHA256=803b018642bd0858f56ead6c716ca6fba38c42c63459b0b402b2a54ad167f392

if [[ -e "$RAW_DIR" || -e "$SNAPSHOT_DIR" ]]; then
    echo "Refusing to overwrite existing run artifacts for $RUN_ID" >&2
    exit 2
fi

mkdir -p "$RAW_DIR/build" "$RAW_DIR/configurations" "$RAW_DIR/tests" "$SNAPSHOT_DIR"

record_exit_code() {
    local path=$1
    local code=$2
    printf '%s\n' "$code" > "$path"
}

run_test() {
    local test_name=$1
    local slug=${test_name//./__}
    local test_dir="$RAW_DIR/tests/$slug"
    mkdir -p "$test_dir"
    printf '%s\n' "./sim --gtest_filter=$test_name" > "$test_dir/command.txt"
    (
        cd "$SNAPSHOT_DIR" || exit 125
        apptainer exec --userns "$CONTAINER_IMAGE" \
            ./sim "--gtest_filter=$test_name"
    ) > "$test_dir/stdout.txt" 2> "$test_dir/stderr.txt"
    local code=$?
    record_exit_code "$test_dir/exit_code.txt" "$code"
    return "$code"
}

{
    echo "run_id=$RUN_ID"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host_uname=$(uname -a)"
    echo "container_image=$CONTAINER_IMAGE"
    echo "container_sha256=$(sha256sum "$CONTAINER_IMAGE" | awk '{print $1}')"
    echo "compiler=$(apptainer exec --userns "$CONTAINER_IMAGE" g++ --version | head -1)"
    echo "python=$(apptainer exec --userns "$CONTAINER_IMAGE" python3 --version 2>&1)"
    apptainer exec --userns "$CONTAINER_IMAGE" dpkg-query -W \
        build-essential g++ gcc scons libgtest-dev googletest libc6-dev
} > "$RAW_DIR/environment.txt"

git -C "$UPSTREAM_DIR" remote -v > "$RAW_DIR/upstream_remote.txt"
git -C "$UPSTREAM_DIR" status --porcelain=v1 --untracked-files=all \
    > "$RAW_DIR/upstream_status_before.txt"
actual_commit=$(git -C "$UPSTREAM_DIR" rev-parse HEAD)
printf '%s\n' "$actual_commit" > "$RAW_DIR/upstream_commit.txt"
if [[ "$actual_commit" != "$LOCKED_COMMIT" ]]; then
    echo "Locked commit mismatch: $actual_commit" > "$RAW_DIR/overall_status.txt"
    exit 3
fi

actual_source_sha=$(git -C "$UPSTREAM_DIR" archive --format=tar HEAD | sha256sum | awk '{print $1}')
printf '%s\n' "$actual_source_sha" > "$RAW_DIR/source_tree_sha256.txt"
if [[ "$actual_source_sha" != "$LOCKED_SOURCE_SHA256" ]]; then
    echo "Locked source SHA-256 mismatch: $actual_source_sha" > "$RAW_DIR/overall_status.txt"
    exit 4
fi

if [[ -s "$RAW_DIR/upstream_status_before.txt" ]]; then
    echo "Locked upstream working tree is not clean" > "$RAW_DIR/overall_status.txt"
    exit 5
fi

(
    cd "$UPSTREAM_DIR" || exit 125
    git archive --format=tar HEAD
) | tar -xf - -C "$SNAPSHOT_DIR"

cp "$SNAPSHOT_DIR/system_hbm.ini" "$RAW_DIR/configurations/system_hbm.ini"
cp "$SNAPSHOT_DIR/system_hbm_1ch.ini" "$RAW_DIR/configurations/system_hbm_1ch.ini"
cp "$SNAPSHOT_DIR/system_hbm_64ch.ini" "$RAW_DIR/configurations/system_hbm_64ch.ini"
cp "$SNAPSHOT_DIR/ini/HBM2_samsung_2M_16B_x64.ini" \
    "$RAW_DIR/configurations/HBM2_samsung_2M_16B_x64.ini"
sha256sum "$RAW_DIR/configurations/"* > "$RAW_DIR/configurations/sha256sums.txt"

{
    echo "apptainer exec --userns $CONTAINER_IMAGE scons"
} > "$RAW_DIR/build/command.txt"
(
    cd "$SNAPSHOT_DIR" || exit 125
    apptainer exec --userns "$CONTAINER_IMAGE" scons
) > "$RAW_DIR/build/stdout.txt" 2> "$RAW_DIR/build/stderr.txt"
build_code=$?
record_exit_code "$RAW_DIR/build/exit_code.txt" "$build_code"
if [[ "$build_code" -ne 0 ]]; then
    echo "FAILED_BUILD" > "$RAW_DIR/overall_status.txt"
    exit "$build_code"
fi

printf '%s\n' "./sim --gtest_list_tests" > "$RAW_DIR/tests/list_command.txt"
(
    cd "$SNAPSHOT_DIR" || exit 125
    apptainer exec --userns "$CONTAINER_IMAGE" ./sim --gtest_list_tests
) > "$RAW_DIR/tests/list_stdout.txt" 2> "$RAW_DIR/tests/list_stderr.txt"
list_code=$?
record_exit_code "$RAW_DIR/tests/list_exit_code.txt" "$list_code"
if [[ "$list_code" -ne 0 ]]; then
    echo "FAILED_TEST_DISCOVERY" > "$RAW_DIR/overall_status.txt"
    exit "$list_code"
fi

required_tests=(
    MemBandwidthFixture.hbm_read_bandwidth
    MemBandwidthFixture.hbm_write_bandwidth
    PIMKernelFixture.gemv_tree
    PIMKernelFixture.gemv
    PIMKernelFixture.add
    PIMKernelFixture.mul
    PIMKernelFixture.relu
    PIMBenchFixture.gemv
    PIMBenchFixture.add
    PIMBenchFixture.mul
    PIMBenchFixture.relu
)

failed=0
for test_name in "${required_tests[@]}"; do
    if ! run_test "$test_name"; then
        failed=1
    fi
done

git -C "$UPSTREAM_DIR" status --porcelain=v1 --untracked-files=all \
    > "$RAW_DIR/upstream_status_after.txt"
printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$RAW_DIR/environment.txt"

if [[ "$failed" -ne 0 ]]; then
    echo "FAILED_REFERENCE_TESTS" > "$RAW_DIR/overall_status.txt"
    exit 1
fi
if [[ -s "$RAW_DIR/upstream_status_after.txt" ]]; then
    echo "FAILED_UPSTREAM_TREE_CHANGED" > "$RAW_DIR/overall_status.txt"
    exit 6
fi

echo "PASSED" > "$RAW_DIR/overall_status.txt"
echo "Reference validation passed; raw artifacts: $RAW_DIR"
