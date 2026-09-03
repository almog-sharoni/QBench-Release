#!/usr/bin/env bash
set -u

ARM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$ARM_DIR/../../.." && pwd)
PHASE2_DIR="$PROJECT_ROOT/runspace/experiments/pimsimulator_feasibility"
DIAGNOSTIC_DIR="$PROJECT_ROOT/runspace/experiments/pimsimulator_add_diagnostic"
SOURCE_REPO="$DIAGNOSTIC_DIR/source/PIMSimulator"
CONTAINER_IMAGE="$PHASE2_DIR/dependencies/pimsimulator-ubuntu20.04.sif"
CASE_MATRIX="$ARM_DIR/case_matrix.csv"
LOCKED_COMMIT=3703d1f19c8f027360cc33a3243eb271e3bb6898
LOCKED_SOURCE_SHA256=803b018642bd0858f56ead6c716ca6fba38c42c63459b0b402b2a54ad167f392
LOCKED_CONTAINER_SHA256=5e9455a5c0cfdf9e68c818473f66498bc1b7e911850f85e5b30a40091ec12db4
LOCKED_PHASE2_BUNDLE_SHA256=8f0709f39267e79414cf9b7c9b2bde675b42531ffa45d90c0c71b730ca50f97b
LOCKED_DIAGNOSTIC_BUNDLE_SHA256=6d64351133f952431d478162f644b0ddab6b5a3fb3fca40190d0fb5dd7fc8e96
LOCKED_SHAPE_SOURCE_SHA256=6a6a4bec2b90a791b92023382a5fbb45354a0c0e629e6c84164ee44b809fae2d
LOCKED_QUALITY_JSON_SHA256=942093102afd8a850bfd790f41adc6c6617486379518dd5ee00e4ae056102501
LOCKED_DATABASE_SHA256=401a06b517765a9811a4f489ba9b7dde58a821dc40bb100d94ca27ff8660ca2b
RUN_ID=${PIMSIM_PHASE3_RUN_ID:-phase3_preliminary_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$ARM_DIR/raw/$RUN_ID"
BUILD_SOURCE="$RUN_DIR/build/source"
TRACE_SOURCE="$RUN_DIR/trace_source"
RAW_RESULTS="$RUN_DIR/results"

if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing Phase-3 run: $RUN_DIR" >&2
    exit 2
fi

mkdir -p "$BUILD_SOURCE" "$RUN_DIR/build" "$RUN_DIR/environment" \
    "$RUN_DIR/reference_checks/functional" "$RUN_DIR/reference_checks/add_performance" \
    "$RUN_DIR/configurations" "$RAW_RESULTS" "$RUN_DIR/traces"

bundle_hash() {
    local bundle_dir=$1
    (
        cd "$bundle_dir" || exit 125
        find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
    ) | sha256sum | awk '{print $1}'
}

record_exit() {
    printf '%s\n' "$2" > "$1"
}

run_container() {
    local container_pwd=$1
    shift
    apptainer exec --userns --bind "$ARM_DIR:/phase3" --pwd "$container_pwd" \
        "$CONTAINER_IMAGE" "$@"
}

phase2_hash=$(bundle_hash "$PHASE2_DIR")
diagnostic_hash=$(bundle_hash "$DIAGNOSTIC_DIR")
printf '%s\n' "$phase2_hash" > "$RUN_DIR/environment/phase2_bundle_sha256.txt"
printf '%s\n' "$diagnostic_hash" > "$RUN_DIR/environment/diagnostic_bundle_sha256.txt"
if [[ "$phase2_hash" != "$LOCKED_PHASE2_BUNDLE_SHA256" ]]; then
    echo "PHASE2_BUNDLE_HASH_MISMATCH" > "$RUN_DIR/status.txt"
    exit 3
fi
if [[ "$diagnostic_hash" != "$LOCKED_DIAGNOSTIC_BUNDLE_SHA256" ]]; then
    echo "DIAGNOSTIC_BUNDLE_HASH_MISMATCH" > "$RUN_DIR/status.txt"
    exit 4
fi

actual_commit=$(git -C "$SOURCE_REPO" rev-parse HEAD)
actual_source_sha=$(git -C "$SOURCE_REPO" archive --format=tar HEAD | sha256sum | awk '{print $1}')
actual_container_sha=$(sha256sum "$CONTAINER_IMAGE" | awk '{print $1}')
printf '%s\n' "$actual_commit" > "$RUN_DIR/environment/source_commit.txt"
printf '%s\n' "$actual_source_sha" > "$RUN_DIR/environment/source_archive_sha256.txt"
printf '%s\n' "$actual_container_sha" > "$RUN_DIR/environment/container_sha256.txt"
git -C "$SOURCE_REPO" status --porcelain=v1 --untracked-files=all \
    > "$RUN_DIR/environment/source_status.txt"
if [[ "$actual_commit" != "$LOCKED_COMMIT" || "$actual_source_sha" != "$LOCKED_SOURCE_SHA256" ]]; then
    echo "SOURCE_LOCK_MISMATCH" > "$RUN_DIR/status.txt"
    exit 5
fi
if [[ -s "$RUN_DIR/environment/source_status.txt" ]]; then
    echo "SOURCE_NOT_CLEAN" > "$RUN_DIR/status.txt"
    exit 6
fi
if [[ "$actual_container_sha" != "$LOCKED_CONTAINER_SHA256" ]]; then
    echo "CONTAINER_HASH_MISMATCH" > "$RUN_DIR/status.txt"
    exit 7
fi

shape_sha=$(sha256sum "$PROJECT_ROOT/runspace/experiments/asic_cache_simulation/simulation_results_resnet50.json" | awk '{print $1}')
quality_sha=$(sha256sum "$PROJECT_ROOT/runspace/experiments/find_optimal_hybrid_quant/results/latest_db_results.json" | awk '{print $1}')
database_sha=$(sha256sum "$PROJECT_ROOT/runspace/database/runs.db" | awk '{print $1}')
printf '%s\n' "$shape_sha" > "$RUN_DIR/environment/shape_source_sha256.txt"
printf '%s\n' "$quality_sha" > "$RUN_DIR/environment/quality_json_sha256.txt"
printf '%s\n' "$database_sha" > "$RUN_DIR/environment/database_sha256.txt"
if [[ "$shape_sha" != "$LOCKED_SHAPE_SOURCE_SHA256" || \
      "$quality_sha" != "$LOCKED_QUALITY_JSON_SHA256" || \
      "$database_sha" != "$LOCKED_DATABASE_SHA256" ]]; then
    echo "PROVENANCE_INPUT_HASH_MISMATCH" > "$RUN_DIR/status.txt"
    exit 8
fi

(
    cd "$SOURCE_REPO" || exit 125
    git archive --format=tar HEAD
) | tar -xf - -C "$BUILD_SOURCE"

cp "$BUILD_SOURCE/system_hbm.ini" "$RUN_DIR/configurations/"
cp "$BUILD_SOURCE/system_hbm_1ch.ini" "$RUN_DIR/configurations/"
cp "$BUILD_SOURCE/system_hbm_64ch.ini" "$RUN_DIR/configurations/"
cp "$BUILD_SOURCE/ini/HBM2_samsung_2M_16B_x64.ini" "$RUN_DIR/configurations/"
sha256sum "$RUN_DIR/configurations/"* > "$RUN_DIR/configurations/sha256sums.txt"

{
    echo "run_id=$RUN_ID"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host_uname=$(uname -a)"
    echo "compiler=$(run_container /phase3 g++ --version 2>/dev/null | head -1)"
    echo "scons=$(run_container /phase3 scons --version 2>/dev/null | head -1)"
    run_container /phase3 dpkg-query -W build-essential g++ gcc googletest libgtest-dev libc6-dev scons
} > "$RUN_DIR/environment/toolchain.txt" 2> "$RUN_DIR/environment/toolchain_stderr.txt"

printf '%s\n' "apptainer exec --userns <locked-container> scons" > "$RUN_DIR/build/scons_command.txt"
run_container "/phase3/raw/$RUN_ID/build/source" scons \
    > "$RUN_DIR/build/scons_stdout.txt" 2> "$RUN_DIR/build/scons_stderr.txt"
build_exit=$?
record_exit "$RUN_DIR/build/scons_exit_code.txt" "$build_exit"
if [[ "$build_exit" -ne 0 ]]; then
    echo "BUILD_FAILED" > "$RUN_DIR/status.txt"
    exit "$build_exit"
fi

printf '%s\n' \
    "g++ -g -O2 -std=c++14 -Wall -Wno-reorder -Wno-sign-compare -Ilib -Isrc -Itools /phase3/adapter/phase3_runner.cpp libdramsim/libdramsim2.a -lgtest -lpthread -o /phase3/raw/$RUN_ID/build/phase3_runner" \
    > "$RUN_DIR/build/adapter_compile_command.txt"
run_container "/phase3/raw/$RUN_ID/build/source" g++ -g -O2 -std=c++14 -Wall \
    -Wno-reorder -Wno-sign-compare -Ilib -Isrc -Itools \
    /phase3/adapter/phase3_runner.cpp libdramsim/libdramsim2.a -lgtest -lpthread \
    -o "/phase3/raw/$RUN_ID/build/phase3_runner" \
    > "$RUN_DIR/build/adapter_compile_stdout.txt" \
    2> "$RUN_DIR/build/adapter_compile_stderr.txt"
adapter_build_exit=$?
record_exit "$RUN_DIR/build/adapter_compile_exit_code.txt" "$adapter_build_exit"
if [[ "$adapter_build_exit" -ne 0 ]]; then
    echo "ADAPTER_BUILD_FAILED" > "$RUN_DIR/status.txt"
    exit "$adapter_build_exit"
fi
sha256sum "$BUILD_SOURCE/sim" "$RUN_DIR/build/phase3_runner" \
    > "$RUN_DIR/build/binary_sha256sums.txt"

printf '%s\n' './sim --gtest_filter=PIMKernelFixture.*' \
    > "$RUN_DIR/reference_checks/functional/command.txt"
run_container "/phase3/raw/$RUN_ID/build/source" ./sim '--gtest_filter=PIMKernelFixture.*' \
    > "$RUN_DIR/reference_checks/functional/stdout.txt" \
    2> "$RUN_DIR/reference_checks/functional/stderr.txt"
functional_exit=$?
record_exit "$RUN_DIR/reference_checks/functional/exit_code.txt" "$functional_exit"
if [[ "$functional_exit" -ne 0 ]]; then
    echo "FUNCTIONAL_REFERENCE_FAILED" > "$RUN_DIR/status.txt"
    exit 20
fi

printf '%s\n' './sim --gtest_filter=PIMBenchFixture.add' \
    > "$RUN_DIR/reference_checks/add_performance/command.txt"
run_container "/phase3/raw/$RUN_ID/build/source" ./sim --gtest_filter=PIMBenchFixture.add \
    > "$RUN_DIR/reference_checks/add_performance/stdout.txt" \
    2> "$RUN_DIR/reference_checks/add_performance/stderr.txt"
add_exit=$?
record_exit "$RUN_DIR/reference_checks/add_performance/exit_code.txt" "$add_exit"
add_non_pim=$(awk '/> Cycle :/{n++; if(n==1) print $4}' "$RUN_DIR/reference_checks/add_performance/stdout.txt")
add_pim=$(awk '/> Cycle :/{n++; if(n==2) print $4}' "$RUN_DIR/reference_checks/add_performance/stdout.txt")
if [[ "$add_exit" -ne 1 || "$add_non_pim" != 6651 || "$add_pim" != 3349 ]]; then
    echo "PRESERVED_ADD_DEVIATION_CHANGED" > "$RUN_DIR/status.txt"
    exit 21
fi

adapter_failure=0
tail -n +2 "$CASE_MATRIX" | while IFS=, read -r split case_id kernel elements output_dim input_dim vectors trace; do
    for repeat in 1 2; do
        case_dir="$RAW_RESULTS/$split/$case_id/repeat_$repeat"
        mkdir -p "$case_dir"
        if [[ "$kernel" == "GEMV" ]]; then
            args=(--case-id "$case_id" --kernel GEMV --output "$output_dim" --input "$input_dim" --vectors "$vectors" --verify)
        else
            args=(--case-id "$case_id" --kernel "$kernel" --elements "$elements" --verify)
        fi
        printf '%q ' "/phase3/raw/$RUN_ID/build/phase3_runner" "${args[@]}" > "$case_dir/command.txt"
        printf '\n' >> "$case_dir/command.txt"
        run_container "/phase3/raw/$RUN_ID/build/source" \
            "/phase3/raw/$RUN_ID/build/phase3_runner" "${args[@]}" \
            > "$case_dir/stdout.txt" 2> "$case_dir/stderr.txt"
        case_exit=$?
        record_exit "$case_dir/exit_code.txt" "$case_exit"
        sha256sum "$case_dir/stdout.txt" "$case_dir/stderr.txt" \
            > "$case_dir/output_sha256sums.txt"
        if [[ "$case_exit" -ne 0 ]]; then
            printf '%s\n' "$case_id repeat=$repeat exit=$case_exit" \
                >> "$RUN_DIR/adapter_failures.txt"
            adapter_failure=1
        fi
    done
done

if [[ -s "$RUN_DIR/adapter_failures.txt" ]]; then
    echo "ADAPTER_GATE_FAILED" > "$RUN_DIR/status.txt"
    exit 22
fi

cp -a "$BUILD_SOURCE" "$TRACE_SOURCE"
sed -i 's/^SHOW_SIM_OUTPUT=false/SHOW_SIM_OUTPUT=true/' "$TRACE_SOURCE/system_hbm_64ch.ini"
sed -i 's/^PRINT_CHAN_STAT=true/PRINT_CHAN_STAT=false/' "$TRACE_SOURCE/system_hbm_64ch.ini"
sed -i 's/^PRINT_MEM_TRACE=true/PRINT_MEM_TRACE=false/' "$TRACE_SOURCE/system_hbm_64ch.ini"
diff -u "$BUILD_SOURCE/system_hbm_64ch.ini" "$TRACE_SOURCE/system_hbm_64ch.ini" \
    > "$RUN_DIR/traces/config_diff.patch" || true

tail -n +2 "$CASE_MATRIX" | while IFS=, read -r split case_id kernel elements output_dim input_dim vectors trace; do
    [[ "$trace" == "true" ]] || continue
    trace_dir="$RUN_DIR/traces/$case_id"
    mkdir -p "$trace_dir"
    if [[ "$kernel" == "GEMV" ]]; then
        args=(--case-id "$case_id" --kernel GEMV --output "$output_dim" --input "$input_dim" --vectors "$vectors" --verify)
    else
        args=(--case-id "$case_id" --kernel "$kernel" --elements "$elements" --verify)
    fi
    printf '%q ' "/phase3/raw/$RUN_ID/build/phase3_runner" "${args[@]}" > "$trace_dir/command.txt"
    printf '\n' >> "$trace_dir/command.txt"
    run_container "/phase3/raw/$RUN_ID/trace_source" \
        "/phase3/raw/$RUN_ID/build/phase3_runner" "${args[@]}" \
        > "$trace_dir/stdout.txt" 2> "$trace_dir/stderr.txt"
    trace_exit=$?
    record_exit "$trace_dir/exit_code.txt" "$trace_exit"
    if [[ "$trace_exit" -ne 0 ]]; then
        printf '%s\n' "$case_id exit=$trace_exit" >> "$RUN_DIR/trace_failures.txt"
    fi
    gzip -n "$trace_dir/stdout.txt"
    sha256sum "$trace_dir/stdout.txt.gz" "$trace_dir/stderr.txt" \
        > "$trace_dir/output_sha256sums.txt"
done

if [[ -s "$RUN_DIR/trace_failures.txt" ]]; then
    echo "TRACE_GATE_FAILED" > "$RUN_DIR/status.txt"
    exit 23
fi

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$RUN_DIR/environment/toolchain.txt"
echo "SIMULATOR_CAPTURE_COMPLETE_ANALYSIS_PENDING" > "$RUN_DIR/status.txt"
python3 "$ARM_DIR/analysis/analyze_phase3.py" --arm-dir "$ARM_DIR" --run-id "$RUN_ID"
analysis_exit=$?
if [[ "$analysis_exit" -ne 0 ]]; then
    echo "NATIVE_ANALYSIS_FAILED" > "$RUN_DIR/status.txt"
    exit "$analysis_exit"
fi
MPLCONFIGDIR="$RUN_DIR/matplotlib" python3 "$ARM_DIR/analysis/complete_stopped_bundle.py" --arm-dir "$ARM_DIR"
presentation_exit=$?
if [[ "$presentation_exit" -ne 0 ]]; then
    echo "STOPPED_BUNDLE_RENDER_FAILED" > "$RUN_DIR/status.txt"
    exit "$presentation_exit"
fi
echo "NATIVE_CAPTURE_COMPLETE_FULL_SPEC_STOP_PRESERVED" > "$RUN_DIR/status.txt"
(
    cd "$RUN_DIR" || exit 125
    find . -type f ! -name artifact_sha256sums.txt -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$RUN_DIR/artifact_sha256sums.txt"
echo "Phase-3 native capture and stopped-arm analysis complete: $RUN_DIR"
