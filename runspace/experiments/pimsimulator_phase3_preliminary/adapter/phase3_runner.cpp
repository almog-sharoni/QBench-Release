#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "MemoryController.h"
#include "MultiChannelMemorySystem.h"
#include "tests/PIMKernel.h"
#include "tests/TestCases.h"

using namespace DRAMSim;

namespace {

constexpr uint64_t kFp16PerBurst = 16;
constexpr uint64_t kElementwiseTileElements = 131072;
constexpr uint64_t kGemvInputTileElements = 128;
constexpr uint64_t kGemvOutputTileElements = 4096;

struct Options {
    std::string case_id;
    std::string kernel;
    uint64_t logical_elements = 0;
    uint64_t output_dim = 0;
    uint64_t input_dim = 0;
    uint64_t vectors = 1;
    bool verify = false;
};

struct Counters {
    uint64_t reads = 0;
    uint64_t writes = 0;
};

struct PhaseStats {
    uint64_t cycles = 0;
    uint64_t reads = 0;
    uint64_t writes = 0;

    PhaseStats& operator+=(const PhaseStats& rhs)
    {
        cycles += rhs.cycles;
        reads += rhs.reads;
        writes += rhs.writes;
        return *this;
    }
};

uint64_t ceil_to(uint64_t value, uint64_t quantum)
{
    if (quantum == 0) throw std::invalid_argument("zero quantum");
    return ((value + quantum - 1) / quantum) * quantum;
}

uint64_t parse_u64(const std::string& name, const std::string& value)
{
    size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) throw std::invalid_argument("invalid " + name + ": " + value);
    return static_cast<uint64_t>(parsed);
}

Options parse_options(int argc, char** argv)
{
    std::map<std::string, std::string> values;
    bool verify = false;
    for (int index = 1; index < argc; ++index) {
        const std::string arg(argv[index]);
        if (arg == "--verify") {
            verify = true;
            continue;
        }
        if (arg.rfind("--", 0) != 0 || index + 1 >= argc)
            throw std::invalid_argument("expected --key value, got: " + arg);
        values[arg.substr(2)] = argv[++index];
    }

    Options options;
    options.case_id = values.at("case-id");
    options.kernel = values.at("kernel");
    options.verify = verify;
    if (options.kernel == "ADD" || options.kernel == "RELU") {
        options.logical_elements = parse_u64("elements", values.at("elements"));
        if (options.logical_elements == 0) throw std::invalid_argument("elements must be positive");
    } else if (options.kernel == "GEMV") {
        options.output_dim = parse_u64("output", values.at("output"));
        options.input_dim = parse_u64("input", values.at("input"));
        options.vectors = parse_u64("vectors", values.at("vectors"));
        if (options.output_dim == 0 || options.output_dim > kGemvOutputTileElements)
            throw std::invalid_argument("GEMV output must be in [1,4096]");
        if (options.input_dim == 0 || options.vectors == 0)
            throw std::invalid_argument("GEMV input and vectors must be positive");
    } else {
        throw std::invalid_argument("unsupported kernel: " + options.kernel);
    }
    return options;
}

Counters snapshot(const std::shared_ptr<MultiChannelMemorySystem>& memory)
{
    Counters result;
    for (auto* channel : memory->channels) {
        result.reads += channel->memoryController->totalReads;
        result.writes += channel->memoryController->totalWrites;
    }
    return result;
}

PhaseStats run_phase(const std::shared_ptr<MultiChannelMemorySystem>& memory,
                     const std::shared_ptr<PIMKernel>& kernel,
                     const std::function<void()>& enqueue)
{
    const uint64_t cycle_before = kernel->getCycle();
    const Counters counters_before = snapshot(memory);
    enqueue();
    kernel->runPIM();
    const uint64_t cycle_after = kernel->getCycle();
    const Counters counters_after = snapshot(memory);
    return {
        cycle_after - cycle_before,
        counters_after.reads - counters_before.reads,
        counters_after.writes - counters_before.writes,
    };
}

NumpyBurstType make_matrix(uint64_t rows, uint64_t columns)
{
    NumpyBurstType result;
    result.shape.push_back(rows);
    result.shape.push_back(columns);
    result.loadTobShape(static_cast<double>(kFp16PerBurst));
    BurstType zero;
    zero.set(convertF2H(0.0f));
    result.bData.assign(result.getTotalDim(), zero);
    return result;
}

void set_matrix_value(NumpyBurstType& matrix, uint64_t row, uint64_t column, float value)
{
    const uint64_t burst_index = row * matrix.bShape[1] + column / kFp16PerBurst;
    matrix.bData.at(burst_index).fp16Data_[column % kFp16PerBurst] = convertF2H(value);
}

void fill_all(NumpyBurstType& tensor, float value)
{
    const fp16 converted = convertF2H(value);
    for (auto& burst : tensor.bData) burst.set(converted);
}

uint64_t fp16_bits(fp16 value)
{
    const fp16i bits(value);
    return bits.ival;
}

void print_common_prefix(const Options& options)
{
    std::cout << "PHASE3_JSON={";
    std::cout << "\"schema_version\":1";
    std::cout << ",\"case_id\":\"" << options.case_id << "\"";
    std::cout << ",\"kernel\":\"" << options.kernel << "\"";
    std::cout << ",\"evidence_label\":\"SIMULATED_PIM_HBM2\"";
    std::cout << ",\"architecture\":\"HBM2\"";
    std::cout << ",\"precision\":\"FP16\"";
    std::cout << ",\"pim_channels\":64";
    std::cout << ",\"pim_ranks\":1";
}

void print_phases(const PhaseStats& residency, const PhaseStats& execution,
                  const PhaseStats& readback)
{
    const PhaseStats total{
        residency.cycles + execution.cycles + readback.cycles,
        residency.reads + execution.reads + readback.reads,
        residency.writes + execution.writes + readback.writes,
    };
    auto emit = [](const char* name, const PhaseStats& value) {
        std::cout << "\"" << name << "\":{";
        std::cout << "\"cycles\":" << value.cycles;
        std::cout << ",\"read_transactions\":" << value.reads;
        std::cout << ",\"write_transactions\":" << value.writes << "}";
    };
    std::cout << ",\"phases\":{";
    emit("initial_residency", residency);
    std::cout << ",";
    emit("kernel_execution", execution);
    std::cout << ",";
    emit("result_readback", readback);
    std::cout << ",";
    emit("native_total", total);
    std::cout << "}";
}

int run_elementwise(const Options& options)
{
    const uint64_t padded_elements = ceil_to(options.logical_elements, kElementwiseTileElements);
    const uint64_t padded_bursts = padded_elements / kFp16PerBurst;
    const uint64_t parallel_tiles = padded_elements / kElementwiseTileElements;
    const uint64_t input_count = options.kernel == "ADD" ? 2 : 1;

    auto memory = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".", options.case_id,
        256 * 64 * 2);
    auto kernel = std::make_shared<PIMKernel>(memory, 64, 1);

    NumpyBurstType input0 = make_matrix(1, padded_elements);
    NumpyBurstType input1 = make_matrix(1, padded_elements);
    if (options.kernel == "ADD") {
        fill_all(input0, 1.0f);
        fill_all(input1, 2.0f);
    } else {
        fill_all(input0, 1.0f);
    }

    const PhaseStats residency = run_phase(memory, kernel, [&]() {
        kernel->preloadNoReplacement(&input0, 0, 0);
        if (options.kernel == "ADD") kernel->preloadNoReplacement(&input1, 128, 0);
    });

    const KernelType kernel_type = options.kernel == "ADD" ? KernelType::ADD : KernelType::RELU;
    const PhaseStats execution = run_phase(memory, kernel, [&]() {
        kernel->executeEltwise(static_cast<int>(padded_bursts), pimBankType::ALL_BANK, kernel_type,
                               0, 256, options.kernel == "ADD" ? 128 : 0);
    });

    std::vector<BurstType> result(padded_bursts);
    const PhaseStats readback = run_phase(memory, kernel, [&]() {
        kernel->readData(result.data(), padded_bursts, 256, 0);
    });

    uint64_t mismatches = 0;
    if (options.verify) {
        const uint64_t expected = fp16_bits(convertF2H(options.kernel == "ADD" ? 3.0f : 1.0f));
        for (uint64_t element = 0; element < options.logical_elements; ++element) {
            const uint64_t burst = element / kFp16PerBurst;
            const uint64_t lane = element % kFp16PerBurst;
            if (result[burst].u16Data_[lane] != expected) ++mismatches;
        }
    }

    print_common_prefix(options);
    std::cout << ",\"logical_elements\":" << options.logical_elements;
    std::cout << ",\"padded_elements\":" << padded_elements;
    std::cout << ",\"padded_bursts\":" << padded_bursts;
    std::cout << ",\"parallel_tiles\":" << parallel_tiles;
    std::cout << ",\"input_count\":" << input_count;
    std::cout << ",\"traffic_bytes\":{";
    std::cout << "\"logical_input\":" << input_count * options.logical_elements * 2;
    std::cout << ",\"padded_input\":" << input_count * padded_elements * 2;
    std::cout << ",\"logical_output\":" << options.logical_elements * 2;
    std::cout << ",\"padded_output_readback\":" << padded_elements * 2 << "}";
    print_phases(residency, execution, readback);
    std::cout << ",\"verification_requested\":" << (options.verify ? "true" : "false");
    std::cout << ",\"exact_bit_mismatches\":" << mismatches;
    std::cout << ",\"outside_pim_uncosted_operations\":0";
    std::cout << "}\n";
    return mismatches == 0 ? 0 : 10;
}

int run_gemv(const Options& options)
{
    const uint64_t padded_input = ceil_to(options.input_dim, kGemvInputTileElements);
    const uint64_t padded_output = kGemvOutputTileElements;
    const uint64_t input_tiles = padded_input / kGemvInputTileElements;
    const uint64_t weight_bursts = padded_output * (padded_input / kFp16PerBurst);
    const uint64_t partial_sum_bursts = options.output_dim * options.vectors;

    auto memory = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".", options.case_id,
        256 * 64 * 2);
    auto kernel = std::make_shared<PIMKernel>(memory, 64, 1);

    NumpyBurstType weights = make_matrix(padded_output, padded_input);
    NumpyBurstType input = make_matrix(1, padded_input);
    fill_all(input, 1.0f);
    if (options.verify) {
        for (uint64_t output = 0; output < options.output_dim; ++output)
            set_matrix_value(weights, output, output % options.input_dim, 1.0f);
    }

    const PhaseStats residency = run_phase(memory, kernel, [&]() {
        kernel->preloadGemv(&weights);
    });

    PhaseStats execution;
    PhaseStats readback;
    uint64_t mismatches = 0;
    const uint64_t expected = fp16_bits(convertF2H(1.0f));
    const unsigned result_col = kernel->getResultColGemv(
        static_cast<int>(padded_input / kFp16PerBurst), static_cast<int>(padded_output));

    for (uint64_t vector_index = 0; vector_index < options.vectors; ++vector_index) {
        execution += run_phase(memory, kernel, [&]() {
            kernel->executeGemv(&weights, &input, false);
        });
        std::vector<BurstType> result(options.output_dim);
        readback += run_phase(memory, kernel, [&]() {
            kernel->readResult(result.data(), pimBankType::ODD_BANK,
                               static_cast<int>(options.output_dim), 0, 0, result_col);
        });
        if (options.verify) {
            for (uint64_t output = 0; output < options.output_dim; ++output) {
                const uint64_t actual = fp16_bits(result[output].fp16ReduceSum());
                if (actual != expected) ++mismatches;
            }
        }
    }

    print_common_prefix(options);
    std::cout << ",\"logical_output_dim\":" << options.output_dim;
    std::cout << ",\"logical_input_dim\":" << options.input_dim;
    std::cout << ",\"vectors\":" << options.vectors;
    std::cout << ",\"padded_output_dim\":" << padded_output;
    std::cout << ",\"padded_input_dim\":" << padded_input;
    std::cout << ",\"input_tiles\":" << input_tiles;
    std::cout << ",\"weight_bursts\":" << weight_bursts;
    std::cout << ",\"partial_sum_bursts\":" << partial_sum_bursts;
    std::cout << ",\"traffic_bytes\":{";
    std::cout << "\"logical_weight\":" << options.output_dim * options.input_dim * 2;
    std::cout << ",\"padded_weight_residency\":" << padded_output * padded_input * 2;
    std::cout << ",\"logical_input\":" << options.vectors * options.input_dim * 2;
    std::cout << ",\"padded_input_upload\":" << options.vectors * padded_input * 2;
    std::cout << ",\"logical_output\":" << options.vectors * options.output_dim * 2;
    std::cout << ",\"partial_sum_readback\":" << partial_sum_bursts * 32 << "}";
    std::cout << ",\"logical_macs\":" << options.output_dim * options.input_dim * options.vectors;
    std::cout << ",\"padded_macs\":" << padded_output * padded_input * options.vectors;
    print_phases(residency, execution, readback);
    std::cout << ",\"verification_requested\":" << (options.verify ? "true" : "false");
    std::cout << ",\"exact_bit_mismatches\":" << mismatches;
    std::cout << ",\"outside_pim_uncosted_operations\":"
              << options.output_dim * options.vectors * 15;
    std::cout << ",\"outside_pim_uncosted_operation_type\":\"FP16 partial-sum reduction adds\"";
    std::cout << "}\n";
    return mismatches == 0 ? 0 : 11;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_options(argc, argv);
        if (options.kernel == "GEMV") return run_gemv(options);
        return run_elementwise(options);
    } catch (const std::exception& error) {
        std::cerr << "phase3_runner_error=" << error.what() << "\n";
        return 2;
    }
}
