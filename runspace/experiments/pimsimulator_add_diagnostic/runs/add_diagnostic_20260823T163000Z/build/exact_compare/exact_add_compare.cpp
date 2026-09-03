#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>

#include "MultiChannelMemorySystem.h"
#include "tests/PIMKernel.h"
#include "tests/TestCases.h"

using namespace DRAMSim;

int main()
{
    constexpr uint32_t kElements = 1024 * 1024;
    constexpr int kInputRow0 = 0;
    constexpr int kInputRow1 = 128;
    constexpr int kResultRow = 256;

    auto mem = std::make_shared<MultiChannelMemorySystem>(
        "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".", "exact_add_compare",
        256 * 64 * 2);
    auto kernel = std::make_shared<PIMKernel>(mem, 64, 1);
    DataDim data(KernelType::ADD, 1, kElements, kElements, true);

    kernel->preloadNoReplacement(&data.input_npbst_, kInputRow0, 0);
    kernel->preloadNoReplacement(&data.input1_npbst_, kInputRow1, 0);
    kernel->executeEltwise(data.dimTobShape(data.output_dim_), pimBankType::ALL_BANK,
                           KernelType::ADD, kInputRow0, kResultRow, kInputRow1);

    const size_t burst_count = data.dimTobShape(data.output_dim_);
    auto result = std::make_unique<BurstType[]>(burst_count);
    kernel->readData(result.get(), burst_count, kResultRow, 0);
    kernel->runPIM();

    uint64_t reference_bit_mismatches = 0;
    uint64_t recomputed_bit_mismatches = 0;
    uint64_t first_reference_mismatch = UINT64_MAX;
    uint64_t first_recomputed_mismatch = UINT64_MAX;

    for (size_t burst = 0; burst < burst_count; ++burst)
    {
        for (size_t lane = 0; lane < 16; ++lane)
        {
            const uint64_t element = burst * 16 + lane;
            const uint16_t actual = result[burst].u16Data_[lane];
            const uint16_t reference = data.output_npbst_.getBurst(burst).u16Data_[lane];
            const fp16 recomputed_fp16 = data.input_npbst_.getBurst(burst).fp16Data_[lane] +
                                        data.input1_npbst_.getBurst(burst).fp16Data_[lane];
            const fp16i recomputed_bits(recomputed_fp16);
            const uint16_t recomputed = recomputed_bits.ival;

            if (actual != reference)
            {
                if (first_reference_mismatch == UINT64_MAX) first_reference_mismatch = element;
                ++reference_bit_mismatches;
            }
            if (actual != recomputed)
            {
                if (first_recomputed_mismatch == UINT64_MAX) first_recomputed_mismatch = element;
                ++recomputed_bit_mismatches;
            }
        }
    }

    std::cout << "classification_label=SIMULATED_PIM_HBM2\n";
    std::cout << "elements_compared=" << kElements << "\n";
    std::cout << "bursts_compared=" << burst_count << "\n";
    std::cout << "reference_bit_mismatches=" << reference_bit_mismatches << "\n";
    std::cout << "recomputed_fp16_bit_mismatches=" << recomputed_bit_mismatches << "\n";
    std::cout << "first_reference_mismatch=";
    if (first_reference_mismatch == UINT64_MAX)
        std::cout << "NONE\n";
    else
        std::cout << first_reference_mismatch << "\n";
    std::cout << "first_recomputed_mismatch=";
    if (first_recomputed_mismatch == UINT64_MAX)
        std::cout << "NONE\n";
    else
        std::cout << first_recomputed_mismatch << "\n";
    std::cout << "simulated_cycles=" << kernel->getCycle() << "\n";

    return (reference_bit_mismatches == 0 && recomputed_bit_mismatches == 0) ? 0 : 1;
}
