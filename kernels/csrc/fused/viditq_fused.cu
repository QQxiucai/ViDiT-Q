/*
 * ViDiT-Q Fused Kernel: channel_mask + Random Sign + FWHT + hadK matmul + Quantization
 *
 * Algorithm (verified against Python matmul_hadU reference):
 *   1. channel_mask scaling + random_sign flip (element-wise)
 *   2. FWHT butterfly: 3 stages for K=1152, K_BLOCK=144
 *      Stage s (unit_size = 2^s, group_size = 2*unit_size):
 *        For each group: pair first unit_size elems with second unit_size elems
 *        output = [first+second, first-second]  (element-wise on unit_size elems)
 *   3. hadK matmul: [144,144] @ data[144,8] -> [144,8]
 *      Operates on the 144 "groups" dimension (K_BLOCK after FWHT)
 *   4. Scale by 1/sqrt(K) + per-token INT8 quantization
 */

#include <cuda_fp16.h>
#include <torch/extension.h>

#include "../utils.cuh"
#include "../reduction_utils.cuh"

// ---------------------------------------------------------------------------
// Main kernel
//   K=1152, K_BLOCK=144, N_STAGES=3 (log2(K/K_BLOCK)), BLOCK_THREADS divisible by 32
// ---------------------------------------------------------------------------
template <int K, int K_BLOCK, int N_STAGES, int BLOCK_THREADS>
__global__ void ViDiTQActQuantKernel(
    const half* __restrict__ input,          // [M, K] FP16
    const half* __restrict__ channel_mask,   // [K] FP16
    const int8_t* __restrict__ random_signs, // [K] INT8 (+/-1)
    const half* __restrict__ hadK,           // [K_BLOCK, K_BLOCK] FP16 row-major
    int8_t* __restrict__ output,             // [M, K] INT8
    half* __restrict__ scale_output,         // [M] FP16
    half* __restrict__ sum_output,           // [M] FP16 (nullable)
    int M)
{
    static_assert(K % K_BLOCK == 0, "K must be divisible by K_BLOCK");
    static_assert(K / K_BLOCK == (1 << N_STAGES), "K/K_BLOCK must be a power of 2");
    static_assert(K % BLOCK_THREADS == 0, "K must be divisible by BLOCK_THREADS");

    constexpr int ELEMS_PER_THREAD = K / BLOCK_THREADS;

    int token_idx = blockIdx.x;
    if (token_idx >= M) return;

    // Shared memory: act_buffer[K] + hadK_smem[K_BLOCK * K_BLOCK]
    extern __shared__ __align__(16) char smem_raw[];
    half* act_buffer  = reinterpret_cast<half*>(smem_raw);
    half* hadK_smem   = reinterpret_cast<half*>(smem_raw + K * sizeof(half));

    int tid = threadIdx.x;
    const half* input_row = input + token_idx * K;

    // =========================================================================
    // Step 1: Load input -> shared memory, apply channel_mask + random_sign
    // =========================================================================
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid * ELEMS_PER_THREAD + i;
        half val = __ldg(input_row + idx);
        val = __hmul(val, __ldg(channel_mask + idx));
        int8_t sign = __ldg(random_signs + idx);
        if (sign < 0) { val = __hneg(val); }
        act_buffer[idx] = val;
    }
    __syncthreads();

    // =========================================================================
    // Step 2: FWHT butterfly stages
    //   Stage s: unit_size=2^s, group_size=2*unit_size, n_groups=K/group_size
    //   For each group: pair first unit_size with second unit_size
    //   Result: [first+second, first-second] concatenated
    // =========================================================================
    #pragma unroll
    for (int stage = 0; stage < N_STAGES; stage++) {
        int unit_size  = 1 << stage;          // 1, 2, 4
        int group_size = 2 * unit_size;       // 2, 4, 8
        int n_groups   = K / group_size;      // 576, 288, 144

        // Each thread processes ELEMS_PER_THREAD elements.
        // The elements are spread across groups. We compute for each
        // element: its group index, whether it's in first or second half,
        // and process the pair.
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int flat_idx = tid * ELEMS_PER_THREAD + i;
            int group    = flat_idx / group_size;
            int offset   = flat_idx % group_size;

            // Only process offsets in the first unit_size (other threads handle the rest)
            if (offset < unit_size) {
                int first_idx  = group * group_size + offset;
                int second_idx = group * group_size + offset + unit_size;

                // Use FP32 for butterfly to match PyTorch precision
                // (PyTorch promotes FP16 add/sub to FP32 internally)
                float a = __half2float(act_buffer[first_idx]);
                float b = __half2float(act_buffer[second_idx]);
                act_buffer[first_idx]  = __float2half_rn(a + b);
                act_buffer[second_idx] = __float2half_rn(a - b);
            }
        }
        __syncthreads();
    }

    // =========================================================================
    // Step 3: hadK matmul
    //   After FWHT: data is [K_BLOCK=144 groups, each of size K/K_BLOCK=8]
    //   i.e., data[g * 8 + j] = g-th group, j-th element within group
    //
    //   Need: hadK @ data_t where data_t has shape [K_BLOCK, K/K_BLOCK]
    //   data_t[j, g] = data[g * 8 + j]  (transpose access pattern)
    //
    //   Result: out[g * 8 + j] = sum_k hadK[g][k] * data_t[k][j]
    //                          = sum_k hadK[g][k] * data[k * 8 + j]
    //
    //   Each thread computes ELEMS_PER_THREAD output elements.
    //   For output element at flat_idx:
    //     group g = flat_idx / 8     (0..143)
    //     elem  j = flat_idx % 8     (0..7)
    //   Accumulate over k = 0..143:
    //     out[flat_idx] += hadK[g][k] * data[k * 8 + j]
    // =========================================================================
    {
        // Load hadK into shared memory (cooperative)
        constexpr int HADK_SIZE = K_BLOCK * K_BLOCK;
        constexpr int HADK_PER_THREAD = (HADK_SIZE + BLOCK_THREADS - 1) / BLOCK_THREADS;
        #pragma unroll
        for (int i = 0; i < HADK_PER_THREAD; i++) {
            int idx = tid * HADK_PER_THREAD + i;
            if (idx < HADK_SIZE) {
                hadK_smem[idx] = __ldg(hadK + idx);
            }
        }
    }
    __syncthreads();

    // Accumulate in registers (FP32 for precision)
    constexpr int N_COLS = K / K_BLOCK;  // 8
    float accum[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        accum[i] = 0.0f;
    }

    // Dot product: for each output element, sum over k = 0..K_BLOCK-1
    // hadK is row-major: hadK[row][col] = hadK_smem[row * K_BLOCK + col]
    #pragma unroll
    for (int k = 0; k < K_BLOCK; k++) {
        half act_col_vals[ELEMS_PER_THREAD];
        half hadK_vals[ELEMS_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int flat_idx = tid * ELEMS_PER_THREAD + i;
            int g = flat_idx / N_COLS;      // group (output row in hadK)
            int j = flat_idx % N_COLS;       // column within group

            // hadK[g][k] = hadK_smem[g * K_BLOCK + k]
            hadK_vals[i] = hadK_smem[g * K_BLOCK + k];
            // data[k * N_COLS + j] = act_buffer[k * N_COLS + j]
            act_col_vals[i] = act_buffer[k * N_COLS + j];
        }

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            accum[i] += __half2float(hadK_vals[i]) * __half2float(act_col_vals[i]);
        }
    }

    // Scale by 1/sqrt(K) and write back to act_buffer
    // 1/sqrt(1152) = 0.0294627825
    constexpr float INV_SQRT_K = 1.0f / 34.176014981f;  // sqrt(1152) ≈ 34.176...
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int flat_idx = tid * ELEMS_PER_THREAD + i;
        act_buffer[flat_idx] = __float2half_rn(accum[i] * INV_SQRT_K);
    }
    __syncthreads();

    // =========================================================================
    // Step 4: Absmax quantization (per-token dynamic)
    // =========================================================================
    {
        float local_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int idx = tid * ELEMS_PER_THREAD + i;
            float abs_val = __half2float(__habs(act_buffer[idx]));
            if (abs_val > local_amax) local_amax = abs_val;
        }

        float block_amax = vllm::blockReduceMax(local_amax);

        __shared__ float s_amax;
        if (tid == 0) {
            s_amax = block_amax;
            if (block_amax < 1e-8f) block_amax = 1e-8f;  // avoid div by zero
            scale_output[token_idx] = __float2half_rn(block_amax / 127.0f);
        }
        __syncthreads();

        float inv_scale = 127.0f / s_amax;

        int8_t* output_row = output + token_idx * K;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int idx = tid * ELEMS_PER_THREAD + i;
            output_row[idx] = float_to_int8_rn(__half2float(act_buffer[idx]) * inv_scale);
        }

        // Optional sum_output for bias correction
        if (sum_output != nullptr) {
            int local_sum_int = 0;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                local_sum_int += static_cast<int>(output_row[tid * ELEMS_PER_THREAD + i]);
            }
            int block_sum = vllm::blockReduceSum(local_sum_int);
            if (tid == 0) {
                sum_output[token_idx] = __float2half_rn(__int2float_rn(block_sum) / inv_scale);
            }
        }
    }
}

// =============================================================================
// Host launch function
// =============================================================================
torch::Tensor viditq_act_quant_fuse(
    torch::Tensor input,           // [M, K] FP16
    torch::Tensor channel_mask,    // [K] FP16
    torch::Tensor random_signs,    // [K] INT8 (+/-1)
    torch::Tensor hadK,            // [K_BLOCK, K_BLOCK] FP16
    torch::Tensor scale_output,    // [M] FP16 (output)
    torch::Tensor sum_output)      // [M] FP16 (output, can be empty)
{
    CHECK_CUDA(input);
    CHECK_CUDA(channel_mask);
    CHECK_CUDA(random_signs);
    CHECK_CUDA(hadK);
    CHECK_CUDA(scale_output);
    CHECK_CUDA(sum_output);

    CHECK_DTYPE(input, torch::kHalf);
    CHECK_DTYPE(channel_mask, torch::kHalf);
    CHECK_DTYPE(random_signs, torch::kInt8);
    CHECK_DTYPE(hadK, torch::kHalf);
    CHECK_DTYPE(scale_output, torch::kHalf);
    CHECK_DTYPE(sum_output, torch::kHalf);

    CHECK_CONTIGUOUS(input);

    const int M = input.size(0);
    const int K = input.size(1);
    const int K_BLOCK = hadK.size(0);

    TORCH_CHECK(K % K_BLOCK == 0, "K must be divisible by K_BLOCK");
    TORCH_CHECK(K == channel_mask.size(0), "channel_mask size mismatch");
    TORCH_CHECK(K == random_signs.size(0), "random_signs size mismatch");
    TORCH_CHECK(K_BLOCK == hadK.size(1), "hadK must be square");

    TORCH_CHECK(scale_output.size(0) == M, "scale_output size mismatch");
    TORCH_CHECK(sum_output.size(0) == M || sum_output.numel() == 0,
                "sum_output size mismatch");

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(input.device());
    torch::Tensor output = torch::empty({M, K}, options);

    TORCH_CHECK(K == 1152 && K_BLOCK == 144,
                "Only K=1152, K_BLOCK=144 supported. Got K=", K, ", K_BLOCK=", K_BLOCK);

    constexpr int K_VAL = 1152;
    constexpr int KB_VAL = 144;
    constexpr int N_STAGES_VAL = 3;
    constexpr int BLOCK_THREADS_VAL = 288;

    const int smem_size = (K_VAL + KB_VAL * KB_VAL) * sizeof(half);

    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    TORCH_CHECK(smem_size <= max_smem,
                "Shared memory required (", smem_size, ") exceeds device limit (", max_smem, ")");

    dim3 grid(M);
    dim3 block(BLOCK_THREADS_VAL);

    auto kernel = ViDiTQActQuantKernel<K_VAL, KB_VAL, N_STAGES_VAL, BLOCK_THREADS_VAL>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    bool has_sum = (sum_output.numel() > 0);

    kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const half*>(input.data_ptr()),
        reinterpret_cast<const half*>(channel_mask.data_ptr()),
        random_signs.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(hadK.data_ptr()),
        output.data_ptr<int8_t>(),
        reinterpret_cast<half*>(scale_output.data_ptr()),
        has_sum ? reinterpret_cast<half*>(sum_output.data_ptr()) : nullptr,
        M);

    return output;
}

// =============================================================================
// Reserved: channel_mask scaling + quantization only (no FWHT/hadK).
// Currently unused — reserved for future non-ViDiT-Q layer optimization.
// =============================================================================
template <int K, int BLOCK_THREADS>
__global__ void ChannelScaleQuantKernel(
    const half* __restrict__ input,
    const half* __restrict__ channel_mask,
    int8_t* __restrict__ output,
    half* __restrict__ scale_output,
    half* __restrict__ sum_output,
    int M)
{
    int token_idx = blockIdx.x;
    if (token_idx >= M) return;

    constexpr int ELEMS_PER_THREAD = K / BLOCK_THREADS;
    int tid = threadIdx.x;
    const half* input_row = input + token_idx * K;

    half local_vals[ELEMS_PER_THREAD];
    float local_amax = 0.0f;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid * ELEMS_PER_THREAD + i;
        half val = __hmul(__ldg(input_row + idx), __ldg(channel_mask + idx));
        local_vals[i] = val;
        float abs_val = __half2float(__habs(val));
        if (abs_val > local_amax) local_amax = abs_val;
    }

    float block_amax = vllm::blockReduceMax(local_amax);

    __shared__ float s_amax;
    if (tid == 0) {
        s_amax = block_amax;
        if (block_amax < 1e-8f) block_amax = 1e-8f;
        scale_output[token_idx] = __float2half_rn(block_amax / 127.0f);
    }
    __syncthreads();

    float inv_scale = 127.0f / s_amax;
    int8_t* output_row = output + token_idx * K;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid * ELEMS_PER_THREAD + i;
        output_row[idx] = float_to_int8_rn(__half2float(local_vals[i]) * inv_scale);
    }

    if (sum_output != nullptr) {
        int local_sum_int = 0;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            local_sum_int += static_cast<int>(output_row[tid * ELEMS_PER_THREAD + i]);
        }
        int block_sum = vllm::blockReduceSum(local_sum_int);
        if (tid == 0) {
            sum_output[token_idx] = __float2half_rn(__int2float_rn(block_sum) / inv_scale);
        }
    }
}

torch::Tensor channel_scale_quant(
    torch::Tensor input,
    torch::Tensor channel_mask,
    torch::Tensor scale_output,
    torch::Tensor sum_output)
{
    CHECK_CUDA(input);
    CHECK_CUDA(channel_mask);
    CHECK_DTYPE(input, torch::kHalf);
    CHECK_DTYPE(channel_mask, torch::kHalf);
    CHECK_CONTIGUOUS(input);

    const int M = input.size(0);
    const int K = input.size(1);
    TORCH_CHECK(K == channel_mask.size(0), "channel_mask size mismatch");
    TORCH_CHECK(K == 1152, "Only K=1152 supported, got K=", K);

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(input.device());
    torch::Tensor output = torch::empty({M, K}, options);

    constexpr int K_VAL = 1152;
    constexpr int BLOCK_THREADS_VAL = 288;

    dim3 grid(M);
    dim3 block(BLOCK_THREADS_VAL);

    bool has_sum = (sum_output.numel() > 0);

    ChannelScaleQuantKernel<K_VAL, BLOCK_THREADS_VAL><<<grid, block>>>(
        reinterpret_cast<const half*>(input.data_ptr()),
        reinterpret_cast<const half*>(channel_mask.data_ptr()),
        output.data_ptr<int8_t>(),
        reinterpret_cast<half*>(scale_output.data_ptr()),
        has_sum ? reinterpret_cast<half*>(sum_output.data_ptr()) : nullptr,
        M);

    return output;
}
