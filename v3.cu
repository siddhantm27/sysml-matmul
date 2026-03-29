#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// ─── Error-checking macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* ============================================================
   Edit ONLY this section.
   ============================================================ */

struct GPUTimer {
    cudaEvent_t start, stop;

    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic() {
        cudaEventRecord(start);
    }

    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

#define BLK_M 64
#define BLK_N 64
#define BLK_K 8

// ✅ Correct per-thread work
#define THR_M 4
#define THR_N 4

#define PAD 1

__global__ void matmul_kernel_optimized(const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ C,
                                        int N)
{
    __shared__ float smem_A[BLK_M][BLK_K + PAD];
    __shared__ float smem_B[BLK_K][BLK_N + PAD];

    const int tid = threadIdx.x;

    const int block_row = blockIdx.y * BLK_M;
    const int block_col = blockIdx.x * BLK_N;

    // ✅ Correct 16×16 thread mapping
    const int thread_row = (tid / 16) * THR_M;
    const int thread_col = (tid % 16) * THR_N;

    float reg_C[THR_M][THR_N];

#pragma unroll
    for (int i = 0; i < THR_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < THR_N; ++j)
        {
            reg_C[i][j] = 0.0f;
        }
    }

    for (int k_base = 0; k_base < N; k_base += BLK_K)
    {

        // Load A
        for (int load_idx = tid; load_idx < BLK_M * BLK_K; load_idx += blockDim.x)
        {
            int sm_row = load_idx / BLK_K;
            int sm_col = load_idx % BLK_K;
            int gm_row = block_row + sm_row;
            int gm_col = k_base + sm_col;

            smem_A[sm_row][sm_col] = (gm_row < N && gm_col < N) ? A[gm_row * N + gm_col] : 0.0f;
        }

        // Load B
        for (int load_idx = tid; load_idx < BLK_K * BLK_N; load_idx += blockDim.x)
        {
            int sm_row = load_idx / BLK_N;
            int sm_col = load_idx % BLK_N;
            int gm_row = k_base + sm_row;
            int gm_col = block_col + sm_col;

            smem_B[sm_row][sm_col] = (gm_row < N && gm_col < N) ? B[gm_row * N + gm_col] : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLK_K; ++k)
        {
            float a_reg[THR_M];
            float b_reg[THR_N];

#pragma unroll
            for (int i = 0; i < THR_M; ++i)
            {
                a_reg[i] = smem_A[thread_row + i][k];
            }

#pragma unroll
            for (int j = 0; j < THR_N; ++j)
            {
                b_reg[j] = smem_B[k][thread_col + j];
            }

#pragma unroll
            for (int i = 0; i < THR_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < THR_N; ++j)
                {
                    reg_C[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

// Store results
#pragma unroll
    for (int i = 0; i < THR_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < THR_N; ++j)
        {
            int c_row = block_row + thread_row + i;
            int c_col = block_col + thread_col + j;
            if (c_row < N && c_col < N)
            {
                C[c_row * N + c_col] = reg_C[i][j];
            }
        }
    }
}

void matmul_gpu(int N,
                const float *A_h,
                const float *B_h,
                float *C_h)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + BLK_N - 1) / BLK_N, (N + BLK_M - 1) / BLK_M);

    GPUTimer timer;
    timer.tic();

    matmul_kernel_optimized<<<grid, block>>>(A_d, B_d, C_d, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = timer.toc();
    printf("V3 Kernel time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

/* ============================================================
   END OF STUDENT CODE — do not modify below this line
   ============================================================ */

// ─── CPU reference ────────────────────────────────────────────────────────────
static void matmul_cpu(int N,
                       const float *A,
                       const float *B,
                       float *C)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < N; ++k)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// ─── Element-wise verification ────────────────────────────────────────────────
static bool verify(int N, const float *ref, const float *gpu,
                   float tol = 1e-2f)
{
    for (int i = 0; i < N * N; ++i)
    {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > tol)
        {
            int row = i / N, col = i % N;
            fprintf(stderr,
                    "MISMATCH at (%d,%d): ref=%.6f  gpu=%.6f  |diff|=%.2e\n",
                    row, col, ref[i], gpu[i], diff);
            return false;
        }
    }
    return true;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main()
{
    // ── Correctness tests (small sizes, CPU reference) ────────
    printf("=== Correctness Tests ===\n");
    {
        const std::vector<int> small_sizes = {64, 128, 256, 512};
        bool all_ok = true;

        for (int N : small_sizes)
        {

            std::vector<float> A(N * N), B(N * N),
                C_cpu(N * N, 0.f),
                C_gpu(N * N, 0.f);

            for (int i = 0; i < N * N; ++i)
            {
                A[i] = (float)(i % 97) / 97.f;
                B[i] = (float)((i * 7 + 3) % 97) / 97.f;
            }

            matmul_cpu(N, A.data(), B.data(), C_cpu.data());
            matmul_gpu(N, A.data(), B.data(), C_gpu.data());

            bool ok = verify(N, C_cpu.data(), C_gpu.data());
            printf("  N = %4d : %s\n", N, ok ? "PASSED" : "FAILED");
            all_ok &= ok;
        }

        if (!all_ok)
        {
            fprintf(stderr,
                    "\nCorrectness FAILED — fix your kernel before optimising.\n");
            return EXIT_FAILURE;
        }
        printf("All correctness tests PASSED.\n\n");
    }

    return EXIT_SUCCESS;
}
