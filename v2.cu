#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// ─── Error-checking macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
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

// Larger tile kernel with different blocking strategy
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define REG_M 4
#define REG_N 4

__global__ void matmul_kernel_large_tile(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int N)
{
    __shared__ float smem_A[TILE_M][TILE_K];
    __shared__ float smem_B[TILE_K][TILE_N];
    
    const int tid = threadIdx.x;
    
    // Block tile position
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;
    
    // ✅ FIXED mapping (16×16 threads, each computes 4×4)
    const int thread_row = (tid / 16) * REG_M;
    const int thread_col = (tid % 16) * REG_N;
    
    float acc[REG_M][REG_N];
    
    #pragma unroll
    for (int i = 0; i < REG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    for (int k_base = 0; k_base < N; k_base += TILE_K) {
        
        // Load A
        for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gm_row = block_row + row;
            int gm_col = k_base + col;
            
            smem_A[row][col] = (gm_row < N && gm_col < N) ? 
                               A[gm_row * N + gm_col] : 0.0f;
        }
        
        // Load B
        for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int gm_row = k_base + row;
            int gm_col = block_col + col;
            
            smem_B[row][col] = (gm_row < N && gm_col < N) ? 
                               B[gm_row * N + gm_col] : 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_reg[REG_M];
            float b_reg[REG_N];
            
            #pragma unroll
            for (int i = 0; i < REG_M; ++i) {
                a_reg[i] = smem_A[thread_row + i][k];
            }
            
            #pragma unroll
            for (int j = 0; j < REG_N; ++j) {
                b_reg[j] = smem_B[k][thread_col + j];
            }
            
            #pragma unroll
            for (int i = 0; i < REG_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_N; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    #pragma unroll
    for (int i = 0; i < REG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_N; ++j) {
            int c_row = block_row + thread_row + i;
            int c_col = block_col + thread_col + j;
            if (c_row < N && c_col < N) {
                C[c_row * N + c_col] = acc[i][j];
            }
        }
    }
}

void matmul_gpu(int N,
                const float* A_h,
                const float* B_h,
                      float* C_h)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    // ── Allocate device buffers ───────────────────────────────
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    // ── Transfer inputs to device ─────────────────────────────
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    // ── Launch matmul kernel ──────────────────────────────────
    {
        dim3 block(256);
        dim3 grid((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);
        
        GPUTimer timer;
        timer.tic();
        matmul_kernel_large_tile<<<grid, block>>>(A_d, B_d, C_d, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms = timer.toc();
        printf("V2 Kernel time: %.3f ms\n", ms);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Copy result back to host ──────────────────────────────
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    // ── Free device memory ────────────────────────────────────
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

/* ============================================================
   END OF STUDENT CODE — do not modify below this line
   ============================================================ */


// ─── CPU reference ────────────────────────────────────────────────────────────
static void matmul_cpu(int N,
                       const float* A,
                       const float* B,
                             float* C)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k)
                s += A[i*N+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

// ─── Element-wise verification ────────────────────────────────────────────────
static bool verify(int N, const float* ref, const float* gpu,
                   float tol = 1e-2f)
{
    for (int i = 0; i < N*N; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > tol) {
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

        for (int N : small_sizes) {

            std::vector<float> A(N*N), B(N*N),
                               C_cpu(N*N, 0.f),
                               C_gpu(N*N, 0.f);

            for (int i = 0; i < N*N; ++i) {
                A[i] = (float)(i % 97) / 97.f;
                B[i] = (float)((i * 7 + 3) % 97) / 97.f;
            }

            matmul_cpu(N, A.data(), B.data(), C_cpu.data());
            matmul_gpu(N, A.data(), B.data(), C_gpu.data());

            bool ok = verify(N, C_cpu.data(), C_gpu.data());
            printf("  N = %4d : %s\n", N, ok ? "PASSED" : "FAILED");
            all_ok &= ok;
        }

        if (!all_ok) {
            fprintf(stderr,
                    "\nCorrectness FAILED — fix your kernel before optimising.\n");
            return EXIT_FAILURE;
        }
        printf("All correctness tests PASSED.\n\n");
    }

    return EXIT_SUCCESS;
}
