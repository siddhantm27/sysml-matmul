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
// ─── CUDA Timer Utility ──────────────────────────────────────
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

// Hierarchical tiling with warp-level cooperation
// BM, BN: thread block output tile
// BK: shared memory tile along K dimension
// TM, TN: thread output tile (register blocking)

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void matmul_kernel_hierarchical(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N)
{
    __shared__ float smem_A[BM][BK];
    __shared__ float smem_B[BK][BN];
    
    const int tid = threadIdx.x;
    
    // Thread block tile location
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    
    // Thread's local tile position within the block
    const int thread_row = (tid / 16) * TM;
    const int thread_col = (tid % 16) * TN;
    
    // Global output position for this thread
    const int out_row = block_row + thread_row;
    const int out_col = block_col + thread_col;
    
    // Register tiles for accumulation
    float reg_C[TM][TN];
    
    // Initialize accumulator
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            reg_C[i][j] = 0.0f;
        }
    }
    
    // Main loop over K dimension
    for (int k_base = 0; k_base < N; k_base += BK) {
        // Cooperatively load A tile into shared memory
        #pragma unroll
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int gm_row = block_row + row;
            int gm_col = k_base + col;
            
            if (gm_row < N && gm_col < N) {
                smem_A[row][col] = A[gm_row * N + gm_col];
            } else {
                smem_A[row][col] = 0.0f;
            }
        }
        
        // Cooperatively load B tile into shared memory
        #pragma unroll
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int gm_row = k_base + row;
            int gm_col = block_col + col;
            
            if (gm_row < N && gm_col < N) {
                smem_B[row][col] = B[gm_row * N + gm_col];
            } else {
                smem_B[row][col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute on the tile with register blocking
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float reg_A[TM];
            float reg_B[TN];
            
            // Load A strip into registers
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int a_row = thread_row + i;
                reg_A[i] = smem_A[a_row][k];
            }
            
            // Load B strip into registers
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int b_col = thread_col + j;
                reg_B[j] = smem_B[k][b_col];
            }
            
            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c_row = out_row + i;
            int c_col = out_col + j;
            if (c_row < N && c_col < N) {
                C[c_row * N + c_col] = reg_C[i][j];
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

    {
        dim3 block(256);  // 256 threads per block (8 warps)
        dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);

        GPUTimer timer;
        timer.tic();

        matmul_kernel_hierarchical<<<grid, block>>>(A_d, B_d, C_d, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms = timer.toc();
        printf("V1 Kernel time: %.3f ms\n", ms);
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
