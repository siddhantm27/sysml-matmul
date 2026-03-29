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

// Hierarchical tiling with warp-level cooperation
// BM, BN: thread block output tile
// BK: shared memory tile along K dimension
// WM, WN: warp output tile
// TM, TN: thread output tile (register blocking)

#define BM 128
#define BN 128
#define BK 16
#define WM 64
#define WN 64
#define TM 8
#define TN 8

#define WARP_SIZE 32
#define WARPS_PER_BLOCK ((BM * BN) / (WM * WN))

// Vectorized load using float4
__device__ __forceinline__ void load_smem_A(float* smem, const float* gmem, 
                                            int tid, int row_base, int N) {
    constexpr int ITEMS_PER_THREAD = (BM * BK) / 256;
    
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i += 4) {
        int idx = tid * ITEMS_PER_THREAD + i;
        int smem_row = idx / BK;
        int smem_col = idx % BK;
        
        if (smem_row < BM && smem_col + 3 < BK) {
            int gmem_row = row_base + smem_row;
            int gmem_col = smem_col;
            
            if (gmem_row < N) {
                float4 val = *reinterpret_cast<const float4*>(&gmem[gmem_row * N + gmem_col]);
                *reinterpret_cast<float4*>(&smem[smem_row * BK + smem_col]) = val;
            }
        }
    }
}

__device__ __forceinline__ void load_smem_B(float* smem, const float* gmem,
                                            int tid, int col_base, int N) {
    constexpr int ITEMS_PER_THREAD = (BK * BN) / 256;
    
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i += 4) {
        int idx = tid * ITEMS_PER_THREAD + i;
        int smem_row = idx / BN;
        int smem_col = idx % BN;
        
        if (smem_row < BK && smem_col + 3 < BN) {
            int gmem_row = smem_row;
            int gmem_col = col_base + smem_col;
            
            if (gmem_col < N) {
                float4 val = *reinterpret_cast<const float4*>(&gmem[gmem_row * N + gmem_col]);
                *reinterpret_cast<float4*>(&smem[smem_row * BN + smem_col]) = val;
            }
        }
    }
}

__global__ void matmul_kernel_hierarchical(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N)
{
    __shared__ float smem_A[BM * BK];
    __shared__ float smem_B[BK * BN];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Thread block tile location
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    
    // Warp tile location within block
    const int warp_row = (warp_id / 2) * WM;
    const int warp_col = (warp_id % 2) * WN;
    
    // Thread tile location within warp
    const int thread_row = (lane_id / 8) * TM;
    const int thread_col = (lane_id % 8) * TN;
    
    // Global output position for this thread
    const int out_row = block_row + warp_row + thread_row;
    const int out_col = block_col + warp_col + thread_col;
    
    // Register tiles for accumulation
    float reg_C[TM][TN] = {0.0f};
    float reg_A[TM];
    float reg_B[TN];
    
    // Main loop over K dimension
    for (int k_base = 0; k_base < N; k_base += BK) {
        // Load tiles into shared memory using vectorized loads
        load_smem_A(smem_A, A + block_row * N + k_base, tid, block_row, N);
        load_smem_B(smem_B, B + k_base * N + block_col, tid, block_col, N);
        
        __syncthreads();
        
        // Compute on the tile with register blocking
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load A strip into registers
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                int a_row = warp_row + thread_row + m;
                if (a_row < BM) {
                    reg_A[m] = smem_A[a_row * BK + k];
                }
            }
            
            // Load B strip into registers
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                int b_col = warp_col + thread_col + n;
                if (b_col < BN) {
                    reg_B[n] = smem_B[k * BN + b_col];
                }
            }
            
            // Outer product accumulation
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    reg_C[m][n] += reg_A[m] * reg_B[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int c_row = out_row + m;
            int c_col = out_col + n;
            if (c_row < N && c_col < N) {
                C[c_row * N + c_col] = reg_C[m][n];
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

        matmul_kernel_hierarchical<<<grid, block>>>(A_d, B_d, C_d, N);
        CUDA_CHECK(cudaGetLastError());
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
