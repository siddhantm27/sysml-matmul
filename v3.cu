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

// Warp-specialized kernel with bank-conflict-free shared memory layout
#define BLK_M 128
#define BLK_N 128
#define BLK_K 16
#define THR_M 8
#define THR_N 8
#define NUM_THREADS 256

// Bank-conflict-free padding
#define SMEM_STRIDE_A (BLK_K + 1)
#define SMEM_STRIDE_B (BLK_N + 1)

// Warp shuffle for intra-warp communication
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized kernel with bank-conflict-free access pattern
__global__ void matmul_kernel_optimized(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N)
{
    // Bank-conflict-free shared memory layout
    __shared__ float smem_A[BLK_M][SMEM_STRIDE_A];
    __shared__ float smem_B[BLK_K][SMEM_STRIDE_B];
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Block tile coordinates
    const int block_row = blockIdx.y * BLK_M;
    const int block_col = blockIdx.x * BLK_N;
    
    // Thread's output tile
    const int thread_row_base = (tid / 16) * THR_M;
    const int thread_col_base = (tid % 16) * THR_N;
    
    // Registers for accumulation and temporary values
    float reg_C[THR_M][THR_N];
    float reg_A[THR_M];
    float reg_B[THR_N];
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < THR_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THR_N; ++j) {
            reg_C[i][j] = 0.0f;
        }
    }
    
    // Main loop over K dimension
    for (int k_base = 0; k_base < N; k_base += BLK_K) {
        // Cooperatively load A tile into shared memory
        // Each thread loads multiple elements using strided access
        #pragma unroll
        for (int load_idx = tid; load_idx < BLK_M * BLK_K; load_idx += num_threads) {
            int sm_row = load_idx / BLK_K;
            int sm_col = load_idx % BLK_K;
            int gm_row = block_row + sm_row;
            int gm_col = k_base + sm_col;
            
            smem_A[sm_row][sm_col] = (gm_row < N && gm_col < N) ? 
                                     A[gm_row * N + gm_col] : 0.0f;
        }
        
        // Cooperatively load B tile into shared memory
        #pragma unroll
        for (int load_idx = tid; load_idx < BLK_K * BLK_N; load_idx += num_threads) {
            int sm_row = load_idx / BLK_N;
            int sm_col = load_idx % BLK_N;
            int gm_row = k_base + sm_row;
            int gm_col = block_col + sm_col;
            
            smem_B[sm_row][sm_col] = (gm_row < N && gm_col < N) ? 
                                     B[gm_row * N + gm_col] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute on the loaded tile
        #pragma unroll
        for (int k = 0; k < BLK_K; ++k) {
            // Load A values into registers
            #pragma unroll
            for (int i = 0; i < THR_M; ++i) {
                int a_row = thread_row_base + i;
                reg_A[i] = smem_A[a_row][k];
            }
            
            // Load B values into registers
            #pragma unroll
            for (int j = 0; j < THR_N; ++j) {
                int b_col = thread_col_base + j;
                reg_B[j] = smem_B[k][b_col];
            }
            
            // Compute outer product and accumulate
            #pragma unroll
            for (int i = 0; i < THR_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THR_N; ++j) {
                    reg_C[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory with bounds checking
    #pragma unroll
    for (int i = 0; i < THR_M; ++i) {
        int c_row = block_row + thread_row_base + i;
        if (c_row >= N) continue;
        
        #pragma unroll
        for (int j = 0; j < THR_N; ++j) {
            int c_col = block_col + thread_col_base + j;
            if (c_col < N) {
                C[c_row * N + c_col] = reg_C[i][j];
            }
        }
    }
}

// Alternative kernel with different tile sizes for different problem sizes
__global__ void matmul_kernel_adaptive(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N)
{
    // Smaller tiles for better occupancy on smaller matrices
    constexpr int TILE = 64;
    constexpr int REG_TILE = 4;
    
    __shared__ float smem_A[TILE][TILE + 1];
    __shared__ float smem_B[TILE][TILE + 1];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int block_row = blockIdx.y * TILE;
    const int block_col = blockIdx.x * TILE;
    
    const int thread_row = (tid / 8) * REG_TILE;
    const int thread_col = (tid % 8) * REG_TILE;
    
    float acc[REG_TILE][REG_TILE] = {0.0f};
    
    for (int k_base = 0; k_base < N; k_base += TILE) {
        // Load tiles with coalesced access
        #pragma unroll 4
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            int row = i / TILE;
            int col = i % TILE;
            
            int gm_row_a = block_row + row;
            int gm_col_a = k_base + col;
            smem_A[row][col] = (gm_row_a < N && gm_col_a < N) ? 
                               A[gm_row_a * N + gm_col_a] : 0.0f;
            
            int gm_row_b = k_base + row;
            int gm_col_b = block_col + col;
            smem_B[row][col] = (gm_row_b < N && gm_col_b < N) ? 
                               B[gm_row_b * N + gm_col_b] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute with register tiling
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a_reg[REG_TILE];
            float b_reg[REG_TILE];
            
            #pragma unroll
            for (int i = 0; i < REG_TILE; ++i) {
                a_reg[i] = smem_A[thread_row + i][k];
            }
            
            #pragma unroll
            for (int j = 0; j < REG_TILE; ++j) {
                b_reg[j] = smem_B[k][thread_col + j];
            }
            
            #pragma unroll
            for (int i = 0; i < REG_TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_TILE; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int i = 0; i < REG_TILE; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE; ++j) {
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

    // Choose kernel based on problem size
    if (N >= 4096) {
        // Use larger tiles for big matrices
        dim3 block(NUM_THREADS);
        dim3 grid((N + BLK_N - 1) / BLK_N, (N + BLK_M - 1) / BLK_M);
        matmul_kernel_optimized<<<grid, block>>>(A_d, B_d, C_d, N);
    } else {
        // Use smaller tiles for better occupancy
        dim3 block(256);
        dim3 grid((N + 63) / 64, (N + 63) / 64);
        matmul_kernel_adaptive<<<grid, block>>>(A_d, B_d, C_d, N);
    }
    
    CUDA_CHECK(cudaGetLastError());
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
