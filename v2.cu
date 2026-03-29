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

// Double-buffered kernel with B transpose optimization
#define TILE_M 128
#define TILE_N 128
#define TILE_K 8
#define REG_M 8
#define REG_N 8
#define THREADS 256

// Transpose B on-the-fly for better memory access pattern
__global__ void transpose_kernel(const float* __restrict__ B,
                                float* __restrict__ B_T,
                                int N)
{
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = B[y * N + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < N && y < N) {
        B_T[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Double-buffered matmul with prefetching
__global__ void matmul_kernel_doublebuf(const float* __restrict__ A,
                                        const float* __restrict__ B_T,
                                        float* __restrict__ C,
                                        int N)
{
    // Double buffering: two sets of shared memory
    __shared__ float smem_A[2][TILE_M * TILE_K];
    __shared__ float smem_B[2][TILE_N * TILE_K];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Output tile position
    const int row_base = blockIdx.y * TILE_M;
    const int col_base = blockIdx.x * TILE_N;
    
    // Thread-local position within tile
    const int thread_row = (tid / 16) * REG_M;
    const int thread_col = (tid % 16) * REG_N;
    
    const int out_row = row_base + thread_row;
    const int out_col = col_base + thread_col;
    
    // Register accumulation
    float acc[REG_M][REG_N];
    #pragma unroll
    for (int i = 0; i < REG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    float reg_A[REG_M];
    float reg_B[REG_N];
    
    int write_buf = 0;
    int read_buf = 0;
    
    // Prefetch first tile
    int k_tile = 0;
    if (k_tile < N) {
        // Load A tile
        #pragma unroll
        for (int i = 0; i < (TILE_M * TILE_K) / THREADS; ++i) {
            int idx = tid + i * THREADS;
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            if (row < TILE_M && (row_base + row) < N && (k_tile + col) < N) {
                smem_A[write_buf][row * TILE_K + col] = A[(row_base + row) * N + k_tile + col];
            }
        }
        
        // Load B_T tile (B_T is already transposed, so rows become columns)
        #pragma unroll
        for (int i = 0; i < (TILE_N * TILE_K) / THREADS; ++i) {
            int idx = tid + i * THREADS;
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            if (row < TILE_N && (col_base + row) < N && (k_tile + col) < N) {
                smem_B[write_buf][row * TILE_K + col] = B_T[(col_base + row) * N + k_tile + col];
            }
        }
    }
    
    __syncthreads();
    
    // Main computation loop with double buffering
    for (k_tile = 0; k_tile < N; k_tile += TILE_K) {
        read_buf = write_buf;
        write_buf = 1 - write_buf;
        
        // Prefetch next tile while computing current
        if (k_tile + TILE_K < N) {
            // Load A tile for next iteration
            #pragma unroll
            for (int i = 0; i < (TILE_M * TILE_K) / THREADS; ++i) {
                int idx = tid + i * THREADS;
                int row = idx / TILE_K;
                int col = idx % TILE_K;
                if (row < TILE_M && (row_base + row) < N && (k_tile + TILE_K + col) < N) {
                    smem_A[write_buf][row * TILE_K + col] = A[(row_base + row) * N + k_tile + TILE_K + col];
                }
            }
            
            // Load B_T tile for next iteration
            #pragma unroll
            for (int i = 0; i < (TILE_N * TILE_K) / THREADS; ++i) {
                int idx = tid + i * THREADS;
                int row = idx / TILE_K;
                int col = idx % TILE_K;
                if (row < TILE_N && (col_base + row) < N && (k_tile + TILE_K + col) < N) {
                    smem_B[write_buf][row * TILE_K + col] = B_T[(col_base + row) * N + k_tile + TILE_K + col];
                }
            }
        }
        
        // Compute using current tile
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            // Load A strip
            #pragma unroll
            for (int i = 0; i < REG_M; ++i) {
                int a_row = thread_row + i;
                if (a_row < TILE_M) {
                    reg_A[i] = smem_A[read_buf][a_row * TILE_K + k];
                }
            }
            
            // Load B strip
            #pragma unroll
            for (int j = 0; j < REG_N; ++j) {
                int b_row = thread_col + j;
                if (b_row < TILE_N) {
                    reg_B[j] = smem_B[read_buf][b_row * TILE_K + k];
                }
            }
            
            // Accumulate outer product
            #pragma unroll
            for (int i = 0; i < REG_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_N; ++j) {
                    acc[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int i = 0; i < REG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_N; ++j) {
            int c_row = out_row + i;
            int c_col = out_col + j;
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
    float *A_d, *B_d, *B_T_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_T_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    // ── Transfer inputs to device ─────────────────────────────
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    // ── Transpose B ───────────────────────────────────────────
    {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (N + 31) / 32);
        transpose_kernel<<<grid, block>>>(B_d, B_T_d, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Launch matmul kernel ──────────────────────────────────
    {
        dim3 block(THREADS);
        dim3 grid((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);

        matmul_kernel_doublebuf<<<grid, block>>>(A_d, B_T_d, C_d, N);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Copy result back to host ──────────────────────────────
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    // ── Free device memory ────────────────────────────────────
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(B_T_d));
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
