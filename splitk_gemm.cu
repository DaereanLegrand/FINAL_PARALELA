#include <iostream>
#include <chrono>

#include "cutlass/gemm/device/gemm.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

using namespace std;

// Helper function to check for CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t cudaError = call; \
    if(cudaError != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s, line %d\n", cudaGetErrorString(cudaError), __LINE__); \
        return cudaError; \
    } \
} while(0)

// Basic kernel for matrix multiplication
__global__ void matmulKernel(int M, int N, int K,
                             float alpha,
                             float const *A, int lda,
                             float const *B, int ldb,
                             float beta,
                             float *C, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += alpha * A[row * lda + k] * B[k * ldb + col];
        }
        C[row * ldc + col] = beta * C[row * ldc + col] + sum;
    }
}

// Function to perform matrix multiplication without Cutlass
cudaError_t matmul(
    int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc) {

    // Configure kernel launch parameters
    dim3 blockDim(16, 16);  // Adjust as needed based on hardware capabilities
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch the matrix multiplication kernel
    matmulKernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Return success if no errors were encountered
    return cudaSuccess;
}

// Function to perform matrix multiplication using Cutlass
cudaError_t CutlassSgemmNN(
    int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc) {

    // Define type definition for single-precision CUTLASS GEMM
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, ColumnMajor, float, ColumnMajor, float, ColumnMajor>;

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object
    CutlassGemm::Arguments args({ M, N, K }, { A, lda }, { B, ldb }, { C, ldc }, { C, ldc }, { alpha, beta });

    // Launch the CUTLASS GEMM kernel
    cutlass::Status status = gemm_operator(args);

    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    // Return success if no errors were encountered
    return cudaSuccess;
}

// Function to generate random data for linear regression
void generateRandomData(int num_samples, int num_features,
                         std::vector<float> &X, std::vector<float> &y) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::normal_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < num_samples; ++i) {
        X.push_back(1.0);  // Bias term
        for (int j = 1; j < num_features; ++j) {
            X.push_back(dist(eng));
        }
        // Generate target values (ground truth)
        y.push_back(dist(eng));
    }
}

// Function to perform linear regression using Cutlass GEMM
void linearRegressionCutlass(const std::vector<float> &X,
                             const std::vector<float> &y,
                             std::vector<float> &weights) {
    int num_samples = X.size() / (weights.size() - 1);  // Exclude bias term

    // Convert input data to matrices
    float *A, *B, *C;
    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(float) * num_samples * (weights.size() - 1));
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(float) * (weights.size() - 1));
    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(float) * num_samples);

    cudaMemcpy(A, X.data(), sizeof(float) * num_samples * (weights.size() - 1), cudaMemcpyHostToDevice);
    cudaMemcpy(B, weights.data(), sizeof(float) * (weights.size() - 1), cudaMemcpyHostToDevice);

    // Perform matrix multiplication (X * weights)
    CutlassSgemmNN(num_samples, 1, weights.size() - 1, 1.0, A, num_samples, B, weights.size() - 1, 0.0, C, num_samples);

    // Copy the result back to the host
    cudaMemcpy(weights.data(), C, sizeof(float) * num_samples, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

void linearRegression(const std::vector<float> &X,
                             const std::vector<float> &y,
                             std::vector<float> &weights) {
    int num_samples = X.size() / (weights.size() - 1);  // Exclude bias term

    // Convert input data to matrices
    float *A, *B, *C;
    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(float) * num_samples * (weights.size() - 1));
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(float) * (weights.size() - 1));
    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(float) * num_samples);

    cudaMemcpy(A, X.data(), sizeof(float) * num_samples * (weights.size() - 1), cudaMemcpyHostToDevice);
    cudaMemcpy(B, weights.data(), sizeof(float) * (weights.size() - 1), cudaMemcpyHostToDevice);

    // Perform matrix multiplication (X * weights)
    matmul(num_samples, 1, weights.size() - 1, 1.0, A, num_samples, B, weights.size() - 1, 0.0, C, num_samples);

    // Copy the result back to the host
    cudaMemcpy(weights.data(), C, sizeof(float) * num_samples, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Put all the created template variables to create GemmSplitKParallel template variable
using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
                                                       LayoutInputA,
                                                       ElementInputB,
                                                       LayoutInputB,
                                                       ElementOutput,
                                                       LayoutOutput,
                                                       ElementAccumulator,
                                                       MMAOp,
                                                       SmArch,
                                                       ShapeMMAThreadBlock,
                                                       ShapeMMAWarp,
                                                       ShapeMMAOp,
                                                       EpilogueOp>;

int run(int m, int n, int k) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 7) {
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  //
  // Define problem size
  //

  // const int length_m = 2560;
  // const int length_n = 4096;
  // const int length_k = 4096;
  const int length_m = m;
  const int length_n = n ;
  const int length_k = k;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0);  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 16 partitions
  int split_k_slices = 32;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);

  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue>
      gemm_device;

  // Launch device reference gemm kernel
  gemm_device(problem_size,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_c.device_ref(),
              tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();


  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  // std::cout << (passed ? "Passed" : "Failed") << std::endl;

  return (passed ? 0  : -1);
}

int main(int argc, char *argv[]) {
    int num_samples = 1024;
    int num_features = 8;

    float scalars[2] = {1, 0};

    std::vector<float> X;
    std::vector<float> y;
    generateRandomData(num_samples, num_features, X, y);

    std::vector<float> weights(num_features, 1.0);
    std::vector<float> weights2(num_features, 1.0);

    auto start = std::chrono::high_resolution_clock::now();
    linearRegressionCutlass(X, y, weights);
    auto end = std::chrono::high_resolution_clock::now();

    auto start1 = std::chrono::high_resolution_clock::now();
    linearRegression(X, y, weights);
    auto end1 = std::chrono::high_resolution_clock::now();
    // Print the resulting weights
    std::cout << "Resulting Weights: ";
    for (float weight : weights) {
        std::cout << weight << " ";
    }
    std::cout << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);

    auto start2 = std::chrono::high_resolution_clock::now();
    // run(num_samples, num_samples, num_samples);
    auto end2 = std::chrono::high_resolution_clock::now();

    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "Execution Time NORMAL: " << duration.count() << std::endl;
    std::cout << "Execution Time CUTLASS: " << duration1.count() << std::endl;
    std::cout << "Execution time CUTLASS SPLITK: " << duration2.count() << std::endl;

    return 0;
}

