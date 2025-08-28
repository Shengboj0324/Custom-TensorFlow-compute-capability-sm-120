/*
 * Pure CUDA Kernels for SM120 Architecture
 * 
 * This file contains self-contained CUDA kernels with no external dependencies.
 * All kernels are designed specifically for SM120 (Compute Capability 12.0).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

// ============================================================================
// SM120 Matrix Multiplication Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4) sm120_matmul_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta) {
    
    // Use cooperative groups for advanced warp coordination
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float accumulator = 0.0f;
        
        // Vectorized memory access for SM120
        for (int k = 0; k < K; k += 4) {
            if (k + 3 < K) {
                // Load 4 elements at once for better memory throughput
                float4 a_vec = reinterpret_cast<const float4*>(&A[row * K + k])[0];
                float4 b_vec = reinterpret_cast<const float4*>(&B[k * N + col])[0];
                
                accumulator += a_vec.x * b_vec.x;
                accumulator += a_vec.y * b_vec.y;
                accumulator += a_vec.z * b_vec.z;
                accumulator += a_vec.w * b_vec.w;
            } else {
                // Handle remaining elements
                for (int kk = k; kk < K; kk++) {
                    accumulator += static_cast<float>(A[row * K + kk]) * 
                                 static_cast<float>(B[kk * N + col]);
                }
            }
        }
        
        // Apply alpha and beta scaling
        float result = alpha * accumulator;
        if (beta != 0.0f) {
            result += beta * static_cast<float>(C[row * N + col]);
        }
        
        C[row * N + col] = static_cast<T>(result);
    }
}

// ============================================================================
// SM120 Convolution Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(512, 2) sm120_conv2d_kernel(
    const T* __restrict__ input,
    const T* __restrict__ filter,
    T* __restrict__ output,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_height, int output_width, int output_channels,
    int filter_height, int filter_width,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    
    int batch_idx = blockIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = threadIdx.z;
    
    if (batch_idx < batch_size && out_y < output_height && 
        out_x < output_width && out_c < output_channels) {
        
        float accumulator = 0.0f;
        
        // Convolution computation
        for (int fh = 0; fh < filter_height; fh++) {
            for (int fw = 0; fw < filter_width; fw++) {
                int in_y = out_y * stride_h - pad_h + fh;
                int in_x = out_x * stride_w - pad_w + fw;
                
                if (in_y >= 0 && in_y < input_height && 
                    in_x >= 0 && in_x < input_width) {
                    
                    for (int in_c = 0; in_c < input_channels; in_c++) {
                        int input_idx = batch_idx * input_height * input_width * input_channels +
                                      in_y * input_width * input_channels +
                                      in_x * input_channels + in_c;
                        
                        int filter_idx = out_c * filter_height * filter_width * input_channels +
                                       fh * filter_width * input_channels +
                                       fw * input_channels + in_c;
                        
                        accumulator += static_cast<float>(input[input_idx]) * 
                                     static_cast<float>(filter[filter_idx]);
                    }
                }
            }
        }
        
        int output_idx = batch_idx * output_height * output_width * output_channels +
                        out_y * output_width * output_channels +
                        out_x * output_channels + out_c;
        
        output[output_idx] = static_cast<T>(accumulator);
    }
}

// ============================================================================
// SM120 Activation Functions Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(1024, 2) sm120_activation_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int size,
    int activation_type) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = static_cast<float>(input[idx]);
        float result;
        
        switch (activation_type) {
            case 0: // RELU
                result = fmaxf(0.0f, x);
                break;
            case 1: // LEAKY_RELU
                result = x > 0.0f ? x : 0.01f * x;
                break;
            case 2: // ELU
                result = x > 0.0f ? x : (expf(x) - 1.0f);
                break;
            case 3: // SELU
                result = x > 0.0f ? 1.0507f * x : 1.0507f * 1.67326f * (expf(x) - 1.0f);
                break;
            case 4: // GELU
                result = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
                break;
            case 5: // SWISH
                result = x / (1.0f + expf(-x));
                break;
            case 6: // MISH
                result = x * tanhf(logf(1.0f + expf(x)));
                break;
            case 7: // TANH
                result = tanhf(x);
                break;
            case 8: // SIGMOID
                result = 1.0f / (1.0f + expf(-x));
                break;
            default:
                result = x;
                break;
        }
        
        output[idx] = static_cast<T>(result);
    }
}

// ============================================================================
// SM120 Reduction Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(1024, 1) sm120_reduction_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int size,
    int reduction_type) {
    
    typedef cub::BlockReduce<float, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float thread_data = 0.0f;
    
    // Load data
    if (tid < size) {
        thread_data = static_cast<float>(input[tid]);
    }
    
    // Perform reduction
    float aggregate;
    switch (reduction_type) {
        case 0: // SUM
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
            break;
        case 1: // MEAN
            aggregate = BlockReduce(temp_storage).Sum(thread_data) / size;
            break;
        case 2: // MAX
            aggregate = BlockReduce(temp_storage).Reduce(thread_data, cub::Max());
            break;
        case 3: // MIN
            aggregate = BlockReduce(temp_storage).Reduce(thread_data, cub::Min());
            break;
        default:
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
            break;
    }
    
    // Write result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = static_cast<T>(aggregate);
    }
}

// ============================================================================
// SM120 Transpose Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4) sm120_transpose_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int rows, int cols) {

    __shared__ T tile[32][33]; // +1 to avoid bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load data into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;

    // Write transposed data
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// SM120 Layer Normalization Kernel with Dynamic Shared Memory
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 2) sm120_layer_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    T* __restrict__ mean,
    T* __restrict__ variance,
    int batch_size, int feature_size,
    float epsilon) {

    // Use dynamic shared memory for better flexibility
    extern __shared__ float sdata[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const T* batch_input = input + batch_idx * feature_size;
    T* batch_output = output + batch_idx * feature_size;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        sum += static_cast<float>(batch_input[i]);
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float batch_mean = sdata[0] / feature_size;
    if (tid == 0 && mean != nullptr) {
        mean[batch_idx] = static_cast<T>(batch_mean);
    }
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float diff = static_cast<float>(batch_input[i]) - batch_mean;
        var_sum += diff * diff;
    }

    sdata[tid] = var_sum;
    __syncthreads();

    // Reduce variance sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float batch_variance = sdata[0] / feature_size;
    if (tid == 0 && variance != nullptr) {
        variance[batch_idx] = static_cast<T>(batch_variance);
    }
    __syncthreads();

    // Apply normalization
    float inv_std = rsqrtf(batch_variance + epsilon);
    for (int i = tid; i < feature_size; i += blockDim.x) {
        float normalized = (static_cast<float>(batch_input[i]) - batch_mean) * inv_std;
        float scaled = normalized;

        if (gamma != nullptr) {
            scaled *= static_cast<float>(gamma[i]);
        }
        if (beta != nullptr) {
            scaled += static_cast<float>(beta[i]);
        }

        batch_output[i] = static_cast<T>(scaled);
    }
}

// ============================================================================
// SM120 Attention Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4) sm120_attention_kernel(
    const T* __restrict__ queries,
    const T* __restrict__ keys,
    const T* __restrict__ values,
    T* __restrict__ output,
    float* __restrict__ attention_weights,
    int batch_size, int seq_len, int head_dim,
    float scale) {

    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int head_idx = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || head_idx >= head_dim) {
        return;
    }

    __shared__ float scores[256];
    __shared__ float max_score;
    __shared__ float sum_exp;

    const T* query = queries + batch_idx * seq_len * head_dim + seq_idx * head_dim;
    const T* key_base = keys + batch_idx * seq_len * head_dim;
    const T* value_base = values + batch_idx * seq_len * head_dim;

    // Compute attention scores
    float max_val = -INFINITY;
    for (int k = 0; k < seq_len; k++) {
        const T* key = key_base + k * head_dim;
        float score = 0.0f;

        for (int d = head_idx; d < head_dim; d += blockDim.x) {
            score += static_cast<float>(query[d]) * static_cast<float>(key[d]);
        }

        // Reduce score across threads
        score *= scale;
        scores[k] = score;
        max_val = fmaxf(max_val, score);
    }

    // Find global max
    if (threadIdx.x == 0) {
        max_score = max_val;
    }
    __syncthreads();

    // Compute softmax
    float sum = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        scores[k] = expf(scores[k] - max_score);
        sum += scores[k];
    }

    if (threadIdx.x == 0) {
        sum_exp = sum;
    }
    __syncthreads();

    // Normalize and store attention weights
    for (int k = 0; k < seq_len; k++) {
        scores[k] /= sum_exp;
        if (attention_weights != nullptr && head_idx == 0) {
            attention_weights[batch_idx * seq_len * seq_len + seq_idx * seq_len + k] = scores[k];
        }
    }

    // Compute output
    T* out = output + batch_idx * seq_len * head_dim + seq_idx * head_dim;
    for (int d = head_idx; d < head_dim; d += blockDim.x) {
        float result = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            const T* value = value_base + k * head_dim;
            result += scores[k] * static_cast<float>(value[d]);
        }
        out[d] = static_cast<T>(result);
    }
}

// ============================================================================
// SM120 Batch Normalization Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 2) sm120_batch_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ scale,
    const T* __restrict__ offset,
    const T* __restrict__ estimated_mean,
    const T* __restrict__ estimated_variance,
    T* __restrict__ output,
    int batch_size, int height, int width, int channels,
    float epsilon) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * height * width * channels;

    if (idx < total_elements) {
        int c = idx % channels;

        float x = static_cast<float>(input[idx]);
        float mean = static_cast<float>(estimated_mean[c]);
        float variance = static_cast<float>(estimated_variance[c]);
        float scale_val = static_cast<float>(scale[c]);
        float offset_val = static_cast<float>(offset[c]);

        // Batch normalization formula: (x - mean) / sqrt(variance + epsilon) * scale + offset
        float normalized = (x - mean) * rsqrtf(variance + epsilon);
        float result = normalized * scale_val + offset_val;

        output[idx] = static_cast<T>(result);
    }
}

// ============================================================================
// SM120 Softmax Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 2) sm120_softmax_kernel(
    const T* __restrict__ logits,
    T* __restrict__ output,
    int outer_size, int axis_size, int inner_size) {

    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Use dynamic shared memory for better flexibility
    extern __shared__ float shared_mem[];
    float* sdata = shared_mem;
    float* max_val = &shared_mem[blockDim.x];
    float* sum_exp = &shared_mem[blockDim.x + 1];

    int tid = threadIdx.x;
    int base_idx = outer_idx * axis_size * inner_size + inner_idx;

    // Find maximum value for numerical stability
    float thread_max = -INFINITY;
    for (int i = tid; i < axis_size; i += blockDim.x) {
        int idx = base_idx + i * inner_size;
        float val = static_cast<float>(logits[idx]);
        thread_max = fmaxf(thread_max, val);
    }

    sdata[tid] = thread_max;
    __syncthreads();

    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_val = sdata[0];
    }
    __syncthreads();

    // Compute sum of exponentials
    float thread_sum = 0.0f;
    for (int i = tid; i < axis_size; i += blockDim.x) {
        int idx = base_idx + i * inner_size;
        float val = static_cast<float>(logits[idx]);
        thread_sum += expf(val - *max_val);
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    // Reduce to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *sum_exp = sdata[0];
    }
    __syncthreads();

    // Compute softmax
    for (int i = tid; i < axis_size; i += blockDim.x) {
        int idx = base_idx + i * inner_size;
        float val = static_cast<float>(logits[idx]);
        float softmax_val = expf(val - *max_val) / *sum_exp;
        output[idx] = static_cast<T>(softmax_val);
    }
}

// ============================================================================
// SM120 Embedding Lookup Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4) sm120_embedding_lookup_kernel(
    const int* __restrict__ ids,
    const T* __restrict__ params,
    T* __restrict__ output,
    int num_ids, int vocab_size, int embed_dim) {

    int id_idx = blockIdx.x;
    int embed_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (id_idx >= num_ids || embed_idx >= embed_dim) return;

    int id = ids[id_idx];

    // Bounds checking for vocabulary
    if (id >= 0 && id < vocab_size) {
        int param_idx = id * embed_dim + embed_idx;
        int output_idx = id_idx * embed_dim + embed_idx;
        output[output_idx] = params[param_idx];
    } else {
        // Out of bounds ID - set to zero
        int output_idx = id_idx * embed_dim + embed_idx;
        output[output_idx] = static_cast<T>(0.0f);
    }
}
