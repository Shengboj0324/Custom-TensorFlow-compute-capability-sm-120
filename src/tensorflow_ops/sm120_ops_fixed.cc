/*
 * TensorFlow Custom Operations for sm_120 Optimizations - FIXED VERSION
 * 
 * This file implements consolidated TensorFlow operations that leverage sm_120 specific
 * CUDA kernels for optimal performance on RTX 50-series GPUs.
 * 
 * This is the definitive implementation combining all SM120 operations.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/gpu_device_functions.h"

#if GOOGLE_CUDA
extern "C" {
#include "cuda_kernels/sm120_c_interface.h"
}
#include "sm120_stream_utils.h"
#endif

#include <cstdlib>

namespace tensorflow {

// Helper macro for CUDA error checking
#define SM120_CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        LOG(ERROR) << "CUDA error: " << cudaGetErrorString(error); \
    } \
} while(0)

// ============================================================================
// Helper Functions
// ============================================================================

static SM120DataType GetSM120DataType(DataType tf_dtype) {
    switch (tf_dtype) {
        case DT_FLOAT:
            return SM120_DTYPE_FLOAT32;
        case DT_HALF:
            return SM120_DTYPE_FLOAT16;
        default:
            return SM120_DTYPE_FLOAT32;
    }
}

static cudaStream_t GetCudaStream(OpKernelContext* context) {
    // Get CUDA stream from TensorFlow context
    auto* stream = context->op_device_context()->stream();
    return stream ? stream->implementation()->GpuStreamHack() : 0;
}

// Security: Check if ops should fallback in training mode when gradients are not implemented
inline bool sm120_safe_training_mode() {
    static bool checked = false;
    static bool enabled = false;
    
    if (!checked) {
        const char* env_var = std::getenv("SM120_SAFE_TRAINING");
        enabled = (env_var == nullptr || std::string(env_var) != "0"); // Default to safe mode
        checked = true;
    }
    
    return enabled;
}

// ============================================================================
// SM120 Optimized Matrix Multiplication Operation
// ============================================================================

REGISTER_OP("SM120MatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {float, half}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle a, b;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

        shape_inference::DimensionHandle a_rows = c->Dim(a, 0);
        shape_inference::DimensionHandle a_cols = c->Dim(a, 1);
        shape_inference::DimensionHandle b_rows = c->Dim(b, 0);
        shape_inference::DimensionHandle b_cols = c->Dim(b, 1);

        shape_inference::DimensionHandle merged;
        TF_RETURN_IF_ERROR(c->Merge(a_cols, b_rows, &merged));

        c->set_output(0, c->Matrix(a_rows, b_cols));
        return Status::OK();
    });

template <typename T>
class SM120MatMulOp : public OpKernel {
 public:
  explicit SM120MatMulOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* context) override {
#if GOOGLE_CUDA
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);

    OP_REQUIRES(context, a.dims() == 2,
                errors::InvalidArgument("Matrix A must be 2-dimensional"));
    OP_REQUIRES(context, b.dims() == 2,
                errors::InvalidArgument("Matrix B must be 2-dimensional"));

    int64 a_rows = transpose_a_ ? a.dim_size(1) : a.dim_size(0);
    int64 a_cols = transpose_a_ ? a.dim_size(0) : a.dim_size(1);
    int64 b_rows = transpose_b_ ? b.dim_size(1) : b.dim_size(0);
    int64 b_cols = transpose_b_ ? b.dim_size(0) : b.dim_size(1);

    OP_REQUIRES(context, a_cols == b_rows,
                errors::InvalidArgument("Matrix dimensions must be compatible"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({a_rows, b_cols}), &output));

    auto* stream = context->op_device_context()->stream();
    cudaStream_t cuda_stream = stream->parent()->implementation()->GpuStreamHack();

    SM120DataType dtype = GetSM120DataType(DataTypeOf<T>::value);

    cudaError_t result = sm120_launch_matmul(
        a.flat<T>().data(),
        b.flat<T>().data(),
        output->flat<T>().data(),
        a_rows, a_cols, b_rows, b_cols,
        transpose_a_, transpose_b_,
        dtype, cuda_stream);

    OP_REQUIRES(context, result == cudaSuccess,
                errors::Internal("SM120MatMul failed: ", cudaGetErrorString(result)));
#else
    context->CtxFailure(errors::Unimplemented("SM120MatMul requires CUDA support"));
#endif
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};

REGISTER_KERNEL_BUILDER(Name("SM120MatMul").Device(DEVICE_GPU).TypeConstraint<float>("T"), SM120MatMulOp<float>);
REGISTER_KERNEL_BUILDER(Name("SM120MatMul").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), SM120MatMulOp<Eigen::half>);

// ============================================================================
// SM120 Flash Attention Operation
// ============================================================================

REGISTER_OP("SM120FlashAttention")
    .Input("queries: T")
    .Input("keys: T")
    .Input("values: T")
    .Output("output: T")
    .Output("attention_weights: T")
    .Attr("T: {float, half}")
    .Attr("scale: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle queries, keys, values;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &queries));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &keys));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &values));

        c->set_output(0, queries);
        c->set_output(1, c->Matrix(c->Dim(queries, 2), c->Dim(keys, 2)));
        return Status::OK();
    });

template <typename T>
class SM120FlashAttentionOp : public OpKernel {
 public:
  explicit SM120FlashAttentionOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
  }

  void Compute(OpKernelContext* context) override {
#if GOOGLE_CUDA
    const Tensor& queries = context->input(0);
    const Tensor& keys = context->input(1);
    const Tensor& values = context->input(2);

    OP_REQUIRES(context, queries.dims() == 4,
                errors::InvalidArgument("Queries must be 4-dimensional [batch, heads, seq_len, head_dim]"));

    int batch_size = queries.dim_size(0);
    int num_heads = queries.dim_size(1);
    int seq_len = queries.dim_size(2);
    int head_dim = queries.dim_size(3);

    Tensor* output = nullptr;
    Tensor* attention_weights = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, queries.shape(), &output));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({seq_len, seq_len}), &attention_weights));

    auto* stream = context->op_device_context()->stream();
    cudaStream_t cuda_stream = stream->parent()->implementation()->GpuStreamHack();

    SM120DataType dtype = GetSM120DataType(DataTypeOf<T>::value);

    cudaError_t result = sm120_launch_flash_attention(
        queries.flat<T>().data(),
        keys.flat<T>().data(),
        values.flat<T>().data(),
        output->flat<T>().data(),
        attention_weights->flat<T>().data(),
        batch_size, num_heads, seq_len, head_dim,
        scale_, dtype, cuda_stream);

    OP_REQUIRES(context, result == cudaSuccess,
                errors::Internal("SM120FlashAttention failed: ", cudaGetErrorString(result)));
#else
    context->CtxFailure(errors::Unimplemented("SM120FlashAttention requires CUDA support"));
#endif
  }

 private:
  float scale_;
};

REGISTER_KERNEL_BUILDER(Name("SM120FlashAttention").Device(DEVICE_GPU).TypeConstraint<float>("T"), SM120FlashAttentionOp<float>);
REGISTER_KERNEL_BUILDER(Name("SM120FlashAttention").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), SM120FlashAttentionOp<Eigen::half>);

// ============================================================================
// SM120 Batch Normalization Operation
// ============================================================================

REGISTER_OP("SM120BatchNormalization")
    .Input("input: T")
    .Input("scale: T")
    .Input("offset: T")
    .Input("mean: T")
    .Input("variance: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("epsilon: float = 1e-5")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

template <typename T>
class SM120BatchNormalizationOp : public OpKernel {
 public:
  explicit SM120BatchNormalizationOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* context) override {
#if GOOGLE_CUDA
    const Tensor& input = context->input(0);
    const Tensor& scale = context->input(1);
    const Tensor& offset = context->input(2);
    const Tensor& mean = context->input(3);
    const Tensor& variance = context->input(4);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    auto* stream = context->op_device_context()->stream();
    cudaStream_t cuda_stream = stream->parent()->implementation()->GpuStreamHack();

    SM120DataType dtype = GetSM120DataType(DataTypeOf<T>::value);

    int64 batch_size = input.dim_size(0);
    int64 channels = input.dim_size(input.dims() - 1);
    int64 spatial_size = input.NumElements() / (batch_size * channels);

    cudaError_t result = sm120_launch_batch_norm(
        input.flat<T>().data(),
        scale.flat<T>().data(),
        offset.flat<T>().data(),
        mean.flat<T>().data(),
        variance.flat<T>().data(),
        output->flat<T>().data(),
        batch_size, channels, spatial_size,
        epsilon_, dtype, cuda_stream);

    OP_REQUIRES(context, result == cudaSuccess,
                errors::Internal("SM120BatchNormalization failed: ", cudaGetErrorString(result)));
#else
    context->CtxFailure(errors::Unimplemented("SM120BatchNormalization requires CUDA support"));
#endif
  }

 private:
  float epsilon_;
};

REGISTER_KERNEL_BUILDER(Name("SM120BatchNormalization").Device(DEVICE_GPU).TypeConstraint<float>("T"), SM120BatchNormalizationOp<float>);
REGISTER_KERNEL_BUILDER(Name("SM120BatchNormalization").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), SM120BatchNormalizationOp<Eigen::half>);

// ============================================================================
// SM120 Convolution Operation
// ============================================================================

REGISTER_OP("SM120Conv2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("strides: list(int)")
    .Attr("padding: string")
    .Attr("data_format: string = 'NHWC'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Simplified shape inference - actual implementation would be more complex
        c->set_output(0, c->input(0));
        return Status::OK();
    });

template <typename T>
class SM120Conv2DOp : public OpKernel {
 public:
  explicit SM120Conv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_));
  }

  void Compute(OpKernelContext* context) override {
#if GOOGLE_CUDA
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("Input must be 4-dimensional"));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("Filter must be 4-dimensional"));

    // For now, allocate output with same shape as input (simplified)
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    auto* stream = context->op_device_context()->stream();
    cudaStream_t cuda_stream = stream->parent()->implementation()->GpuStreamHack();

    SM120DataType dtype = GetSM120DataType(DataTypeOf<T>::value);

    int batch_size = input.dim_size(0);
    int input_height = input.dim_size(1);
    int input_width = input.dim_size(2);
    int input_channels = input.dim_size(3);

    cudaError_t result = sm120_launch_conv2d(
        input.flat<T>().data(),
        filter.flat<T>().data(),
        output->flat<T>().data(),
        batch_size, input_height, input_width, input_channels,
        filter.dim_size(0), filter.dim_size(1), filter.dim_size(3),
        strides_[1], strides_[2],
        dtype, cuda_stream);

    OP_REQUIRES(context, result == cudaSuccess,
                errors::Internal("SM120Conv2D failed: ", cudaGetErrorString(result)));
#else
    context->CtxFailure(errors::Unimplemented("SM120Conv2D requires CUDA support"));
#endif
  }

 private:
  std::vector<int32> strides_;
  string padding_;
  string data_format_;
};

REGISTER_KERNEL_BUILDER(Name("SM120Conv2D").Device(DEVICE_GPU).TypeConstraint<float>("T"), SM120Conv2DOp<float>);
REGISTER_KERNEL_BUILDER(Name("SM120Conv2D").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"), SM120Conv2DOp<Eigen::half>);

} // namespace tensorflow
