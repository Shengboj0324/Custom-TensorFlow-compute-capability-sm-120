/*
 * TensorFlow Operations for SM120 CUDA Kernels
 * 
 * This file provides TensorFlow custom operations that call the C interface.
 * This layer handles all TensorFlow-specific code and data type conversions.
 */

// TensorFlow includes
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/default/logging.h"

// CUDA includes
#include <cuda_runtime.h>

// Our C interface
extern "C" {
#include "../cuda_kernels/sm120_c_interface.h"
}

using namespace tensorflow;

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

// ============================================================================
// SM120 Matrix Multiplication Operation
// ============================================================================

REGISTER_OP("SM120MatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("c: T")
    .Attr("T: {float, half}")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        auto a_shape = c->input(0);
        auto b_shape = c->input(1);
        
        // Basic matrix multiplication shape inference
        auto m = c->Dim(a_shape, 0);
        auto n = c->Dim(b_shape, 1);
        
        c->set_output(0, c->Matrix(m, n));
        return Status::OK();
    });

class SM120MatMulOp : public OpKernel {
public:
    explicit SM120MatMulOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
        OP_REQUIRES_OK(context, context->GetAttr("beta", &beta_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& a = context->input(0);
        const Tensor& b = context->input(1);
        
        // Get dimensions
        int M = a.dim_size(0);
        int K = a.dim_size(1);
        int N = b.dim_size(1);
        
        // Allocate output
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({M, N}), &output));
        
        // Launch kernel
        SM120DataType dtype = GetSM120DataType(a.dtype());
        cudaStream_t stream = GetCudaStream(context);
        
        cudaError_t error = sm120_launch_matmul(
            a.tensor_data().data(),
            b.tensor_data().data(),
            output->tensor_data().data(),
            M, N, K,
            alpha_, beta_,
            dtype, stream);
        
        OP_REQUIRES(context, error == cudaSuccess, 
                   errors::Internal("SM120MatMul failed: ", cudaGetErrorString(error)));
    }

private:
    float alpha_, beta_;
};

REGISTER_KERNEL_BUILDER(Name("SM120MatMul").Device(DEVICE_GPU), SM120MatMulOp);

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
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // Simplified shape inference for convolution
        auto input_shape = c->input(0);
        auto filter_shape = c->input(1);
        
        auto batch = c->Dim(input_shape, 0);
        auto out_channels = c->Dim(filter_shape, 0);
        
        // For simplicity, assume same spatial dimensions
        auto height = c->Dim(input_shape, 1);
        auto width = c->Dim(input_shape, 2);
        
        c->set_output(0, c->MakeShape({batch, height, width, out_channels}));
        return Status::OK();
    });

class SM120Conv2DOp : public OpKernel {
public:
    explicit SM120Conv2DOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        const Tensor& filter = context->input(1);
        
        // Get dimensions
        int batch_size = input.dim_size(0);
        int input_height = input.dim_size(1);
        int input_width = input.dim_size(2);
        int input_channels = input.dim_size(3);
        
        int filter_height = filter.dim_size(0);
        int filter_width = filter.dim_size(1);
        int output_channels = filter.dim_size(3);
        
        // Calculate output dimensions (simplified)
        int output_height = input_height;
        int output_width = input_width;
        
        // Allocate output
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, 
            TensorShape({batch_size, output_height, output_width, output_channels}), &output));
        
        // Launch kernel
        SM120DataType dtype = GetSM120DataType(input.dtype());
        cudaStream_t stream = GetCudaStream(context);
        
        cudaError_t error = sm120_launch_conv2d(
            input.tensor_data().data(),
            filter.tensor_data().data(),
            output->tensor_data().data(),
            batch_size,
            input_height, input_width, input_channels,
            output_height, output_width, output_channels,
            filter_height, filter_width,
            strides_[1], strides_[2],  // stride_h, stride_w
            0, 0,  // pad_h, pad_w (simplified)
            dtype, stream);
        
        OP_REQUIRES(context, error == cudaSuccess, 
                   errors::Internal("SM120Conv2D failed: ", cudaGetErrorString(error)));
    }

private:
    std::vector<int32> strides_;
    string padding_;
};

REGISTER_KERNEL_BUILDER(Name("SM120Conv2D").Device(DEVICE_GPU), SM120Conv2DOp);

// ============================================================================
// SM120 Activation Operation
// ============================================================================

REGISTER_OP("SM120Activation")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("activation: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class SM120ActivationOp : public OpKernel {
public:
    explicit SM120ActivationOp(OpKernelConstruction* context) : OpKernel(context) {
        string activation_str;
        OP_REQUIRES_OK(context, context->GetAttr("activation", &activation_str));
        
        // Convert string to enum
        if (activation_str == "relu") {
            activation_type_ = SM120_ACTIVATION_RELU;
        } else if (activation_str == "gelu") {
            activation_type_ = SM120_ACTIVATION_GELU;
        } else if (activation_str == "swish") {
            activation_type_ = SM120_ACTIVATION_SWISH;
        } else {
            activation_type_ = SM120_ACTIVATION_RELU;  // default
        }
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        
        // Allocate output
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
        
        // Launch kernel
        SM120DataType dtype = GetSM120DataType(input.dtype());
        cudaStream_t stream = GetCudaStream(context);
        
        int size = input.NumElements();
        
        cudaError_t error = sm120_launch_activation(
            input.tensor_data().data(),
            output->tensor_data().data(),
            size,
            activation_type_,
            dtype, stream);
        
        OP_REQUIRES(context, error == cudaSuccess, 
                   errors::Internal("SM120Activation failed: ", cudaGetErrorString(error)));
    }

private:
    SM120ActivationType activation_type_;
};

REGISTER_KERNEL_BUILDER(Name("SM120Activation").Device(DEVICE_GPU), SM120ActivationOp);

// ============================================================================
// SM120 Layer Normalization Operation
// ============================================================================

REGISTER_OP("SM120LayerNorm")
    .Input("input: T")
    .Input("gamma: T")
    .Input("beta: T")
    .Output("output: T")
    .Output("mean: T")
    .Output("variance: T")
    .Attr("T: {float, half}")
    .Attr("epsilon: float = 1e-5")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        auto input_shape = c->input(0);
        c->set_output(0, input_shape);  // output same as input

        // mean and variance have shape [batch_size]
        auto batch_dim = c->Dim(input_shape, 0);
        c->set_output(1, c->Vector(batch_dim));
        c->set_output(2, c->Vector(batch_dim));
        return Status::OK();
    });

class SM120LayerNormOp : public OpKernel {
public:
    explicit SM120LayerNormOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        const Tensor& gamma = context->input(1);
        const Tensor& beta = context->input(2);

        int batch_size = input.dim_size(0);
        int feature_size = input.NumElements() / batch_size;

        // Allocate outputs
        Tensor* output = nullptr;
        Tensor* mean = nullptr;
        Tensor* variance = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({batch_size}), &mean));
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({batch_size}), &variance));

        // Launch kernel
        SM120DataType dtype = GetSM120DataType(input.dtype());
        cudaStream_t stream = GetCudaStream(context);

        cudaError_t error = sm120_launch_layer_norm(
            input.tensor_data().data(),
            gamma.tensor_data().data(),
            beta.tensor_data().data(),
            output->tensor_data().data(),
            mean->tensor_data().data(),
            variance->tensor_data().data(),
            batch_size, feature_size,
            epsilon_,
            dtype, stream);

        OP_REQUIRES(context, error == cudaSuccess,
                   errors::Internal("SM120LayerNorm failed: ", cudaGetErrorString(error)));
    }

private:
    float epsilon_;
};

REGISTER_KERNEL_BUILDER(Name("SM120LayerNorm").Device(DEVICE_GPU), SM120LayerNormOp);

// ============================================================================
// SM120 Transpose Operation
// ============================================================================

REGISTER_OP("SM120Transpose")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        auto input_shape = c->input(0);

        // Transpose 2D matrix: [M, N] -> [N, M]
        auto dim0 = c->Dim(input_shape, 0);
        auto dim1 = c->Dim(input_shape, 1);

        c->set_output(0, c->Matrix(dim1, dim0));
        return Status::OK();
    });

class SM120TransposeOp : public OpKernel {
public:
    explicit SM120TransposeOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);

        int rows = input.dim_size(0);
        int cols = input.dim_size(1);

        // Allocate output
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({cols, rows}), &output));

        // Launch kernel
        SM120DataType dtype = GetSM120DataType(input.dtype());
        cudaStream_t stream = GetCudaStream(context);

        cudaError_t error = sm120_launch_transpose(
            input.tensor_data().data(),
            output->tensor_data().data(),
            rows, cols,
            dtype, stream);

        OP_REQUIRES(context, error == cudaSuccess,
                   errors::Internal("SM120Transpose failed: ", cudaGetErrorString(error)));
    }
};

REGISTER_KERNEL_BUILDER(Name("SM120Transpose").Device(DEVICE_GPU), SM120TransposeOp);
