// SM120 Gradient Registrations for Custom TensorFlow Operations
// Provides backward propagation support for all SM120 optimized kernels
// Copyright 2024 - TensorFlow SM120 Optimization Project

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "cuda_kernels/sm120_kernel_launcher_fixed.h"
#include "cuda_kernels/sm120_backward_kernels.h"

namespace tensorflow {

// SM120MatMul gradient registration
REGISTER_OP_GRADIENT("SM120MatMul", [](const AttrSlice& attrs,
                                        const FunctionDef& fdef,
                                        const std::vector<NodeDef*>& grad_inputs,
                                        std::vector<NodeDef*>* grad_outputs) {
  bool transpose_a;
  bool transpose_b;
  TF_RETURN_IF_ERROR(GetNodeAttr(fdef.signature().input_arg(0), "transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(GetNodeAttr(fdef.signature().input_arg(0), "transpose_b", &transpose_b));
  
  // grad_output = grad_inputs[0]
  // a = fdef.input(0)
  // b = fdef.input(1)
  
  NodeDef* grad_a = new NodeDef();
  grad_a->set_op("SM120MatMulGradA");
  grad_a->add_input(grad_inputs[0]->name());  // grad_output
  grad_a->add_input(fdef.signature().input_arg(1).name());  // b
  (*grad_a->mutable_attr())["transpose_a"].set_b(transpose_a);
  (*grad_a->mutable_attr())["transpose_b"].set_b(transpose_b);
  (*grad_a->mutable_attr())["T"] = fdef.attr().at("T");
  
  NodeDef* grad_b = new NodeDef();
  grad_b->set_op("SM120MatMulGradB");
  grad_b->add_input(fdef.signature().input_arg(0).name());  // a
  grad_b->add_input(grad_inputs[0]->name());  // grad_output
  (*grad_b->mutable_attr())["transpose_a"].set_b(transpose_a);
  (*grad_b->mutable_attr())["transpose_b"].set_b(transpose_b);
  (*grad_b->mutable_attr())["T"] = fdef.attr().at("T");
  
  grad_outputs->push_back(grad_a);
  grad_outputs->push_back(grad_b);
  
  return Status::OK();
});

// SM120MatMulGradA operation
REGISTER_OP("SM120MatMulGradA")
    .Input("grad_output: T")
    .Input("b: T")
    .Output("grad_a: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {float, half, bfloat16}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle grad_output, b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &grad_output));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));
      
      // grad_a shape = [M, K] where grad_output is [M, N] and b is [K, N]
      shape_inference::DimensionHandle m = c->Dim(grad_output, 0);
      shape_inference::DimensionHandle k = c->Dim(b, 0);
      
      c->set_output(0, c->Matrix(m, k));
      return Status::OK();
    });

// SM120MatMulGradB operation
REGISTER_OP("SM120MatMulGradB")
    .Input("a: T")
    .Input("grad_output: T")
    .Output("grad_b: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {float, half, bfloat16}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle a, grad_output;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &grad_output));
      
      // grad_b shape = [K, N] where a is [M, K] and grad_output is [M, N]
      shape_inference::DimensionHandle k = c->Dim(a, 1);
      shape_inference::DimensionHandle n = c->Dim(grad_output, 1);
      
      c->set_output(0, c->Matrix(k, n));
      return Status::OK();
    });

// SM120MatMulGradA kernel implementation
template<typename T>
class SM120MatMulGradAOp : public OpKernel {
 public:
  explicit SM120MatMulGradAOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_output_tensor = context->input(0);
    const Tensor& b_tensor = context->input(1);
    
    // Get dimensions
    const auto& grad_output_shape = grad_output_tensor.shape();
    const auto& b_shape = b_tensor.shape();
    
    OP_REQUIRES(context, grad_output_shape.dims() == 2,
                errors::InvalidArgument("grad_output must be 2D"));
    OP_REQUIRES(context, b_shape.dims() == 2,
                errors::InvalidArgument("b must be 2D"));
    
    int M = grad_output_shape.dim_size(0);
    int N = grad_output_shape.dim_size(1);
    int K = b_shape.dim_size(0);
    
    // Allocate output tensor
    TensorShape output_shape({M, K});
    Tensor* grad_a_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &grad_a_tensor));
    
    // Launch SM120 gradient kernel
    auto* stream = context->op_device_context()->stream();
    const T* grad_output = grad_output_tensor.flat<T>().data();
    const T* b = b_tensor.flat<T>().data();
    T* grad_a = grad_a_tensor->flat<T>().data();
    
    auto cuda_status = LaunchSM120MatMulGradA<T>(
        grad_output, b, grad_a, M, N, K, 1.0f, stream->parent());
    
    OP_REQUIRES(context, cuda_status == cudaSuccess,
                errors::Internal("SM120MatMulGradA kernel failed: ", 
                               cudaGetErrorString(cuda_status)));
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};

// SM120MatMulGradB kernel implementation
template<typename T>
class SM120MatMulGradBOp : public OpKernel {
 public:
  explicit SM120MatMulGradBOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& a_tensor = context->input(0);
    const Tensor& grad_output_tensor = context->input(1);
    
    // Get dimensions
    const auto& a_shape = a_tensor.shape();
    const auto& grad_output_shape = grad_output_tensor.shape();
    
    OP_REQUIRES(context, a_shape.dims() == 2,
                errors::InvalidArgument("a must be 2D"));
    OP_REQUIRES(context, grad_output_shape.dims() == 2,
                errors::InvalidArgument("grad_output must be 2D"));
    
    int M = a_shape.dim_size(0);
    int K = a_shape.dim_size(1);
    int N = grad_output_shape.dim_size(1);
    
    // Allocate output tensor
    TensorShape output_shape({K, N});
    Tensor* grad_b_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &grad_b_tensor));
    
    // Launch SM120 gradient kernel
    auto* stream = context->op_device_context()->stream();
    const T* a = a_tensor.flat<T>().data();
    const T* grad_output = grad_output_tensor.flat<T>().data();
    T* grad_b = grad_b_tensor->flat<T>().data();
    
    auto cuda_status = LaunchSM120MatMulGradB<T>(
        a, grad_output, grad_b, M, N, K, 1.0f, stream->parent());
    
    OP_REQUIRES(context, cuda_status == cudaSuccess,
                errors::Internal("SM120MatMulGradB kernel failed: ", 
                               cudaGetErrorString(cuda_status)));
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};

// Register gradient kernels
REGISTER_KERNEL_BUILDER(Name("SM120MatMulGradA").Device(DEVICE_GPU).TypeConstraint<float>("T"),
                        SM120MatMulGradAOp<float>);
REGISTER_KERNEL_BUILDER(Name("SM120MatMulGradA").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
                        SM120MatMulGradAOp<Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("SM120MatMulGradA").Device(DEVICE_GPU).TypeConstraint<Eigen::bfloat16>("T"),
                        SM120MatMulGradAOp<Eigen::bfloat16>);

REGISTER_KERNEL_BUILDER(Name("SM120MatMulGradB").Device(DEVICE_GPU).TypeConstraint<float>("T"),
                        SM120MatMulGradBOp<float>);
REGISTER_KERNEL_BUILDER(Name("SM120MatMulGradB").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
                        SM120MatMulGradBOp<Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("SM120MatMulGradB").Device(DEVICE_GPU).TypeConstraint<Eigen::bfloat16>("T"),
                        SM120MatMulGradBOp<Eigen::bfloat16>);

// SM120Conv2D gradient registration
REGISTER_OP_GRADIENT("SM120Conv2D", [](const AttrSlice& attrs,
                                        const FunctionDef& fdef,
                                        const std::vector<NodeDef*>& grad_inputs,
                                        std::vector<NodeDef*>* grad_outputs) {
  // Extract attributes
  std::vector<int32> strides;
  std::string padding;
  TF_RETURN_IF_ERROR(GetNodeAttr(fdef.signature().input_arg(0), "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(fdef.signature().input_arg(0), "padding", &padding));
  
  // Create gradient nodes for input and filter
  NodeDef* grad_input = new NodeDef();
  grad_input->set_op("SM120Conv2DBackpropInput");
  grad_input->add_input("input_sizes");  // Will be computed from input shape
  grad_input->add_input(fdef.signature().input_arg(1).name());  // filter
  grad_input->add_input(grad_inputs[0]->name());  // grad_output
  (*grad_input->mutable_attr())["strides"].mutable_list()->clear_i();
  for (int32 stride : strides) {
    (*grad_input->mutable_attr())["strides"].mutable_list()->add_i(stride);
  }
  (*grad_input->mutable_attr())["padding"].set_s(padding);
  (*grad_input->mutable_attr())["T"] = fdef.attr().at("T");
  
  NodeDef* grad_filter = new NodeDef();
  grad_filter->set_op("SM120Conv2DBackpropFilter");
  grad_filter->add_input(fdef.signature().input_arg(0).name());  // input
  grad_filter->add_input("filter_sizes");  // Will be computed from filter shape
  grad_filter->add_input(grad_inputs[0]->name());  // grad_output
  (*grad_filter->mutable_attr())["strides"].mutable_list()->clear_i();
  for (int32 stride : strides) {
    (*grad_filter->mutable_attr())["strides"].mutable_list()->add_i(stride);
  }
  (*grad_filter->mutable_attr())["padding"].set_s(padding);
  (*grad_filter->mutable_attr())["T"] = fdef.attr().at("T");
  
  grad_outputs->push_back(grad_input);
  grad_outputs->push_back(grad_filter);
  
  return Status::OK();
});

// SM120Conv2DBackpropInput operation
REGISTER_OP("SM120Conv2DBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("strides: list(int)")
    .Attr("padding: {'SAME', 'VALID'}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_sizes_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_sizes_shape));
      
      const Tensor* input_sizes = c->input_tensor(0);
      if (input_sizes != nullptr) {
        std::vector<shape_inference::DimensionHandle> dims;
        for (int i = 0; i < input_sizes->NumElements(); ++i) {
          dims.push_back(c->MakeDim(input_sizes->flat<int32>()(i)));
        }
        c->set_output(0, c->MakeShape(dims));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    });

// SM120Conv2DBackpropFilter operation
REGISTER_OP("SM120Conv2DBackpropFilter")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("strides: list(int)")
    .Attr("padding: {'SAME', 'VALID'}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle filter_sizes_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &filter_sizes_shape));
      
      const Tensor* filter_sizes = c->input_tensor(1);
      if (filter_sizes != nullptr) {
        std::vector<shape_inference::DimensionHandle> dims;
        for (int i = 0; i < filter_sizes->NumElements(); ++i) {
          dims.push_back(c->MakeDim(filter_sizes->flat<int32>()(i)));
        }
        c->set_output(0, c->MakeShape(dims));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      return Status::OK();
    });

// SM120SoftmaxGrad gradient registration
REGISTER_OP_GRADIENT("SM120Softmax", [](const AttrSlice& attrs,
                                         const FunctionDef& fdef,
                                         const std::vector<NodeDef*>& grad_inputs,
                                         std::vector<NodeDef*>* grad_outputs) {
  NodeDef* grad_node = new NodeDef();
  grad_node->set_op("SM120SoftmaxGrad");
  grad_node->add_input(grad_inputs[0]->name());  // grad_output
  grad_node->add_input(fdef.ret().begin()->second);  // softmax_output (forward result)
  (*grad_node->mutable_attr())["T"] = fdef.attr().at("T");
  
  grad_outputs->push_back(grad_node);
  
  return Status::OK();
});

// SM120SoftmaxGrad operation
REGISTER_OP("SM120SoftmaxGrad")
    .Input("grad_output: T")
    .Input("softmax_output: T")
    .Output("grad_input: T")
    .Attr("T: {float, half}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

// SM120SoftmaxGrad kernel implementation
template<typename T>
class SM120SoftmaxGradOp : public OpKernel {
 public:
  explicit SM120SoftmaxGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_output_tensor = context->input(0);
    const Tensor& softmax_output_tensor = context->input(1);
    
    OP_REQUIRES(context, grad_output_tensor.shape() == softmax_output_tensor.shape(),
                errors::InvalidArgument("grad_output and softmax_output must have same shape"));
    
    const auto& input_shape = grad_output_tensor.shape();
    OP_REQUIRES(context, input_shape.dims() == 2,
                errors::InvalidArgument("Input must be 2D"));
    
    int N = input_shape.dim_size(0);
    int D = input_shape.dim_size(1);
    
    // Allocate output tensor
    Tensor* grad_input_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input_tensor));
    
    // Launch SM120 softmax gradient kernel
    auto* stream = context->op_device_context()->stream();
    const T* grad_output = grad_output_tensor.flat<T>().data();
    const T* softmax_output = softmax_output_tensor.flat<T>().data();
    T* grad_input = grad_input_tensor->flat<T>().data();
    
    auto cuda_status = LaunchSM120SoftmaxGrad<T>(
        grad_output, softmax_output, grad_input, N, D, stream->parent());
    
    OP_REQUIRES(context, cuda_status == cudaSuccess,
                errors::Internal("SM120SoftmaxGrad kernel failed: ", 
                               cudaGetErrorString(cuda_status)));
  }
};

// Register softmax gradient kernel
REGISTER_KERNEL_BUILDER(Name("SM120SoftmaxGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),
                        SM120SoftmaxGradOp<float>);
REGISTER_KERNEL_BUILDER(Name("SM120SoftmaxGrad").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
                        SM120SoftmaxGradOp<Eigen::half>);

// SM120ReLUGrad gradient registration
REGISTER_OP_GRADIENT("SM120ReLU", [](const AttrSlice& attrs,
                                      const FunctionDef& fdef,
                                      const std::vector<NodeDef*>& grad_inputs,
                                      std::vector<NodeDef*>* grad_outputs) {
  NodeDef* grad_node = new NodeDef();
  grad_node->set_op("SM120ReLUGrad");
  grad_node->add_input(grad_inputs[0]->name());  // grad_output
  grad_node->add_input(fdef.signature().input_arg(0).name());  // input
  (*grad_node->mutable_attr())["T"] = fdef.attr().at("T");
  
  grad_outputs->push_back(grad_node);
  
  return Status::OK();
});

// SM120ReLUGrad operation
REGISTER_OP("SM120ReLUGrad")
    .Input("grad_output: T")
    .Input("input: T")
    .Output("grad_input: T")
    .Attr("T: {float, half}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

}  // namespace tensorflow
