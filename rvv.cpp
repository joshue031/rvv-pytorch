#include<torch/script.h>
#include<riscv_vector.h>
#include<iostream>

using namespace torch;

// Print the contents of a matrix in memory.
void print_matrix(const float * mat, size_t nelems) {

  std::cout << "Printing matrix with " << nelems << " elements\n\n[" << std::endl;
  for(int i = 0; i < nelems; i++) std::cout << mat[i] << ", ";
  std::cout << "]\n" << std::endl;
}

// RISCV vector add
// Compute a = b + c
void rvv_vec_add(float* a, float* b, float* c, size_t n) {

  vfloat32m2_t va, vb, vc;

  for(size_t vl; n > 0; n -= vl) {

    vl = vsetvl_e32m2(n);

    vb = vle32_v_f32m2(b, vl);
    vc = vle32_v_f32m2(c, vl);
    va = vfadd_vv_f32m2(vb, vc, vl);
    vse32_v_f32m2(a, va, vl);

    a += vl;
    b += vl;
    c += vl;
  }
}

// RISCV vector multiply
// Compute a = b * c
void rvv_vec_mul_const(float* a, float* b, const float c, size_t n) {

  vfloat32m2_t va, vb;

  for(size_t vl; n > 0; n -= vl) {

    vl = vsetvl_e32m2(n);

    vb = vle32_v_f32m2(b, vl);
    va = vfmul_vf_f32m2(vb, c, vl);
    vse32_v_f32m2(a, va, vl);

    a += vl;
    b += vl;
  }
}

//
// RISCV matrix multiplication
//
// Code from: https://github.com/riscv-non-isa/rvv-intrinsic-doc/blob/master/examples/rvv_sgemm.c
// Copyright (c) 2021 Hsiangkai Wang et al. All rights reserved.
// BSD 3-clause license
//
// reference https://github.com/riscv/riscv-v-spec/blob/master/example/sgemm.S
// c += a*b (alpha=1, no transpose on input matrices)
// matrices stored in C row-major order
void rvv_vec_sgemm(size_t size_m, size_t size_n, size_t size_k,
               const float *a, // m * k matrix
               size_t lda,
               const float *b, // k * n matrix
               size_t ldb,
               float *c, // m * n matrix
               size_t ldc) {
  size_t vl;
  for (size_t m = 0; m < size_m; ++m) {
    const float *b_n_ptr = b;
    float *c_n_ptr = c;
    for (size_t c_n_count = size_n; c_n_count; c_n_count -= vl) {
      vl = vsetvl_e32m2(c_n_count );
      const float *a_k_ptr = a;
      const float *b_k_ptr = b_n_ptr;
      vfloat32m2_t acc = vle32_v_f32m2(c_n_ptr, vl);
      for (size_t k = 0; k < size_k; ++k) {
        vfloat32m2_t b_n_data = vle32_v_f32m2(b_k_ptr, vl);
        acc = vfmacc_vf_f32m2(acc, *a_k_ptr, b_n_data, vl);
        b_k_ptr += ldb;
        a_k_ptr++;
      }
      vse32_v_f32m2(c_n_ptr, acc, vl);
      c_n_ptr += vl;
      b_n_ptr += vl;
    }
    a += lda;
    c += ldc;
  }
}

// Operator to replace aten::add.Tensor
// result = self + alpha*other
Tensor rvv_add(const Tensor& self, const Tensor& other, const Scalar & alpha) {

  Tensor cself, cother;
  bool ok = true;

  // Expand tensors with fewer dimensions if possible.
  if(self.sizes() == other.sizes()) {
    cself = self.contiguous();
    cother = other.contiguous();
  }
  else if((self.dim() != other.dim()) && (self.size(-1) == other.size(-1))) {
    if(self.dim() > other.dim()) {
      cself = self.contiguous();
      cother = Tensor(other).expand(self.sizes()).contiguous();
    }
    else if(self.dim() < other.dim()) {
      cself = Tensor(self).expand(other.sizes()).contiguous();
      cother = other.contiguous();
    }
  }
  else ok = false;

  TORCH_CHECK(ok);
  TORCH_INTERNAL_ASSERT(self.device().type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(other.device().type() == DeviceType::CPU);

  // Get the data pointers of the tensors to be added.
  float * self_data = cself.data_ptr<float>();
  float * other_data = cother.data_ptr<float>();

  // Create a new tensor to store the result.
  size_t n = cself.numel();
  Tensor sum = torch::zeros(cself.sizes(), cself.options());
  float * sum_data = sum.data_ptr<float>();

  // Call the vector operations.
  rvv_vec_mul_const(sum_data, other_data, alpha.to<float>(),n);
  rvv_vec_add(sum_data, self_data, sum_data, n);

  return sum;
}

// Operator to replace aten::mm.out
Tensor & rvv_mm_out(const Tensor & self, const Tensor & mat2, Tensor & out) {

  Tensor ct_self = self.contiguous();
  Tensor ct_mat2 = mat2.contiguous();
  Tensor ct_out  = out.contiguous();

  // Get the data pointers.
  float * self_data = ct_self.data_ptr<float>();
  float * mat2_data = ct_mat2.data_ptr<float>();
  float * out_data = ct_out.data_ptr<float>();

  // Call the vector matrix multiplication.
  size_t m = self.size(0);
  size_t n = mat2.size(1);
  size_t k = self.size(1);
  rvv_vec_sgemm(m,n,k,self_data,k,mat2_data,n,out_data,n);

  return out;
}

Tensor rvv_matmul(const Tensor & self, const Tensor & other) {

  Tensor ct_self = self.contiguous();
  Tensor ct_other = other.contiguous();

  // Get the data pointers.
  float * self_data = ct_self.data_ptr<float>();
  float * other_data = ct_other.data_ptr<float>();

  // Create a new matrix of zeros to contain the results.
  Tensor out = torch::zeros({self.size(0),other.size(1)}, ct_self.options());
  float * out_data = out.data_ptr<float>();

  //std::cout << "matrix self = " << std::endl;
  //print_matrix(self_data, m*k);
  //std::cout << "matrix other = " << std::endl;
  //print_matrix(other_data, other.size(0)*n);

  // Call the vector matrix multiplication.
  size_t m = self.size(0);
  size_t n = other.size(1);
  size_t k = self.size(1);
  rvv_vec_sgemm(m,n,k,self_data,k,other_data,n,out_data,n);
  //std::cout << "matrix out = " << std::endl;
  //print_matrix(out_data,k*n);

  return out;
}

// Operator to replace aten::linear
Tensor rvv_linear(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias_opt) {

  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  // Multiply the weight matrix.
  Tensor out = rvv_matmul(input, weight.t());
  //std::cout << " input size: " << input.sizes() << ", weight size: " << weight.sizes() << ", bias size: " << bias->sizes() << " and output size: " << out.sizes() << std::endl;

  // Add the bias, if defined.
  if(bias->defined()) out = rvv_add(out, *bias, 1.0);

  return out;
}

// Operator to replace aten::linear_backward
std::tuple<Tensor,Tensor,Tensor> rvv_linear_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, std::array<bool,3> output_mask) {

  Tensor grad_input, grad_weight, grad_bias;

  // Compute the gradients specified by the output mask.
  if(output_mask[0]) {
    grad_input = rvv_matmul(grad_output, weight);
  }
  if(output_mask[1]) {
    grad_weight = rvv_matmul(grad_output.t(), self);
  }
  if(output_mask[2]) {
    grad_bias = grad_output.sum(0);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

// For now, override the CPU registrations.
TORCH_LIBRARY_IMPL(aten, CPU, m) {
	m.impl("aten::add.Tensor", &rvv_add);
	m.impl("aten::matmul", &rvv_matmul);
	m.impl("aten::mm.out", &rvv_mm_out);
	m.impl("aten::linear", &rvv_linear);
	m.impl("aten::linear_backward", &rvv_linear_backward);
}
