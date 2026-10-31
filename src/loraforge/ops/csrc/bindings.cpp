// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> tsqr_factor(torch::Tensor, int64_t, int64_t);
int64_t tsqr_shared_memory_limit(torch::Tensor);
torch::Tensor tsqr_applyQ(torch::Tensor, const std::vector<torch::Tensor>&,
                         int64_t, int64_t, bool);
std::vector<torch::Tensor> wide_factor(torch::Tensor);
torch::Tensor wide_apply(torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> jacobi_eigh(torch::Tensor, int64_t, double);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shared_memory_limit", &tsqr_shared_memory_limit);
    m.def("factor", &tsqr_factor);
    m.def("apply_q", &tsqr_applyQ);
    m.def("wide_factor", &wide_factor);
    m.def("wide_apply", &wide_apply);
    m.def("jacobi_eigh", &jacobi_eigh);
}
