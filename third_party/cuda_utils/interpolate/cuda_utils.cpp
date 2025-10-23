#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// Interpolation fucntions from dense Tensor to query points
// Similar to the MinkowskiEngine Interpolation, but doesnt take into account unoccupied vertices
void trilinear_interpolate_forward(at::Tensor occupancy_3d,
                                   at::Tensor features_3d,
                                   at::Tensor query_coords,
                                   at::Tensor query_features,
                                   at::Tensor interpolation_indices,
                                   at::Tensor interpolation_weights,
                                   at::Tensor accum_voxel_weights);

void trilinear_interpolate_backward(at::Tensor interpolation_indices,
                                    at::Tensor interpolation_weights,
                                    at::Tensor accum_voxel_weights,
                                    at::Tensor query_grads,
                                    at::Tensor voxel_grads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("trilinear_interpolate", &trilinear_interpolate_forward, "Interpolate features for query coordinates from occupied source voxels");
m.def("trilinear_interpolate_backward", &trilinear_interpolate_backward, "Backward function for Trilinear interpolation");
}
