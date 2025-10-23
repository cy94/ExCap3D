#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "include/cuda_SimpleMatrixUtil.h"

#include <vector>
#include <cmath>
#include <chrono>

#define _DEBUG
#define T_PER_BLOCK 512
// #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


using namespace torch::indexing;

/// Helper functions ///
void __device__ coordinate_to_tensor_index(const unsigned int batch_id, const int3 coord,
                                           const int dimx, const int dimy, const int dimz, long *occupancy_id) {

    *occupancy_id = (batch_id * dimx * dimy * dimz) + (coord.x * dimy * dimz) + (coord.y * dimz) + coord.z;
}

/// Forward functions ///
void __global__ trilinear_interpolate_forward_kernel(
        const long *__restrict__ occupancy_3d,
        const float *__restrict__ features_3d,
        const float *__restrict__ query_coords,
        float *query_features,
        long *interpolation_indices,
        float *interpolation_weights,
        float *accum_voxel_weights,
        const int batch_size, const int query_num, const int souce_voxel_num,
        const int feature_dim, const int query_dim,
        const int dimx, const int dimy, const int dimz) {

    // This should be the voxel diagonal, and other hyperparameters for regularization
    // NOTE: works only for 2cm voxels?
    const float voxel_max_dist = sqrt(3);
    const float eps = 10e-5;
    float weight_sum = eps;
    float dist, weight;

    //Get voxel index and position
    const unsigned int query_id = blockIdx.x * blockDim.x + threadIdx.x;
    // coords are all of the points batched, where the first element is the batch id in float = 4 ints
    const unsigned int batch = (unsigned int) query_coords[query_id * 4];  
    const unsigned int neighbor_ind_shift = query_id * 8;
    const unsigned int query_feat_ind_shift = query_id * feature_dim;

    // return for invalid voxels and non 3D coords
    if (query_id >= query_num || query_dim != 4) { return; }

    // get query point location - we only work with 3D coords here
    const float3 query_coord = make_float3(query_coords[query_id * query_dim + 1],
                                           query_coords[query_id * query_dim + 2],
                                           query_coords[query_id * query_dim + 3]);

    // start interpolation with 8 neighbors and accumulate weights, indices

    //// N0 xyz ///
    // Add weights with relative distance between sample point and voxel vertices
    int3 voxel_coord = make_int3(query_coord.x, query_coord.y, query_coord.z);

    // check if voxel is within limits - this is happening for all neighbours
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    long voxel_index;
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    // check if the voxel is occupied, if not - dont use it
    long occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        // distance to the input voxel
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        // weight is normalized by the max distance!
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        // summ of all weights
        weight_sum += weight;
        // store the weight and index
        interpolation_weights[neighbor_ind_shift] = weight;
        interpolation_indices[neighbor_ind_shift] = occ_value;
    }


    //// N1 x+1,y,z ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x + 1, query_coord.y, query_coord.z);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 1] = weight;
        interpolation_indices[neighbor_ind_shift + 1] = occ_value;
    }

    //// N2 x,y+1,z ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x, query_coord.y + 1, query_coord.z);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 2] = weight;
        interpolation_indices[neighbor_ind_shift + 2] = occ_value;
    }

    //// N3 x,y,z+1 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x, query_coord.y, query_coord.z + 1);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 3] = weight;
        interpolation_indices[neighbor_ind_shift + 3] = occ_value;
    }

    //// N4 x+1,y+1,z ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x + 1, query_coord.y + 1, query_coord.z);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 4] = weight;
        interpolation_indices[neighbor_ind_shift + 4] = occ_value;
    }

    //// N5 x,y+1,z+1 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x, query_coord.y + 1, query_coord.z + 1);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 5] = weight;
        interpolation_indices[neighbor_ind_shift + 5] = occ_value;
    }

    //// N6 x+1,y,z+1 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x + 1, query_coord.y, query_coord.z + 1);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 6] = weight;
        interpolation_indices[neighbor_ind_shift + 6] = occ_value;
    }

    //// N7 x+1,y+1,z+1 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x + 1, query_coord.y + 1, query_coord.z + 1);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 7] = weight;
        interpolation_indices[neighbor_ind_shift + 7] = occ_value;
    }

    if (weight_sum == eps) { return; } // no neighbour was used

    // Normalize weights with sum
    interpolation_weights[neighbor_ind_shift + 0] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 1] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 2] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 3] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 4] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 5] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 6] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 7] /= weight_sum;

    // iterate over all neighbours and copy voxel features to query features
    for (int n = 0; n < 8; n++) {

        const long voxel_id = interpolation_indices[neighbor_ind_shift + n];
        const float voxel_weight = interpolation_weights[neighbor_ind_shift + n];

        // check if voxel index is meaningful - dont update accumulators for unobserved regions
        if (voxel_id < 0 || voxel_weight == 0) { continue; }

        // update the voxel weight accumulator
        atomicAdd(&accum_voxel_weights[voxel_id], voxel_weight);

        // get feature index of the voxel space
        const long feature_shift_index = voxel_id * feature_dim;

        // iterate over all features
#pragma unroll 1
        for (int f = 0; f < feature_dim; f++) {
            atomicAdd(&query_features[query_feat_ind_shift + f], features_3d[feature_shift_index + f] * voxel_weight);
        }

    }
}

void trilinear_interpolate_forward(at::Tensor occupancy_3d, // (batch_size, dim_x, dim_y, dim_z)
                                   at::Tensor features_3d,  // SparseTensor.F -- indexes are stored in occupancy_3d
                                   at::Tensor query_coords, // coordinate locations of the query field
                                   at::Tensor query_features, // to be used to copy query features to
                                   at::Tensor interpolation_indices, // to be used to store interpolation indices for backward
                                   at::Tensor interpolation_weights, // to be used to store interpolation voxel weights
                                   at::Tensor accum_voxel_weights)  //storing the weighted number of voxel-point associations to use for voxel grad normalization in backward pass

{
    // Check inputs
    CHECK_INPUT(occupancy_3d);
    CHECK_INPUT(features_3d);
    CHECK_INPUT(query_coords);
    CHECK_INPUT(query_features);
    CHECK_INPUT(interpolation_indices);
    CHECK_INPUT(interpolation_weights);
    CHECK_INPUT(accum_voxel_weights);

    // Get dimensions
    const int batch_size = occupancy_3d.size(0);
    const int query_num = query_coords.size(0);
    const int feature_dim = features_3d.size(1);
    const int souce_voxel_num = features_3d.size(0);
    const int query_dim = query_coords.size(1);
    const int dimx = occupancy_3d.size(1);
    const int dimy = occupancy_3d.size(2);
    const int dimz = occupancy_3d.size(3);

    const dim3 gridSize((query_num + T_PER_BLOCK - 1) / T_PER_BLOCK, 1, 1);
    const dim3 blockSize(T_PER_BLOCK, 1, 1);

    trilinear_interpolate_forward_kernel<<<gridSize, blockSize>>>(
            occupancy_3d.data<long>(),
            features_3d.data<float>(),
            query_coords.data<float>(),
            query_features.data<float>(),
            interpolation_indices.data<long>(),
            interpolation_weights.data<float>(),
            accum_voxel_weights.data<float>(),
            batch_size, query_num, souce_voxel_num,
            feature_dim, query_dim,
            dimx, dimy, dimz);

    cutilSafeCall(cudaDeviceSynchronize());
    // cudaDeviceSynchronize();
    cutilCheckMsg(__FUNCTION__);
}


/// Backward functions ///
void __global__ trilinear_interpolate_backward_kernel(const long *__restrict__ interpolation_indices,
                                                      const float *__restrict__ interpolation_weights,
                                                      const float *__restrict__ accum_voxel_weights,
                                                      const float *__restrict__ query_grad,
                                                      float *voxel_grad,
                                                      const int query_num, const int grad_dim, const int voxel_num){

    // Get voxel index and position
    const unsigned int query_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int weight_ind_shift = query_id * 8;
    const unsigned int query_grad_shift = query_id * grad_dim;

    // return for invalid query points
    if (query_id >= query_num) { return; }

    // For all neighbours do the grad summary
    int voxel_index, n_grad_shift;
    float n_weight, voxel_accum_weight;
    for (int n = 0; n < 8; n++){

        voxel_index = interpolation_indices[weight_ind_shift + n];
        n_weight = interpolation_weights[weight_ind_shift + n];
        n_grad_shift = voxel_index * grad_dim;

        // if neighbour was not initialized during interpolation skip, or index is invalid return
        if (voxel_index < 0 || voxel_index >= voxel_num) { continue; }

        // Get the accumulated weight only of it is valid voxel index
        voxel_accum_weight = accum_voxel_weights[voxel_index];
        if (voxel_accum_weight <= 0.) { continue; }

        // for all grad dimensions add the value
#pragma unroll 1
        for (int f = 0; f < grad_dim; f++){
            atomicAdd(&voxel_grad[n_grad_shift + f], query_grad[query_grad_shift + f] * n_weight / voxel_accum_weight);
        }
    }
}

void trilinear_interpolate_backward(at::Tensor interpolation_indices,
                                    at::Tensor interpolation_weights,
                                    at::Tensor accum_voxel_weights,
                                    at::Tensor query_grads,
                                    at::Tensor voxel_grads){

    // Check inputs
    CHECK_INPUT(interpolation_indices);
    CHECK_INPUT(interpolation_weights);
    CHECK_INPUT(accum_voxel_weights);
    CHECK_INPUT(query_grads);
    CHECK_INPUT(voxel_grads);

    // Get dimensions
    const unsigned int query_num = query_grads.size(0);
    const unsigned int grad_dim = query_grads.size(1);
    const unsigned int voxel_num = voxel_grads.size(0);

    const dim3 gridSize((query_num + T_PER_BLOCK - 1) / T_PER_BLOCK, 1);
    const dim3 blockSize(T_PER_BLOCK, 1, 1);

    trilinear_interpolate_backward_kernel<<<gridSize, blockSize>>>(
            interpolation_indices.data<long>(),
            interpolation_weights.data<float>(),
            accum_voxel_weights.data<float>(),
            query_grads.data<float>(),
            voxel_grads.data<float>(),
            query_num, grad_dim, voxel_num);

    cutilSafeCall(cudaDeviceSynchronize());
    // cudaDeviceSynchronize();
    cutilCheckMsg(__FUNCTION__);

}