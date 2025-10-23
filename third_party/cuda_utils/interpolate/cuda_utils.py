import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch import nn
from torch.autograd import Function

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

import custom_cuda_utils

'''
Interpolate forward and backward query point features from a sparseTensor (occupancy values will be made sense in runtime)
'''
class InterpolationFunction(Function):
    @staticmethod
    def forward(ctx, occ3d, features_3d, query_coords, query_features, interpolation_indices, interpolation_weights):

        # Storing accumulated voxel weights for normalizing the gradients later
        grad_shape = features_3d.shape
        accum_voxel_weights = torch.zeros(features_3d.shape[0], device=features_3d.device).contiguous()
        d_voxels = torch.zeros(grad_shape, device=features_3d.device).contiguous()

        # Do the interpolation function with the CUDA API
        custom_cuda_utils.trilinear_interpolate(occ3d, features_3d, query_coords, query_features, interpolation_indices, interpolation_weights, accum_voxel_weights)

        # Save for backward pass
        variables = [d_voxels, interpolation_indices, interpolation_weights, accum_voxel_weights]
        ctx.save_for_backward(*variables)

        # Return all necessary components
        return query_features.contiguous(), interpolation_indices, interpolation_weights, accum_voxel_weights

    @staticmethod
    def backward(ctx, grad_query_points, grad_query_inds, grad_query_ind_weights, grad_accum_voxel_weights):

        # Retrieve forward params
        d_voxels, interpolation_indices, interpolation_weights, accum_voxel_weights = ctx.saved_variables

        # Backward pass is essentially a weighted averaging of sampled point gradients
        custom_cuda_utils.trilinear_interpolate_backward(interpolation_indices, interpolation_weights, accum_voxel_weights, grad_query_points.contiguous(), d_voxels)
        # Only return grad for the feature space from the 3D model
        return None, d_voxels, None, None, None, None, None, None

    @staticmethod
    def backward_debug(grad_query_points, d_voxels, interpolation_indices, interpolation_weights, accum_voxel_weights):
        custom_cuda_utils.trilinear_interpolate_backward(interpolation_indices, interpolation_weights, accum_voxel_weights, grad_query_points.contiguous(), d_voxels)
        return d_voxels


class TrilinearInterpolateFeatures(nn.Module):
    def __init__(self, ignore_label=-1):
        
        super(TrilinearInterpolateFeatures, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, s_3d_output, query_points, scale_factor=1, scale_input_coords=True, scale_query_coords=True):
        '''
        Default scale both input and query coords by scale_factor if its not 1
        '''

        # Check for query cords if contiguous
        query_points = query_points if query_points.is_contiguous() else query_points.contiguous()

        # Get batch size and allocate indexes image
        # last batch index + 1 = num samples in batch
        batch_size = s_3d_output.C[-1, 0] + 1
        feature_dim = s_3d_output.F.shape[1]
        voxel_num = s_3d_output.C.shape[0]
        device = s_3d_output.C.device
        
        # Shift coordinates and poses for cuda occupancy tensor
        coords = s_3d_output.C.detach().clone().contiguous()
        local_query = query_points.clone().contiguous()

        # renormalize for interpolation of low_res coords
        if scale_factor != 1 and scale_input_coords:
            coords[:, 1:] = coords[:, 1:] // scale_factor
        if scale_factor != 1 and scale_query_coords:
            local_query /= scale_factor

        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            query_batch_mask = local_query[:, 0] == b
            # subtract the min of coords from input and query coords
            shift = torch.amin(coords[batch_mask, 1:], 0)
            coords[batch_mask, 1:] -= shift
            # NOTE: query coords also has batch index in the first column!
            local_query[query_batch_mask, 1:] -= shift

        # create occupancy tensor
        # features = 0,1,2,3,4.. num input voxels - ignore label -> why?
        occ3d = SparseTensor(features=torch.arange(voxel_num, device=device).view(-1, 1) - self.ignore_label,
                             coordinates=coords,
                             device=device)
        # add back the ignore label
        # coordinates that are occupied - how is this different from input coords
        occ3d = occ3d.dense()[0].long().squeeze(1).contiguous() + self.ignore_label  # (batch, x, y, z)

        # Create container tensors
        query_features = torch.zeros(query_points.shape[0], feature_dim, dtype=torch.float, device=device).contiguous()
        # empty tensors for indices and weights -> for each output, 8 neighbors = 8 indices and 8 weights
        # why add ignore label?
        interpolation_indices = torch.zeros(query_points.shape[0], 8, dtype=torch.long, device=device).contiguous() + self.ignore_label
        interpolation_weights = torch.zeros(query_points.shape[0], 8, dtype=torch.float, device=device).contiguous()

        # Calculate interpolation with autograd
        return InterpolationFunction.apply(occ3d, s_3d_output.F.contiguous(), local_query, query_features, interpolation_indices, interpolation_weights)
