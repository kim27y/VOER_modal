import torch
from torch import nn
import copy
import numpy as np
from pytorchvideo.models.head import ResNetBasicHead
from torch.functional import F

def time_to_second(times):
    return int(times.split(':')[0])*3600 \
            + int(times.split(':')[1])*60 \
            + int(times.split(':')[2])

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self, slowfast_alpha):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list
    
class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=2)
        return x

def add_quant_op(module, size):
    # For slowfast
    for name, child in module.named_children():

        # if isinstance(child, ResNetBasicHead):
        #     quant_conv = GlobalAvgPool()
        #     module._modules[name] = quant_conv
        # if name == 'activation' or name == 'dropout' or 'output_pool' or 'proj':
        #     print(name, child)
        if name == 'activation' or name == 'dropout':
        # if name =='proj' or name == 'activation' or name == 'dropout':
            # print(f"chaneg {name}")
            module._modules[name] = nn.Identity()
        elif name =='output_pool':
            # print(f"chaneg {name}")
            quant_conv = GlobalAvgPool()
            module._modules[name] = quant_conv
        else:
            add_quant_op(child,size)
            pass

# 김종구, 400에서 1024
def prepare(model, inplace=False, size = 400):
    if not inplace:
        model = copy.deepcopy(model)
        return model
    add_quant_op(
        model,size
    )
    return model


class TemporalSegmentSubsample(torch.nn.Module):
    """
    Note:
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.
    """

    def __init__(self, 
                num_segments:int, 
                frames_per_segment:int, 
                temporal_dim:int=-3, 
                test_mode:bool= False):
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self.num_segments = num_segments
        self.frames_per_segments = frames_per_segment
        self._temporal_dim = temporal_dim
        self.test_mode = test_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return get_start_indices(
            x = x,
            num_segments=self.num_segments,
            frames_per_segment=self.frames_per_segments,
            temporal_dim=self._temporal_dim,
            test_mode=self.test_mode
        )

def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)

def get_start_indices(x: torch.Tensor,  
                    num_segments:int, 
                    frames_per_segment:int, 
                    temporal_dim:int=-3, 
                    test_mode:bool= False) -> 'np.ndarray[int]':
    
    
    total_frames = x.shape[temporal_dim] #total length

    if (total_frames > frames_per_segment and total_frames > 0 and not test_mode):
        pass
    else:
        return uniform_temporal_subsample(x, num_samples=num_segments*frames_per_segment, temporal_dim=temporal_dim)
    # choose start indices that are perfectly evenly spread across the video frames.

    lin_indices = np.linspace(0, total_frames, num_segments+1, dtype=int)
    pairs = list(zip(lin_indices[::1], lin_indices[1::1]))
    indices =[]
    for pair in pairs:
        lin_indices = np.random.choice(range(*pair), frames_per_segment, replace=True)
        order = lin_indices.argsort()
        indices.extend(lin_indices[order])
    indices = torch.LongTensor(indices)
    indices = torch.index_select(x, temporal_dim, indices)
    return indices
    
if __name__ == '__main__':
    x = torch.Tensor(1, 3, 5, 224, 224)
    # res = get_start_indices(x = x, 
    #                 num_segments=4, 
    #                 frames_per_segment=8, 
    #                 temporal_dim=-3, 
    #                 test_mode=True)
    res = torch.mean(x, dim=2)
    res = res.unsqueeze(2)
    print(res.shape)