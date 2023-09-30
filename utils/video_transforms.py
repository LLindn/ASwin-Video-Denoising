import os
import random
from typing import Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


def patching(
    video,
    net,
    size_patch_testing=1024,
    overlap_size_patch=20,
    not_overlap_border=True,
):

    b, c, d, h, w = video.size()
    stride = size_patch_testing - overlap_size_patch
    h_idx_list = list(range(0, h - size_patch_testing, stride)) + [
        max(0, h - size_patch_testing)
    ]
    w_idx_list = list(range(0, w - size_patch_testing, stride)) + [
        max(0, w - size_patch_testing)
    ]
    E = torch.zeros(b, c, d, h, w).cuda()
    W = torch.zeros_like(E).cuda()

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = (
                video[
                    ...,
                    h_idx : h_idx + size_patch_testing,
                    w_idx : w_idx + size_patch_testing,
                ]
                .cuda()
                .half()
            )

            out_patch = net(in_patch)
            out_patch_mask = torch.ones_like(out_patch).cuda()

            if not_overlap_border:
                if h_idx < h_idx_list[-1]:
                    out_patch[..., -overlap_size_patch // 2 :, :] *= 0
                    out_patch_mask[..., -overlap_size_patch // 2 :, :] *= 0
                if w_idx < w_idx_list[-1]:
                    out_patch[..., :, -overlap_size_patch // 2 :] *= 0
                    out_patch_mask[..., :, -overlap_size_patch // 2 :] *= 0
                if h_idx > h_idx_list[0]:
                    out_patch[..., : overlap_size_patch // 2, :] *= 0
                    out_patch_mask[..., : overlap_size_patch // 2, :] *= 0
                if w_idx > w_idx_list[0]:
                    out_patch[..., :, : overlap_size_patch // 2] *= 0
                    out_patch_mask[..., :, : overlap_size_patch // 2] *= 0

            E[
                ...,
                h_idx : (h_idx + size_patch_testing),
                w_idx : (w_idx + size_patch_testing),
            ].add_(out_patch)
            W[
                ...,
                h_idx : (h_idx + size_patch_testing),
                w_idx : (w_idx + size_patch_testing),
            ].add_(out_patch_mask)

    return E.div_(W)


class VideoFolderPathToTensor(object):
    """ load video at given folder path to torch.Tensor (C x L x H x W)
        It can be composed with torchvision.transforms.Compose().

    Args:
        max_len (int): Maximum output time depth (L <= max_len). Default is None.
            If it is set to None, it will output all frames.
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, max_len=None, padding_mode=None):
        self.max_len = max_len
        assert padding_mode in (None, "zero", "last")
        self.padding_mode = padding_mode

    def __call__(self, path):
        """
        Args:
            path (str): path of video folder.

        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """

        # get video properity
        frames_path = sorted(
            [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
        )
        frame = cv2.imread(frames_path[0])
        height, width, channels = frame.shape
        time_len = len(frames_path)

        frames = torch.FloatTensor(channels, time_len, height, width)

        # load the video to tensor
        for i in range(time_len):

            if i < time_len:
                # frame exists
                # read frame
                frame = cv2.imread(frames_path[i])
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames[:, i, :, :] = frame.float()
            else:
                # reach the end of the video
                if self.padding_mode == "zero":
                    # fill the rest frames with 0.0
                    frames[:, i:, :, :] = 0
                elif self.padding_mode == "last":
                    # fill the rest frames with the last frame
                    assert i > 0
                    frames[:, i:, :, :] = frames[:, i - 1, :, :].view(
                        channels, 1, height, width
                    )
                break

        frames /= 255
        return frames


class VideoRandomCrop(object):
    """ Crop the given Video Tensor (C x L x H x W) at a random location.

    Args:
        size (sequence): Desired output size like (h, w).
    """

    def __init__(self, size):
        assert len(size) == 2
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.

        Returns:
            torch.Tensor: Cropped video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = np.random.randint(0, H - h)
        left = np.random.randint(0, W - w)

        video = video[:, :, top : top + h, left : left + w]

        return video


class VideoCenterCrop(object):
    """ Crops the given video tensor (C x L x H x W) at the center.

    Args:
        size (sequence): Desired output size of the crop like (h, w).
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.

        Returns:
            torch.Tensor: Cropped Video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = int((H - h) / 2)
        left = int((W - w) / 2)

        video = video[:, :, top : top + h, left : left + w]

        return video


class VideoRandomHorizontalFlip(object):
    """ Horizontal flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.

        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([3])

        return video


class VideoRandomVerticalFlip(object):
    """ Vertical flip the given video tensor (C x L x H x W) randomly with a given probability.

    Args:
        p (float): probability of the video being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to flipped.

        Returns:
            torch.Tensor: Randomly flipped video.
        """

        if random.random() < self.p:
            # horizontal flip the video
            video = video.flip([2])

        return video


class Rotate:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class AddGaussianNoise(object):
    """Add Gaussian noise to sample.

    Args:
        sigma: Desired standard deviation of the noise.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, video):

        # define the noisy data
        c, l, h, w = video.shape
        noisy_video = video + self.sigma * np.random.randn(c, l, h, w).astype(
            np.float32
        )

        return {"target": video, "input": noisy_video}
