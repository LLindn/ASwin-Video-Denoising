import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from network import Net
from utils import video_datasets, video_transforms
from utils.video_transforms import patching


def real_noise_inference(
    num_frame_testing=24,
    num_frame_overlapping=2,
    patch_img=True,
    patch_size=1024,
    file_list="./data/test/files.csv",
):

    assert num_frame_testing % 6 == 0
    assert num_frame_overlapping < num_frame_testing

    load = True
    save_image = True

    dataset = video_datasets.VideoDataset(
        file_list,
        transform=torchvision.transforms.Compose(
            [video_transforms.VideoFolderPathToTensor(padding_mode="last")]
        ),
    )

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8
    )

    net = Net(num_frame_testing, in_nc=3, dim=64).cuda()
    for k, v in net.named_parameters():
        v.requires_grad = False

    # using half precision during inference
    net = net.half()

    if load:
        print("Loading model parameters ... ")
        time.sleep(0.1)

        path = "./model_weights/blind_real_noise"

        model_path = path + "/weights.pth"
        state_dict = torch.load(model_path)

        net.load_state_dict(state_dict)

    with torch.no_grad():

        net.eval()

        for video, sample in enumerate(test_loader):

            u = sample

            b, c, d, h, w = u.size()

            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d - num_frame_testing, stride)) + [
                max(0, d - num_frame_testing)
            ]
            E_all = torch.zeros(b, c, d, h, w).cuda()
            W_all = torch.zeros(b, 1, d, 1, 1).cuda()

            with tqdm(
                total=(int(len(d_idx_list))),
                desc=f"Video {video + 1}:",
                bar_format="{desc}  {percentage:3.0f}%|{bar}|",
            ) as pbar:

                for d_idx in d_idx_list:
                    video_sequence = (
                        u[:, :, d_idx : d_idx + num_frame_testing, ...].cuda().half()
                    )

                    if patch_img == True and (h > patch_size or w > patch_size):
                        u_rec_tmp = patching(
                            video_sequence,
                            net,
                            size_patch_testing=patch_size,
                            overlap_size_patch=20,
                            not_overlap_border=True,
                        )
                    else:
                        u_rec_tmp = net(video_sequence)

                    u_rec_tmp = u_rec_tmp.float()

                    out_clip_mask = torch.ones(
                        (b, 1, min(num_frame_testing, d), 1, 1)
                    ).cuda()

                    E_all[:, :, d_idx : d_idx + num_frame_testing, ...].add_(u_rec_tmp)
                    W_all[:, :, d_idx : d_idx + num_frame_testing, ...].add_(
                        out_clip_mask
                    )

                    del video_sequence, out_clip_mask
                    torch.cuda.empty_cache()

                    pbar.update(1)

                u_rec = E_all.div_(W_all)

                if save_image:

                    for i in range(u.size(2)):
                        plt.imsave(
                            f"./results/noisy_{video:03d}_{i:03d}.png",
                            (
                                u[0, :, i]
                                .squeeze()
                                .clip(0, 1)
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                                * 255
                            ).astype(np.uint8),
                        )
                        plt.imsave(
                            f"./results/res_{video:03d}_{i:03d}.png",
                            (
                                u_rec[0, :, i]
                                .squeeze()
                                .clip(0, 1)
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                                * 255
                            ).astype(np.uint8),
                        )

                pbar.update(0)

                del u, u_rec, u_rec_tmp
                torch.cuda.empty_cache()

            del E_all, W_all
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for real_noise_inference function"
    )

    parser.add_argument(
        "--num_frame_testing",
        type=int,
        default=24,
        help="The number of frames for testing  (must be a multiple of 6 due to temporal window size).",
    )
    parser.add_argument(
        "--num_frame_overlapping",
        type=int,
        default=2,
        help="The number of overlapping frames",
    )
    parser.add_argument(
        "--patch_img", type=bool, default=False, help="Patch the video spatially",
    )
    parser.add_argument(
        "--patch_size", type=int, default=1024, help="Size of spatial patch"
    )
    parser.add_argument(
        "--file_list",
        type=str,
        default="./data/test_real_noise/files.csv",
        help="Dataset file",
    )

    args = parser.parse_args()

    real_noise_inference(
        num_frame_testing=args.num_frame_testing,
        num_frame_overlapping=args.num_frame_overlapping,
        patch_img=args.patch_img,
        patch_size=args.patch_size,
        file_list=args.file_list,
    )
