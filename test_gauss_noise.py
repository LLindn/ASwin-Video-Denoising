import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from network import Net
from utils import video_datasets, video_transforms
from utils.batch_psnr import batch_psnr
from utils.video_transforms import patching


def gauss_noise_inference(
    noise_sigma=20,
    num_frame_testing=24,
    num_frame_overlapping=2,
    patch_img=False,
    patch_size=1024,
    file_list="./data/davis2017/files.csv",
):

    assert noise_sigma in [10, 20, 30, 40, 50]
    assert num_frame_testing % 6 == 0
    assert num_frame_overlapping < num_frame_testing

    load = True
    save_image = True
    calc_psnr = True

    sigma = noise_sigma / 255.0

    dataset = video_datasets.VideoDataset(
        file_list,
        transform=torchvision.transforms.Compose(
            [
                video_transforms.VideoFolderPathToTensor(padding_mode="last"),
                video_transforms.AddGaussianNoise(sigma),
            ]
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

        path = "./model_weights/gauss%s" % str(noise_sigma)

        model_path = path + "/weights.pth"
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict)

    psnr_total = []

    with torch.no_grad():

        net.eval()

        for video, sample in enumerate(test_loader):

            psnr_all = []
            psnr_init_all = []

            u = sample["input"]
            u_target = sample["target"]

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
                unit="",
                bar_format="{desc}  {percentage:3.0f}%|{bar}| {unit}",
            ) as pbar:

                for d_idx in d_idx_list:
                    lq_clip = (
                        u[:, :, d_idx : d_idx + num_frame_testing, ...].cuda().half()
                    )

                    if patch_img == True and (h > patch_size or w > patch_size):
                        u_rec_tmp = patching(
                            lq_clip,
                            net,
                            size_patch_testing=patch_size,
                            overlap_size_patch=20,
                            not_overlap_border=True,
                        )
                    else:
                        u_rec_tmp = net(lq_clip)
                    u_rec_tmp = u_rec_tmp.float()

                    out_clip_mask = torch.ones(
                        (b, 1, min(num_frame_testing, d), 1, 1)
                    ).cuda()

                    E_all[:, :, d_idx : d_idx + num_frame_testing, ...].add_(u_rec_tmp)
                    W_all[:, :, d_idx : d_idx + num_frame_testing, ...].add_(
                        out_clip_mask
                    )

                    del lq_clip, out_clip_mask
                    torch.cuda.empty_cache()

                    pbar.update(1)

                u_rec = E_all.div_(W_all)

                if calc_psnr:
                    psnr_final = batch_psnr(
                        u_rec.detach().cpu().squeeze().permute(1, 0, 2, 3),
                        u_target.squeeze().permute(1, 0, 2, 3),
                        1.0,
                    )
                    psnr_init = batch_psnr(
                        u.detach().cpu().squeeze().permute(1, 0, 2, 3),
                        u_target.squeeze().permute(1, 0, 2, 3),
                        1.0,
                    )

                    psnr_all.append(psnr_final.item())
                    psnr_init_all.append(psnr_init.item())
                    psnr_total.append(psnr_final.item())

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
                            f"./results/target_{video:03d}_{i:03d}.png",
                            (
                                u_target[0, :, i]
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
                            f"./results/result_{video:03d}_{i:03d}.png",
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

                pbar.unit = (
                    "  initial PSNR: "
                    + str(np.round(np.mean(psnr_init_all), 4))
                    + "    denoised PSNR: "
                    + str(np.round(np.mean(psnr_all), 4))
                )
                pbar.update(0)

                del u, u_target, u_rec, u_rec_tmp
                torch.cuda.empty_cache()
            #
            del E_all, W_all
            torch.cuda.empty_cache()

    if calc_psnr:
        print("===========================================================")
        print("===================Final result============================")
        print("PSNR: " + str(np.round(np.mean(psnr_total), 4)))
        print("===========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for gauss_noise_inference function"
    )

    parser.add_argument(
        "--noise_sigma", type=int, default=20, help="The sigma value for the noise"
    )
    parser.add_argument(
        "--num_frame_testing",
        type=int,
        default=24,
        help="The number of frames for testing (must be a multiple of 6 due to temporal window size).",
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
        default="./data/davis2017/files.csv",
        help="Dataset file",
    )

    args = parser.parse_args()

    gauss_noise_inference(
        noise_sigma=args.noise_sigma,
        num_frame_testing=args.num_frame_testing,
        num_frame_overlapping=args.num_frame_overlapping,
        patch_img=args.patch_img,
        patch_size=args.patch_size,
        file_list=args.file_list,
    )
