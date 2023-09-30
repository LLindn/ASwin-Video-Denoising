import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def batch_psnr(img, imclean, data_range):
    r"""
    Computes the PSNR along the batch dimension

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
        minimum and maximum possible values).
    """

    imgclean_cpu = imclean.data.cpu().numpy()

    # converting output back to Float32 to evaluate in full precision
    img_cpu = img.data.cpu().numpy().astype(np.float32)

    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(
            imgclean_cpu[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range
        )
    return psnr / img_cpu.shape[0]
