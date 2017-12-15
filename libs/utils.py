import numpy as np

def compute_psnr(im1, im2):

    assert im1.shape == im2.shape

    if im1.ndim == 3:
        err = np.mean((im1 - im2) ** 2, axis = (0, 1))
        err = -10 * np.mean(np.log10(err))
    elif im1.ndim == 4:
        err = np.mean((im1 - im2) ** 2, axis=(1, 2))
        err = -10 * np.mean(np.log10(err))

    # err = np.mean((im1 - im2) ** 2)
    # err = -10 * np.log10(err)

    return err
