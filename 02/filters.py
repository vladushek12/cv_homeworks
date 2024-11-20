import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    image = np.pad(image, Hk // 2)

    for i in range(Hi):
        for j in range(Wi):            
            for ki in range(Hk):
                for kj in range(Wk):
                    ii = i + ki
                    jj = j + kj
                    
                    out[i, j] += image[ii, jj] * kernel[Hk - 1 - ki, Wk - 1 - kj]
            
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """
    Hi, Wi = image.shape
    out = np.zeros((Hi + 2 * pad_height, Wi + 2 * pad_width))

    # Вставляем оригинальное изображение в центр нового массива
    out[pad_height:pad_height + Hi, pad_width:pad_width + Wi] = image

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    padded_image = zero_pad(image, Hk // 2, Wk // 2)

    flipped_kernel = np.flip(kernel)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded_image[i:i + Hk, j:j + Wk] * flipped_kernel)

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    padded_image = zero_pad(image, Hk // 2, Wk // 2)

    flipped_kernel = np.flip(kernel)

    for i in range(Hi):
        for j in range(Wi):
            region = padded_image[i:i + Hk, j:j + Wk]

            out[i, j] = np.sum(region * flipped_kernel)

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    out = np.zeros((Hf, Wf))
    
    img_pad = zero_pad(f, Hg // 2, Wg // 2)
    
    kernel_sum_of_squares = np.sum(g ** 2)

    for i in range(Hf):
        for j in range(Wf):
            img_slice = img_pad[i:i + Hg, j:j + Wg]
            
            norm_coeff = np.sqrt(kernel_sum_of_squares * np.sum(img_slice ** 2))
            
            out[i, j] = np.sum(img_slice * g) / norm_coeff if norm_coeff != 0 else 0

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    g_zero_mean = g - np.mean(g)
    
    out = cross_correlation(f, g_zero_mean)

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    g = g.astype(np.float64)
    f = f.astype(np.float64)

    Hf, Wf = f.shape
    Hg, Wg = g.shape

    out = np.zeros((Hf, Wf))

    img_pad = zero_pad(f, Hg // 2, Wg // 2)

    g_std = np.std(g)
    g_mean = np.mean(g)
    g_norm = (g - g_mean) / g_std
    g_sum_sq = np.sum(g ** 2)

    for i in range(Hf):
        for j in range(Wf):
            img_slice = img_pad[i:i + Hg, j:j + Wg]
            
            img_slice_mean = np.mean(img_slice)
            img_slice_std = np.std(img_slice)
            img_slice_sum_sq = np.sum(img_slice ** 2)
            
            coeff = np.sqrt(g_sum_sq * img_slice_sum_sq)
 
            out[i, j] = np.sum(((img_slice - img_slice_mean) / img_slice_std) * g_norm) / coeff

    return out