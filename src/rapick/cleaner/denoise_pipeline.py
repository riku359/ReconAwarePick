"""Standalone micrograph-denoising pipeline, vendored from the upstream CryoSegNet
picker (originally `cryosegnet/denoise.py`; the CryoSegNet picker itself is not part
of this release).

overlay_panel.py's denoise_flip_frame() needs this exact processing chain to draw a
background for a micrograph that has no pre-computed denoised JPG, so the function
is vendored here rather than dropped with the rest of that picker: it has no torch /
SAM / CryoSegNet-specific config dependency, just numpy/opencv/scipy.

    standard_scaler (Gaussian blur + z-score -> uint8)
        -> contrast_enhancement (fastNlMeansDenoising)
        -> wiener_filter (9x9 Gaussian kernel, K=30)
        -> CLAHE
        -> guided_filter
"""
import cv2
import numpy as np
from numpy.fft import fft2, ifft2

try:                                    # scipy >= 1.13 moved this out of scipy.signal
    from scipy.signal.windows import gaussian
except ImportError:                     # pragma: no cover - older scipy
    from scipy.signal import gaussian


def transform(image):
    i_min = image.min()
    i_max = image.max()
    image = ((image - i_min) / (i_max - i_min)) * 255
    return image.astype(np.uint8)


def standard_scaler(image):
    image = image.astype(np.float32)
    kernel_size = 9
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    mu = np.mean(image)
    sigma = np.std(image)
    image = (image - mu) / sigma
    image = transform(image).astype(np.uint8)
    return image


def contrast_enhancement(image):
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)


def gaussian_kernel(kernel_size=3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy


def clahe(image):
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return c.apply(transform(image))


def guided_filter(input_image, guidance_image, radius=20, epsilon=0.1):
    input_image = input_image.astype(np.float32) / 255.0
    guidance_image = guidance_image.astype(np.float32) / 255.0
    mean_guidance = cv2.boxFilter(guidance_image, -1, (radius, radius))
    mean_input = cv2.boxFilter(input_image, -1, (radius, radius))
    mean_guidance_input = cv2.boxFilter(guidance_image * input_image, -1, (radius, radius))
    covariance_guidance_input = mean_guidance_input - mean_guidance * mean_input
    mean_guidance_sq = cv2.boxFilter(guidance_image * guidance_image, -1, (radius, radius))
    variance_guidance = mean_guidance_sq - mean_guidance * mean_guidance
    a = covariance_guidance_input / (variance_guidance + epsilon)
    b = mean_input - a * mean_guidance
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    output_image = mean_a * guidance_image + mean_b
    return transform(output_image)


KERNEL = gaussian_kernel(kernel_size=9)
