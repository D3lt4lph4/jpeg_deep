""" Helper functions for the generation of data-augmented images.

All the functions were taken from https://github.com/rykov8/ssd_keras/blob/master/SSD_training.ipynb
"""

import numpy as np


def grayscale(rgb):
    return rgb.dot([0.299, 0.587, 0.114])


def saturation(rgb, saturation_var=0.5):
    gs = grayscale(rgb)
    alpha = 2 * np.random.random() * saturation_var
    alpha = alpha + 1 - saturation_var
    rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
    return np.array(np.clip(rgb, 0, 255), dtype=np.uint8)


def brightness(rgb, brightness_var=0.5, saturation_var=0.5):
    alpha = 2 * np.random.random() * brightness_var
    alpha = alpha + 1 - saturation_var
    rgb = rgb * alpha
    return np.array(np.clip(rgb, 0, 255), dtype=np.uint8)


def contrast(rgb, contrast_var=0.5):
    gs = grayscale(rgb).mean() * np.ones_like(rgb)
    alpha = 2 * np.random.random() * contrast_var
    alpha = alpha + 1 - contrast_var
    rgb = rgb * alpha + (1 - alpha) * gs
    return np.array(np.clip(rgb, 0, 255), dtype=np.uint8)


def lighting(img, lighting_std=0.5):
    cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    noise = np.random.randn(3) * lighting_std
    noise = eigvec.dot(eigval * noise) * 255
    img = img + noise
    return np.array(np.clip(img, 0, 255), dtype=np.uint8)


def horizontal_flip(img, y, hflip_prob=0.5):
    if np.random.random() < hflip_prob:
        img = img[:, ::-1]
        y[:, [0, 2]] = 1 - y[:, [2, 0]]
    return img, y


def vertical_flip(img, y, vflip_prob=0.5):
    if np.random.random() < vflip_prob:
        img = img[::-1]
        y[:, [1, 3]] = 1 - y[:, [3, 1]]
    return img, y