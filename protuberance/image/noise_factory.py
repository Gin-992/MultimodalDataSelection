import numpy as np
import cv2


class NoiseFactory:
    _noise_handlers = {}

    @classmethod
    def register(cls, name):
        """装饰器：注册噪声处理方法"""
        def decorator(func):
            cls._noise_handlers[name] = func
            return func
        return decorator

    @classmethod
    def apply_noise(cls, image, noise_type, **kwargs):
        """工厂方法：动态调用噪声处理"""
        handler = cls._noise_handlers.get(noise_type)
        if not handler:
            raise ValueError(f"未注册的噪声类型: {noise_type}")
        return handler(image, **kwargs)

# 注册现有方法
@NoiseFactory.register('gaussian')
def _gaussian_noise(image, sigma=25):
    image_copy = image.copy()
    noise = np.random.normal(0, sigma, image_copy.shape)
    return np.clip(image_copy + noise, 0, 255).astype(np.uint8)

@NoiseFactory.register('salt_pepper')
def _salt_pepper_noise(image, prob=0.05):
    noisy = image.copy()
    salt_mask = np.random.rand(*image.shape[:2]) < prob / 2
    pepper_mask = np.random.rand(*image.shape[:2]) < prob / 2
    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0
    return noisy
