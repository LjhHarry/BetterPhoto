"""
Data utilities -- shared data generation and processing for all experiments
数据工具 —— 所有实验共用的数据生成和处理函数

Support: synthetic degradation data, data augmentation, DataLoader construction
支持：合成退化数据、数据增强、Dataloader构建

v3.0: Added SDImageDataset using SD v1.5 for training image generation
v3.0: 新增 SDImageDataset 使用 SD v1.5 生成训练图片
"""
import os
import io
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Callable
from PIL import Image

# SD v1.5 prompt list -- standard photographic diversity prompts (shared with B1)
# SD v1.5 提示词列表 —— 标准摄影多样性提示（与 B1 共用）
_SD_PROMPTS = [
    "professional portrait photo, studio lighting, 85mm lens, sharp focus",
    "detailed macro photo of a flower, morning dew, natural sunlight, bokeh",
    "street photography, urban landscape, golden hour lighting",
    "landscape photo of mountains at sunset, dramatic clouds, wide angle",
    "product photography of a watch, reflective surface, studio lighting",
    "food photography, restaurant presentation, warm lighting, top-down view",
    "architectural photography of a modern building, clean lines, blue sky",
    "still life photography of fruits in a bowl, kitchen, soft diffused light",
    "night cityscape, neon lights, wet pavement reflections, cinematic",
    "close-up of a cat with green eyes, natural window light, shallow DOF",
    "fashion photography, editorial style, natural outdoor location",
    "aerial photography of a coastline, turquoise water, white sand",
    "black and white portrait, dramatic lighting, film grain, classic look",
    "wildlife photography of a bird in flight, frozen motion, natural habitat",
    "interior design photo of a living room, natural light, cozy minimal style",
    "macro photo of water droplets on a leaf, refraction, morning light",
    "vintage car detail shot, chrome reflections, sunny day, retro styling",
    "portrait of an elderly person, character wrinkles, high detail, natural light",
    "aerial view of a forest in autumn, vibrant colors, drone photography",
    "sports photography, runner in dynamic pose, motion blur background",
]

# SD v1.5 local cache path
# SD v1.5 本地缓存路径
_SD_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stable-diffusion-v1-5",
)

# ---- Synthetic degradation data generation ----
# ---- 合成退化数据生成 ----
def generate_synthetic_pair(size: int = 64, num_defects: int = 3, 
                             defect_size: int = 8, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single pair of synthetic degradation data
    生成单对合成退化数据

    Args:
        size: Image size
        size: 图像尺寸
        num_defects: Number of defect blocks
        num_defects: 缺陷块数量
        defect_size: Defect block size
        defect_size: 缺陷块大小

    Returns:
        dirty: Degraded image (size, size, 3)
        dirty: 退化图像 (size, size, 3)
        clean: Clean image (size, size, 3)
        clean: 干净图像 (size, size, 3)
        gt_mask: Defect mask (size, size, 1)
        gt_mask: 缺陷掩膜 (size, size, 1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random gradient color background
    # 随机渐变色背景
    base_val = np.random.rand() * 0.7 + 0.3
    variation = np.random.rand() * 0.3
    clean = np.random.rand(size, size, 3) * variation + base_val
    clean = np.clip(clean, 0.0, 1.0).astype(np.float32)
    
    dirty = clean.copy()
    gt_mask = np.zeros((size, size, 1), dtype=np.float32)
    
    for _ in range(num_defects):
        x = np.random.randint(0, size - defect_size)
        y = np.random.randint(0, size - defect_size)
        is_black = np.random.rand() > 0.5
        color = 0.0 if is_black else 1.0
        
        dirty[y:y+defect_size, x:x+defect_size, :] = color
        gt_mask[y:y+defect_size, x:x+defect_size, 0] = 1.0
    
    return dirty, clean, gt_mask


def generate_batch_synthetic(batch_size: int, size: int = 64, 
                              num_defects: int = 3, defect_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of synthetic degradation data
    批量生成合成退化数据
    """
    dirty_list, clean_list, mask_list = [], [], []
    for _ in range(batch_size):
        d, c, m = generate_synthetic_pair(size, num_defects, defect_size)
        dirty_list.append(d.transpose(2, 0, 1))
        clean_list.append(c.transpose(2, 0, 1))
        mask_list.append(m.transpose(2, 0, 1))
    
    return (torch.from_numpy(np.stack(dirty_list)),
            torch.from_numpy(np.stack(clean_list)),
            torch.from_numpy(np.stack(mask_list)))


# ---- Multiple degradation types ----
# ---- 多种退化类型 ----
def apply_gaussian_blur(img: np.ndarray, num_blobs: int = 3, 
                         max_sigma: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gaussian blur degradation (simulates camera shake/defocus)
    应用高斯模糊退化 (模拟相机抖动/失焦)
    """
    from scipy.ndimage import gaussian_filter
    
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 1), dtype=np.float32)
    degraded = img.copy()
    
    for _ in range(num_blobs):
        cy, cx = np.random.randint(0, h), np.random.randint(0, w)
        radius = np.random.randint(h//8, h//4)
        sigma = np.random.uniform(2.0, max_sigma)
        
        y1, y2 = max(0, cy-radius), min(h, cy+radius)
        x1, x2 = max(0, cx-radius), min(w, cx+radius)
        
        patch = degraded[y1:y2, x1:x2]
        for c in range(3):
            degraded[y1:y2, x1:x2, c] = gaussian_filter(patch[..., c], sigma=sigma)
        mask[y1:y2, x1:x2, 0] = 1.0
    
    return np.clip(degraded, 0, 1), mask


def apply_jpeg_artifact(img: np.ndarray, num_regions: int = 3,
                         quality: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply JPEG compression artifact degradation
    应用 JPEG 压缩伪影退化
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 1), dtype=np.float32)
    degraded = img.copy()
    
    for _ in range(num_regions):
        cy, cx = np.random.randint(0, h), np.random.randint(0, w)
        radius = np.random.randint(h//8, h//3)
        
        y1, y2 = max(0, cy-radius), min(h, cy+radius)
        x1, x2 = max(0, cx-radius), min(w, cx+radius)
        
        patch = (degraded[y1:y2, x1:x2] * 255).astype(np.uint8)
        pil_img = Image.fromarray(patch)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        degraded_patch = np.array(Image.open(buf)).astype(np.float32) / 255.0
        
        # Handle possible size mismatch
        # 处理可能的尺寸不匹配
        if degraded_patch.shape[:2] == (y2-y1, x2-x1):
            degraded[y1:y2, x1:x2] = degraded_patch
        mask[y1:y2, x1:x2, 0] = 1.0
    
    return np.clip(degraded, 0, 1), mask


def apply_pixel_noise(img: np.ndarray, salt_pepper_prob: float = 0.05,
                       gaussian_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply pixel noise degradation
    应用像素噪声退化
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 1), dtype=np.float32)
    degraded = img.copy()
    
    # Global noise
    # 全局噪声
    noise = np.random.randn(h, w, 3) * gaussian_std
    degraded += noise
    
    # Salt-and-pepper noise
    # 椒盐噪声
    salt_mask = np.random.rand(h, w, 3) < salt_pepper_prob / 2
    pepper_mask = np.random.rand(h, w, 3) < salt_pepper_prob / 2
    degraded[salt_mask] = 1.0
    degraded[pepper_mask] = 0.0
    
    # Noise mask = global (used for training localization)
    # 噪声掩膜 = 全局 (用于训练定位)
    mask[:, :, 0] = 1.0
    
    return np.clip(degraded, 0, 1), mask


def apply_random_degradation(img: np.ndarray, degradation_type: Optional[str] = None
                              ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Randomly select a degradation type to apply
    随机选择一种退化类型应用
    """
    if degradation_type is None:
        degradation_type = np.random.choice(
            ["block", "blur", "jpeg", "noise", "mixed"])
    
    if degradation_type == "block":
        d, m = _apply_block_occlusion(img)
    elif degradation_type == "blur":
        d, m = apply_gaussian_blur(img)
    elif degradation_type == "jpeg":
        d, m = apply_jpeg_artifact(img)
    elif degradation_type == "noise":
        d, m = apply_pixel_noise(img)
    elif degradation_type == "mixed":
        d, m = apply_gaussian_blur(img)
        d, m2 = apply_jpeg_artifact(d)
        m = np.maximum(m, m2)
    else:
        d, m = _apply_block_occlusion(img)
    
    return d, m, degradation_type


def _apply_block_occlusion(img: np.ndarray, num_blocks: int = 4,
                            max_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply block occlusion degradation
    应用块遮挡退化
    """
    h, w = img.shape[:2]
    degraded = img.copy()
    mask = np.zeros((h, w, 1), dtype=np.float32)
    
    for _ in range(num_blocks):
        bh = np.random.randint(4, max_size)
        bw = np.random.randint(4, max_size)
        y = np.random.randint(0, h - bh)
        x = np.random.randint(0, w - bw)
        
        # Adaptive color (contrast with background)
        # 自适应颜色（与背景形成对比）
        local_mean = degraded[y:y+bh, x:x+bw].mean()
        color = 0.0 if local_mean > 0.5 else 1.0
        
        degraded[y:y+bh, x:x+bw, :] = color
        mask[y:y+bh, x:x+bw, 0] = 1.0
    
    return degraded, mask


# ---- Real image loading ----
# ---- 真实图片加载 ----
def load_real_images(directory: str, target_size: int = 64,
                      max_images: int = None) -> List[np.ndarray]:
    """
    Load real images from directory and resize
    从目录加载真实图片并调整大小
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = []
    
    files = [f for f in os.listdir(directory) 
             if os.path.splitext(f)[1].lower() in supported]
    
    if max_images:
        files = files[:max_images]
    
    for f in files:
        path = os.path.join(directory, f)
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((target_size, target_size), Image.LANCZOS)
            img_np = np.array(img).astype(np.float32) / 255.0
            images.append(img_np)
        except Exception:
            continue
    
    return images


# ---- Dataset classes ----
# ---- Dataset 类 ----
class SyntheticDataset(Dataset):
    """
    Synthetic degradation dataset -- online generation
    合成退化数据集 —— 在线生成
    """
    
    def __init__(self, num_samples: int, size: int = 64, 
                 num_defects: int = 3, defect_size: int = 8,
                 degradation_fn: Optional[Callable] = None,
                 seed: int = 42):
        self.num_samples = num_samples
        self.size = size
        self.num_defects = num_defects
        self.defect_size = defect_size
        self.degradation_fn = degradation_fn
        np.random.seed(seed)
        # Pre-generate random seeds for reproducibility
        # 预生成随机种子以确保可重复性
        self.seeds = np.random.randint(0, 2**31-1, size=num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        np.random.seed(self.seeds[idx])
        
        if self.degradation_fn:
            dirty, clean, gt_mask = self.degradation_fn(
                size=self.size, seed=self.seeds[idx])
        else:
            dirty, clean, gt_mask = generate_synthetic_pair(
                self.size, self.num_defects, self.defect_size, 
                seed=self.seeds[idx])
        
        return (torch.from_numpy(dirty.transpose(2, 0, 1)).float(),
                torch.from_numpy(clean.transpose(2, 0, 1)).float(),
                torch.from_numpy(gt_mask.transpose(2, 0, 1)).float())


class RealImageDataset(Dataset):
    """
    Real image degradation dataset
    真实图片退化数据集
    """
    
    def __init__(self, images: List[np.ndarray], num_samples: int,
                 degradation_types: List[str] = None, seed: int = 42):
        self.images = images
        self.num_samples = num_samples
        self.degradation_types = degradation_types or ["block", "blur", "jpeg", "noise"]
        np.random.seed(seed)
        self.seeds = np.random.randint(0, 2**31-1, size=num_samples)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        np.random.seed(self.seeds[idx])
        
        # Randomly select an image
        # 随机选择一张图片
        img_idx = np.random.randint(0, len(self.images))
        clean = self.images[img_idx].copy()
        
        # Random degradation
        # 随机退化
        deg_type = np.random.choice(self.degradation_types)
        dirty, gt_mask, _ = apply_random_degradation(clean, deg_type)
        
        return (torch.from_numpy(dirty.transpose(2, 0, 1)).float(),
                torch.from_numpy(clean.transpose(2, 0, 1)).float(),
                torch.from_numpy(gt_mask.transpose(2, 0, 1)).float())


# ---- SD v1.5 image generation Dataset ----
# ---- SD v1.5 图片生成 Dataset ----
# Global SD pipeline cache (singleton) to avoid repeated loading
# 全局 SD pipeline 缓存（单例），避免重复加载
_sd_pipe_cache = {"pipe": None, "device": None, "loaded": False}


def _load_sd_pipeline(device, model_path=None, force_reload=False):
    """
    Load SD v1.5 pipeline (singleton cache). local_files_only=True for offline cloud GPU.
    加载 SD v1.5 pipeline（单例缓存）。local_files_only=True 适配无网云 GPU。
    """
    if _sd_pipe_cache["loaded"] and not force_reload:
        if _sd_pipe_cache["device"] == str(device):
            return _sd_pipe_cache["pipe"]

    if model_path is None:
        model_path = _SD_CACHE_DIR

    print(f"\n  [SD] Loading Stable Diffusion v1.5 from: {model_path}")
    try:
        from diffusers import StableDiffusionPipeline

        if os.path.isdir(model_path) and os.path.isdir(os.path.join(model_path, "unet")):
            print(f"  [SD] Using local cache (local_files_only=True)")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path, torch_dtype=torch.float16,
                safety_checker=None, local_files_only=True,
            )
        else:
            print(f"  [SD] Local cache not found at {model_path}")
            print(f"  [SD] Falling back to synthetic data")
            return None

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        print(f"  [SD] Pipeline loaded successfully")
    except Exception as e:
        print(f"  [SD] Failed to load pipeline: {e}")
        print(f"  [SD] Falling back to synthetic data")
        return None

    _sd_pipe_cache["pipe"] = pipe
    _sd_pipe_cache["device"] = str(device)
    _sd_pipe_cache["loaded"] = True
    return pipe


def _generate_sd_images(pipe, num_images, size, seed, device, prompts=None, batch_size=2):
    """
    Generate images in batch using SD v1.5, return list of numpy arrays
    使用 SD v1.5 批量生成图片，返回 numpy 数组列表。
    """
    if prompts is None:
        prompts = _SD_PROMPTS

    n_prompts = len(prompts)
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []

    print(f"  [SD] Generating {num_images} images at {size}x{size}...")
    t0 = time.time()

    idx = 0
    while len(images) < num_images:
        pb = []
        for _ in range(min(batch_size, num_images - len(images))):
            pb.append(prompts[idx % n_prompts])
            idx += 1

        with torch.autocast(device_type=str(device)):
            outputs = pipe(pb, height=size, width=size,
                          num_inference_steps=30, guidance_scale=7.5,
                          generator=generator).images

        for img in outputs:
            img = img.resize((size, size), Image.LANCZOS)
            img_np = np.array(img).astype(np.float32) / 255.0
            images.append(img_np)

        if len(images) % 20 == 0:
            print(f"  [SD] Generated {len(images)}/{num_images}...")

    elapsed = time.time() - t0
    print(f"  [SD] Done: {len(images)} images in {elapsed:.1f}s ({elapsed/max(len(images),1):.2f}s/img)")
    return images


class SDImageDataset(Dataset):
    """Training dataset that pre-generates high-quality images using SD v1.5, then applies degradation.

    Auto-cached in resources/ directory: images with the same (size, num_samples, seed)
    are generated only once; subsequent experiments load them directly to avoid redundant GPU time.
    使用 SD v1.5 预生成高质量图片，再施加退化的训练数据集。

    自动缓存到 resources/ 目录：相同 (size, num_samples, seed) 的图片只生成一次，
    后续实验直接加载，避免重复消耗 GPU 时间。
    """

    # Global cache directory: shared SD pre-generated images for all experiments
    # 全局缓存目录：所有实验共享的 SD 预生成图片
    _RESOURCES_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "resources", "sd_images")

    def __init__(self, num_samples, size=64, degradation_types=None,
                 degradation_fn=None, sd_model_path=None, seed=42,
                 device="cuda", cache_dir=None):
        self.num_samples = num_samples
        self.size = size
        self.degradation_types = degradation_types or ["block"]
        self.degradation_fn = degradation_fn
        self.seed = seed

        np.random.seed(seed)
        self.seeds = np.random.randint(0, 2**31 - 1, size=num_samples)

        # Prefer loading cache from resources/ (overrides cache_dir param for unified management)
        # 优先从 resources/ 加载缓存（覆盖 cache_dir 参数，统一管理）
        cache_dir = cache_dir or self._RESOURCES_DIR
        cache_path = os.path.join(
            cache_dir, f"sd_s{size}_n{num_samples}_seed{seed}.pt")

        if os.path.exists(cache_path):
            # Exact match: load directly
            # 精确匹配：直接加载
            print(f"  [SDImageDataset] Loading cached SD images from {cache_path}")
            self.clean_images = [img.numpy() for img in torch.load(cache_path, weights_only=False)]
            print(f"  [SDImageDataset] Loaded {len(self.clean_images)} cached images")

        elif os.path.isdir(cache_dir):
            # Look for larger cache (same size + seed, but larger num_samples), take subset
            # 查找更大的缓存（相同 size + seed，但 num_samples 更大），取子集
            import glob
            pattern = os.path.join(cache_dir, f"sd_s{size}_n*_seed{seed}.pt")
            candidates = glob.glob(pattern)
            # Parse num_samples, filter for those >= required
            # 解析 num_samples，筛选出 >= 需求的
            usable = []
            for c in candidates:
                try:
                    n = int(os.path.basename(c).split("_n")[1].split("_")[0])
                    if n >= num_samples:
                        usable.append((n, c))
                except (IndexError, ValueError):
                    continue
            if usable:
                # Pick the smallest valid one (save loading time)
                # 选最小的满足条件的（节省加载时间）
                usable.sort()
                _, best_path = usable[0]
                print(f"  [SDImageDataset] Loading {num_samples} from larger cache: {best_path}")
                all_imgs = [img.numpy() for img in torch.load(best_path, weights_only=False)]
                self.clean_images = all_imgs[:num_samples]
                print(f"  [SDImageDataset] Loaded {len(self.clean_images)} images (subset of {len(all_imgs)})")

            else:
                # No usable cache, generate new images
                # 没有可用缓存，生成新图片
                device_obj = torch.device(device)
                pipe = _load_sd_pipeline(device_obj, model_path=sd_model_path)

                if pipe is None:
                    print(f"  [SDImageDataset] SD unavailable, fallback to random noise")
                    self.clean_images = None
                else:
                    self.clean_images = _generate_sd_images(
                        pipe, num_samples, size, seed, device_obj)
                    # Save to resources/ for reuse by subsequent experiments
                    # 保存到 resources/ 供后续实验复用
                    os.makedirs(cache_dir, exist_ok=True)
                    torch.save([torch.from_numpy(img) for img in self.clean_images], cache_path)
                    print(f"  [SDImageDataset] Saved {len(self.clean_images)} images to {cache_path}")

        else:
            # Load SD or fallback
            # 加载 SD 或降级
            device_obj = torch.device(device)
            pipe = _load_sd_pipeline(device_obj, model_path=sd_model_path)

            if pipe is None:
                print(f"  [SDImageDataset] SD unavailable, fallback to random noise")
                self.clean_images = None
            else:
                self.clean_images = _generate_sd_images(
                    pipe, num_samples, size, seed, device_obj)
                # Save to resources/ for reuse by subsequent experiments
                # 保存到 resources/ 供后续实验复用
                os.makedirs(cache_dir, exist_ok=True)
                torch.save([torch.from_numpy(img) for img in self.clean_images], cache_path)
                print(f"  [SDImageDataset] Saved {len(self.clean_images)} images to {cache_path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        np.random.seed(self.seeds[idx])

        if self.clean_images is not None:
            clean = self.clean_images[idx % len(self.clean_images)].copy()
        else:
            base = np.random.rand() * 0.7 + 0.3
            var = np.random.rand() * 0.3
            clean = np.clip(
                np.random.rand(self.size, self.size, 3) * var + base,
                0, 1).astype(np.float32)

        if self.degradation_fn:
            dirty, gt_mask = self.degradation_fn(clean)
        else:
            deg_type = np.random.choice(self.degradation_types)
            dirty, gt_mask, _ = apply_random_degradation(clean, deg_type)

        return (torch.from_numpy(dirty.transpose(2, 0, 1)).float(),
                torch.from_numpy(clean.transpose(2, 0, 1)).float(),
                torch.from_numpy(gt_mask.transpose(2, 0, 1)).float())

# ---- DataLoader construction ----
# ---- DataLoader 构建 ----
def create_dataloaders(config, degradation_fn=None) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test DataLoaders
    创建训练和测试 DataLoader
    """
    if degradation_fn is None:
        train_dataset = SyntheticDataset(
            num_samples=config.train_samples,
            size=config.image_size,
            seed=config.seed,
        )
        test_dataset = SyntheticDataset(
            num_samples=config.test_samples,
            size=config.image_size,
            seed=config.seed + 1000,
        )
    else:
        train_dataset = SyntheticDataset(
            num_samples=config.train_samples,
            size=config.image_size,
            degradation_fn=degradation_fn,
            seed=config.seed,
        )
        test_dataset = SyntheticDataset(
            num_samples=config.test_samples,
            size=config.image_size,
            degradation_fn=degradation_fn,
            seed=config.seed + 1000,
        )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader
