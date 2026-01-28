import gc
import os
import random
import struct

import numpy as np
import torch


class SeedManager:
    @staticmethod
    def set_seed(seed: int) -> None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def reset_seed() -> None:
        rng_seed = struct.unpack("I", os.urandom(4))[0]
        SeedManager.set_seed(rng_seed)

    @staticmethod
    def clear_gpu_cache() -> None:
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
