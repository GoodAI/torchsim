from typing import Tuple

from torch import Tensor

class Stream: ...

class Device: ...

def synchronize() -> None: ...

def stream(stream: Stream) -> None: ...

def current_stream() -> Stream: ...

def empty_cache() -> None: ...

def memory_allocated() -> float: ...

def max_memory_allocated() -> float: ...

def manual_seed(seed: int) -> None: ...

def manual_seed_all(seed: int) -> None: ...

def is_available() -> bool: ...

def device_count() -> int: ...

def device_of(tensor: Tensor) -> int: ...

def get_device_capability(device: int) -> Tuple[int, int]: ...

def memory_cached(device: int = None) -> int: ...

def max_memory_cached(device: int = None) -> int: ...

class cudaStatus(object): ...

def check_error(res: cudaStatus): ...