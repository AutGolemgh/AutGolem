import secrets
import math

def uniform(a: float = 0.0, b: float = 1.0)-> float:
    """Cryptographically secure uniform sample."""
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53) # in [0, 1)
    return a + (b- a) * u

def exponentialdist(lam: float) -> float:
    """Cryptographically secure exponential sample (inverse transform sampling)."""
    if lam <= 0:
        raise ValueError("lam must be > 0")
    y = 1 - uniform()
    return -math.log(y) / lam
    
