from .base import IAllocator
from .voting import VotingAllocator
from .ilp import ILPAllocator
from .jv import JVAllocator
from .cbba import CBBAAllocator
from .ssia import SSIAAllocator

ALLOCATORS = {
    "voting": VotingAllocator,  # needs extra args at init (method, k, etc.)
    "ilp": ILPAllocator,
    "jv": JVAllocator,
    "cbba": CBBAAllocator,
    "ssia": SSIAAllocator,
}
