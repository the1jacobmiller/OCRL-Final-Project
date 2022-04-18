import numpy as np
from typing import Callable
from rich.progress import track


class NLPProblem:
    """
    Uses IPOPT to solve the NLP we have. The obstacle constraints are the main reason we need to do this. 
    """
    def __init__(self) -> None:
        pass
