from typing import NewType
import numpy.typing as npt
import numpy as np
from torch import Tensor

# Lets make some types to make type annotation easier
State = NewType("State", npt.NDArray[np.float32])
DiscreteAction = NewType("Action", int)
Reward = NewType("Reward", float)
LogProb = NewType("LogProb", Tensor)
Loss = NewType("Loss", Tensor)
Done = NewType("Done", bool)
