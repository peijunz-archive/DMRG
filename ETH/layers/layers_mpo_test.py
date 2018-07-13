from .LayersMPO import transform
import numpy as np
from numpy import random
import numpy.testing as npt

def test_MPO():
    A = random.rand(1000)
    U = random.rand(16)
    npt.assert_almost_equal(np.)
