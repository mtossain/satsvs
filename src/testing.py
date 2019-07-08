import unittest
import misc_fn
import numpy as np

class TestBasic(unittest.TestCase):

    def test_basic(self):
        a=np.array([1,2,3])
        b=np.array([3,4,5])
        c = misc_fn.plane_normal(a, b)
        c2 = list(np.cross(a,b).flatten())
        assert c==c2
