import unittest
import numpy as np
import bsp


class TestBisectingSegments(unittest.TestCase):
    def test_segment_in_front(self):
        segments = np.array([
            [[0, 1], [1, 1]],
            [[1, 1], [1, 2]],
            [[0, 1], [1, 0]]
        ])
        segment = np.array([[0, 1], [1, 1]])
        line = np.array([[0, 0], [1, 0]])

        ahead, behind, colinear = bsp.bisect(segment, line)
        self.assertTrue(np.allclose(segment, ahead))

        ahead, behind, colinear = bsp.bisect(segments, line)
        self.assertEqual(behind.size, 0)
        self.assertEqual(colinear.size, 0)
        self.assertTrue(np.allclose(segments, ahead))

    def test_segment_behind(self):
        segments = np.array([
            [[0, -1], [1, -1]],
            [[1, -1], [1, -2]],
            [[0, -1], [1, 0]]
        ])
        segment = np.array([[0, -1], [1, -1]])
        line = np.array([[0, 0], [1, 0]])

        ahead, behind, colinear = bsp.bisect(segment, line)
        self.assertEqual(behind.shape, (1,*segment.shape))
        self.assertTrue(np.allclose(segment, behind))

        ahead, behind, colinear = bsp.bisect(segments, line)
        self.assertEqual(behind.shape, segments.shape)
        self.assertEqual(ahead.size, 0)
        self.assertEqual(colinear.size, 0)
        self.assertTrue(np.allclose(segments, behind))

    def test_segment_colinear(self):
        segments = np.array([
            [[-1, 0], [1, 0]],
            [[-60, 0], [-30, 0]],
            [[2, 0], [5, 0]]
        ])
        segment = np.array([[0, 0], [1, 0]])
        line = np.array([[0, 0], [1, 0]])

        ahead, behind, colinear = bsp.bisect(segment, line)
        self.assertEqual(colinear.shape, (1,*segment.shape))
        self.assertTrue(np.allclose(segment, colinear))

        ahead, behind, colinear = bsp.bisect(segments, line)
        self.assertEqual(colinear.shape, segments.shape)
        self.assertEqual(ahead.size, 0)
        self.assertEqual(behind.size, 0)
        self.assertTrue(np.allclose(segments, colinear))

    def test_segment_bisecting(self):
        segment = np.array([[0,-1],[0,1]])
        line = np.array([[0, 0], [1, 0]])

        ahead, behind, colinear = bsp.bisect(segment, line)
        self.assertEqual(ahead.shape, (1,*segment.shape)) # should return a segment ahead and a segment behind
        self.assertEqual(behind.shape, (1,*segment.shape)) # should return a segment ahead and a segment behind
        self.assertTrue(np.allclose(np.array([[[0,-1],[0,0]]]), behind), behind)
        self.assertTrue(np.allclose(np.array([[[0,0],[0,1]]]), ahead), ahead)



if __name__ == '__main__':
    unittest.main()
