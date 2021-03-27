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
        self.assertEqual(behind.shape, (1, *segment.shape))
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
        self.assertEqual(colinear.shape, (1, *segment.shape))
        self.assertTrue(np.allclose(segment, colinear))

        ahead, behind, colinear = bsp.bisect(segments, line)
        self.assertEqual(colinear.shape, segments.shape)
        self.assertEqual(ahead.size, 0)
        self.assertEqual(behind.size, 0)
        self.assertTrue(np.allclose(segments, colinear))

    def test_segment_bisecting(self):
        segment = np.array([[0, -1], [0, 1]])
        line = np.array([[0, 0], [1, 0]])

        ahead, behind, colinear = bsp.bisect(segment, line)
        self.assertEqual(ahead.shape, (1, *segment.shape))  # should return a segment ahead and a segment behind
        self.assertEqual(behind.shape, (1, *segment.shape))  # should return a segment ahead and a segment behind
        self.assertTrue(np.allclose(np.array([[[0, -1], [0, 0]]]), behind), behind)
        self.assertTrue(np.allclose(np.array([[[0, 0], [0, 1]]]), ahead), ahead)


class TestBuildingTree(unittest.TestCase):
    def test_trivial_case(self):
        # all segments should be behind eachother
        segments = np.array([
            [[0, 1], [1, 1]],
            [[0, 0], [1, 0]],
            [[0, -1], [1, -1]]
        ])

        graph = bsp.build_tree(segments)
        self.assertEqual(len(graph.nodes), 3)
        # every edge should be behind
        positions = [edge[2]['position'] for edge in graph.edges.data()]
        self.assertTrue(all([position == -1 for position in positions]))

    def test_all_colinear(self):
        # all segments should be behind eachother
        segments = np.array([
            [[0, 0], [10, 0]],
            [[0, 0], [1, 0]],
            [[-5, 0], [1, 0]]
        ])

        graph = bsp.build_tree(segments)
        # the graph should only have one node
        self.assertEqual(len(graph.nodes), 1)
        # there should be no edges
        self.assertFalse(bool(graph.edges))

        # all colinear lines should be in the first segment
        self.assertTrue(np.allclose(graph.nodes[0]['colinear_segments'], segments))

    def test_subdividing(self):
        # all segments should be behind eachother
        segments = np.array([
            [[-1, 0], [1, 0]],
            [[0, -1], [0, 1]]
        ])

        graph = bsp.build_tree(segments)

        # should have three nodes, because the latter is split
        self.assertEqual(len(graph.nodes), 3)
        # every edge should be behind

        # if the edge has a position of -1 we expect the line to be  ((0, -1),(0,0))
        for u, v, c in graph.edges.data('position'):
            if c == -1:
                self.assertTrue(np.allclose(graph.nodes[v]['line'], np.array([[0, -1], [0, 0]])))

            if c == 1:
                self.assertTrue(np.allclose(graph.nodes[v]['line'], np.array([[0, 0], [0, 1]])))


if __name__ == '__main__':
    unittest.main()
