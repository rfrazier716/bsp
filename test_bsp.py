import unittest
import numpy as np
import bsp
import networkx as nx


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


class TestAddingRootNode(unittest.TestCase):
    def test_single_node_graph(self):
        segments = np.array([
            [[-1, 0], [1, 0]]
        ])
        split_line = np.array([[0, -1], [0, 1]])

        tree = bsp.build_tree(segments)
        new_tree = bsp.add_root(tree, split_line)
        # new tree should have three nodes
        self.assertEqual(new_tree.number_of_nodes(), 3)

        # the root node should have no segments
        self.assertEqual(new_tree.nodes[0]["colinear_segments"].size, 0)

    def test_adding_root_that_does_not_bisect(self):
        segments = np.array([
            [[-1, 0], [1, 0]]
        ])
        split_line = np.array([[-1, -1], [1, -1]])

        tree = bsp.build_tree(segments)
        new_tree = bsp.add_root(tree, split_line)
        # new tree should have only one node
        self.assertEqual(new_tree.number_of_nodes(), 1)

        # all segments are in that one node
        self.assertTrue(np.allclose(new_tree.nodes[0]["colinear_segments"], segments))

    def test_adding_colinear_root(self):
        segments = np.array([
            [[-1, 0], [1, 0]]
        ])

        tree = bsp.build_tree(segments)
        new_tree = bsp.add_root(tree, segments[0])

        # the tree should still have only one node
        self.assertEqual(new_tree.number_of_nodes(), 1)

        # all segments are in that one node
        self.assertTrue(np.allclose(new_tree.nodes[0]["colinear_segments"], segments))

    def test_adding_root_above(self):
        segments = np.array([
            [[-1, 0], [1, 0]]
        ])
        split_line = np.array([[0, -1], [0, 1]])
        tree = bsp.build_tree(segments, split_line)

        split_line = np.array(
            [[-1, 5], [1, 5]]
        )

        new_tree = bsp.add_root(tree, split_line)
        self.assertTrue(new_tree.number_of_nodes(),4)


class TestMergingGraphs(unittest.TestCase):
    def test_merging_all_behind(self):
        segments = np.array([
            [[1, -1], [-1, -1]],
            [[-1, -1], [-1, 1]],
            [[-1, 1], [1, 1]],
            [[1, 1], [1, -1]]
        ])

        segments2 = np.array([
            [[2, -2], [0, -2]],
            [[0, -2], [0, 0]],
            [[0, 0], [2, 0]],
            [[2, 0], [2, -2]]
        ])

        T0 = bsp.build_tree(segments)
        T1 = bsp.build_tree(segments2)
        merged = bsp.project_tree(T0, T1, trim=False)
        bsp.draw_segments(merged)



class TestTrimmingNodes(unittest.TestCase):

    def setUp(self) -> None:
        # make a basic dummy tree where every child is behind the parent
        self.tree = nx.DiGraph()
        for j in range(3):
            self.tree.add_node(j, colinear_segments=np.zeros((1, 2, 2)))
            if j != 0:
                self.tree.add_edge(j - 1, j, position=-1)

    def test_trimming_bottom_tree(self):

        # make the last tree empty
        self.tree.nodes[2]["colinear_segments"] = np.empty((0, 2, 2))

        bsp.trim_leaves(self.tree)
        self.assertEqual(self.tree.number_of_nodes(), 2)

        # the third node should have been deleted
        self.assertEqual((0, 1), tuple(self.tree.nodes))

    def test_trimming_top_of_tree(self):

        # make the root node empty
        self.tree.nodes[0]["colinear_segments"] = np.empty((0, 2, 2))

        bsp.trim_leaves(self.tree)
        self.assertEqual(self.tree.number_of_nodes(), 2)

        # the third node should have been deleted
        self.assertEqual((1, 2), tuple(self.tree.nodes))

    def test_keeping_root_node(self):
        # make the root node empty
        self.tree.nodes[0]["colinear_segments"] = np.empty((0, 2, 2))

        # trimming should have no effect
        bsp.trim_leaves(self.tree, trim_root=False)
        self.assertEqual(self.tree.number_of_nodes(), 3)

        # the third node should have been deleted
        self.assertEqual((0, 1, 2), tuple(self.tree.nodes))


    def test_trimming_middle_of_tree(self):
        # make the middle node empty
        self.tree.nodes[1]["colinear_segments"] = np.empty((0, 2, 2))

        bsp.trim_leaves(self.tree)
        self.assertEqual(self.tree.number_of_nodes(), 2)

        # the third node should have been deleted
        self.assertEqual((0, 2), tuple(self.tree.nodes))

        # there should be an edge between node 0 and 2 now
        self.assertTrue(self.tree.has_edge(0, 2))

        # the back child of node 0 should now be node 2
        # there should be an edge between node 0 and 2 now
        self.assertTrue(bsp.get_child_behind(self.tree, 0), 2)

    def test_keeping_empty_node_with_two_children(self):
        # make the middle node empty but add a front child
        self.tree.nodes[1]["colinear_segments"] = np.empty((0, 2, 2))
        self.tree.add_node(3, colinear_segments=np.zeros((1, 2, 2)))
        self.tree.add_edge(1, 3, position=1)

        bsp.trim_leaves(self.tree)
        self.assertEqual(self.tree.number_of_nodes(), 4)

        # but if that third node is empty, both 1 and 3 are trimmed
        self.tree.nodes[3]["colinear_segments"] = np.empty((0, 2, 2))
        bsp.trim_leaves(self.tree)
        self.assertEqual(self.tree.number_of_nodes(), 2)


if __name__ == '__main__':
    unittest.main()
