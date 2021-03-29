import numpy as np
import networkx as nx
import networkx.algorithms.traversal.depth_first_search as dfs
from typing import Tuple
from copy import deepcopy
import matplotlib.pyplot as plt


def bisect(segments: np.ndarray, line: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a set of segments that are in front, behind, and coplanar to the line
    If the line bisects a segment, it will be split into two segments, one in front, one behind

    :param segment_start:
    :param segment_end:
    :param line_p1:
    :param line_p2:
    :return:
    """

    single_segment = segments.ndim == 2
    if single_segment:
        segments = segments[np.newaxis, ...]

    segment_start = segments[..., 0, :]
    segment_end = segments[..., 1, :]

    v0 = segment_end - segment_start
    v1 = line[1] - line[0]

    # need to solve for the intersection equation, first find the numerator and denominator

    numerator = np.cross((line[0] - segment_start), v1)
    denominator = np.cross(v0, v1)

    # if the denominator is zero the lines are parallel
    parallel = np.isclose(denominator, 0)
    not_parallel = np.logical_not(parallel)

    # the intersection time is the point along the line segment where the line bisects it
    intersection = numerator / (denominator + parallel)

    ahead = np.logical_or(numerator > 0, np.logical_and(np.isclose(numerator, 0), denominator < 0))
    behind = np.logical_or(numerator < 0, np.logical_and(np.isclose(numerator, 0), denominator > 0))

    # segments are colinear if they are parallel and the numerator is zero
    colinear = np.logical_and(parallel, np.isclose(numerator, 0))

    # bisected segments are segments that aren't parallel and t is in (0,1)
    bisected = np.logical_and(
        not_parallel,
        np.logical_and(intersection > 0, intersection < 1)
    )

    # bisected lines need to be split up into new segments that are ahead and behind, first separate out the bisected
    # doing this to all segments to avoid fancy indexing at first
    intersection_points = segment_start + intersection[..., np.newaxis] * v0
    l_segments = np.stack((segments[..., 0, :], intersection_points), axis=1)
    r_segments = np.stack((intersection_points, segments[..., 1, :]), axis=1)

    mask = numerator[..., np.newaxis, np.newaxis] > 0
    bisected_ahead = np.where(mask, l_segments, r_segments)[bisected]
    bisected_behind = np.where(np.logical_not(mask), l_segments, r_segments)[bisected]

    # need to return 3 sets of segments, those in front, those colinear, and those behind
    ahead_mask = np.logical_and(ahead, np.logical_not(bisected))
    behind_mask = np.logical_and(behind, np.logical_not(bisected))

    if bisected_ahead.size != 0:
        if np.any(ahead_mask):
            all_ahead = np.concatenate((segments[ahead_mask], bisected_ahead))
        else:
            all_ahead = bisected_ahead
    else:
        all_ahead = segments[ahead_mask]
    if bisected_behind.size != 0:
        if np.any(behind_mask):
            all_behind = np.concatenate((segments[behind_mask], bisected_behind))
        else:
            all_behind = bisected_behind
    else:
        all_behind = segments[behind_mask]

    all_colinear = segments[colinear]

    return all_ahead, all_behind, all_colinear


def add_root(tree: nx.DiGraph, split_line: np.ndarray, trim=True) -> nx.DiGraph:
    # create two new trees that are copies of the original tree
    root_node = list(nx.topological_sort(tree))[0]

    front_tree = nx.relabel_nodes(tree.copy(), {x: f"f-{x}" for x in tree.nodes})
    back_tree = nx.relabel_nodes(tree.copy(), {x: f"b-{x}" for x in tree.nodes})

    # will add a root node to both graphs which is what will ultimately compose them together
    # the front tree root will hold the actual attributes
    front_tree.add_node('root', line=split_line, colinear_segments=np.empty((0, 2, 2)))
    front_tree.add_edge('root', f"f-{root_node}", position=1)
    back_tree.add_node('root')
    back_tree.add_edge('root', f"b-{root_node}", position=-1)

    # for every node in the tree, bisect the segments in the node with the split line, put front segments in the front
    # tree, back segments in the back tree, and coplanar segments in the new root node
    for node in tree.nodes:
        front_node = f"f-{node}"
        back_node = f"b-{node}"
        ahead, behind, colinear = bisect(tree.nodes[node]["colinear_segments"], split_line)
        front_tree.nodes[front_node]["colinear_segments"] = ahead  # all lines in front go into the front tree
        back_tree.nodes[back_node]["colinear_segments"] = behind  # all lines behind go into the back tree

        # colinear lines are added to the root node
        front_tree.nodes['root']["colinear_segments"] = np.concatenate((
            front_tree.nodes['root']["colinear_segments"],
            colinear
        ))

    # now compose the two trees together by the root node
    rooted_tree = nx.compose(back_tree, front_tree)
    if trim:
        trim_leaves(rooted_tree)  # trim any dead leaves

    # relabel nodes based on a topological sort
    new_labels = {node: n for n, node in enumerate(nx.topological_sort(rooted_tree))}
    return nx.relabel_nodes(rooted_tree, new_labels)


def build_tree(segments: np.ndarray, starting_segment: np.ndarray = None) -> nx.DiGraph:
    def bsp_helper(segments: np.ndarray, division_line: np.ndarray, graph: nx.DiGraph):
        ahead, behind, colinear = bisect(segments, division_line)  # get the bisected segments
        node_id = id(division_line)  # make your line hashable so it's usable as a node
        graph.add_node(node_id, line=division_line, colinear_segments=colinear)  # add the node to the graph
        if behind.size != 0:  # if there's any elements behind
            node_behind = bsp_helper(behind, behind[0], graph)  # recursively call for all segments behind
            graph.add_edge(node_id, node_behind, position=-1)  # add an edge from this node to the behind node
        if ahead.size != 0:
            node_ahead = bsp_helper(ahead, ahead[0], graph)  # recursively call for all segments in front
            graph.add_edge(node_id, node_ahead, position=1)  # add an edge from this node to the front node
        return node_id  # return the hashed id

    graph = nx.DiGraph()  # make a new directed graph
    if starting_segment is None:
        starting_segment = segments[0]

    # run the recursive helper function, which should add all nodes and edges
    bsp_helper(segments, starting_segment, graph)
    return nx.relabel.convert_node_labels_to_integers(graph)


def project_tree(T0: nx.DiGraph, T1: nx.DiGraph, trim=True) -> nx.DiGraph:
    """
    Returns a new graph where T0 has been projected into T1's bisecting planes
    :param T0:
    :param T1:
    :return:
    """

    def merge_helper(merge_onto_graph: nx.DiGraph, merge_node: object, to_merge_subgraph: nx.DiGraph) -> nx.DiGraph:
        # insert a node onto the subgraph which has the same line as the merge graph
        rooted_graph = add_root(to_merge_subgraph, merge_onto_graph.nodes[merge_node]["line"], trim=False)
        trim_leaves(rooted_graph, trim_root=False)
        root_node = list(nx.topological_sort(rooted_graph))[0]

        ahead_child = get_child_ahead(merge_onto_graph, merge_node)
        behind_child = get_child_behind(merge_onto_graph, merge_node)

        rooted_ahead = get_child_ahead(rooted_graph, root_node)
        rooted_behind = get_child_behind(rooted_graph, root_node)

        if rooted_ahead is not None:
            ahead_subgraph = rooted_graph.subgraph(dfs.dfs_tree(rooted_graph, rooted_ahead))
        else:
            ahead_subgraph = nx.DiGraph()

        if rooted_behind is not None:
            behind_subgraph = rooted_graph.subgraph(dfs.dfs_tree(rooted_graph, rooted_behind))
        else:
            behind_subgraph = nx.DiGraph()
        # check if the graph being merged onto has a ahead child, if so take the ahead child of the rooted graph
        # and insert it after this node
        if ahead_child is not None and (rooted_ahead is not None):
            # recursively re-enter this function and build a subgraph of all children ahead

            # this is going to be a subgraph with the root at node zero
            ahead_subgraph = merge_helper(merge_onto_graph,
                                          ahead_child,
                                          ahead_subgraph)

        if behind_child is not None and (rooted_behind is not None):
            # recursively re-enter this function and build a subgraph of all children ahead

            # this is going to be a subgraph with the root at node zero
            behind_subgraph = merge_helper(merge_onto_graph,
                                           behind_child,
                                           behind_subgraph)

        # now we want to do the disjoint union of both graphs
        inserted_tree = nx.disjoint_union(ahead_subgraph, behind_subgraph)
        # copy the colinear root node too
        inserted_tree.add_node('root',
                               line=rooted_graph.nodes[root_node]['line'],
                               colinear_segments=rooted_graph.nodes[root_node]["colinear_segments"])

        # now connect the root node to the
        n_ahead = ahead_subgraph.number_of_nodes()
        n_behind = behind_subgraph.number_of_nodes()
        # if there's any nodes ahead add an edge
        if n_ahead:
            inserted_tree.add_edge('root', 0, position=1)
        if n_behind:
            inserted_tree.add_edge('root', n_ahead, position=-1)

        # return the inserted tree
        return inserted_tree



    # generate and return the graph projected into the new space
    t1_root = list(nx.topological_sort(T1))[0]
    merged_graph = merge_helper(T1, t1_root, T0)
    if trim:
        trim_leaves(merged_graph)

    new_labels = {node: n for n, node in enumerate(nx.topological_sort(merged_graph))}
    return nx.relabel_nodes(merged_graph, new_labels)


def clip_to(T0: nx.DiGraph, T1: nx.DiGraph) -> nx.DiGraph:
    """
    Clips BSP tree T0 to T1, returning a tree containing only polygons from T0 that fall within T1
    :param T0:
    :param T1:
    :return:
    """
    projected_tree = project_tree(T0, T1)



def trim_leaves(tree: nx.DiGraph, trim_root=True) -> None:
    # removes any nodes that are empty and have less than two out_edges

    # get a set of nodes that don't have any colinear segments
    empty_nodes = [key for key, value in nx.get_node_attributes(tree, "colinear_segments").items() if
                   np.size(value) == 0]

    # need make sure the nodes are in reverse topological order or parents won't be trimmed even if their children are
    empty_nodes = reversed([node for node in nx.topological_sort(tree) if node in empty_nodes])

    # iterate over all empty nodes
    for node in empty_nodes:
        # if it's empty and has 1 or fewer children it can be deleted
        # second argument prevents deleting the root node

        if tree.out_degree(node) < 2 and (tree.in_degree(node)!=0 or trim_root):
            # if there's an in edge and an out edge, join the two otherwise just delete it
            if tree.in_degree(node) == 1 and tree.out_degree(node) == 1:
                parent_edge = tuple(tree.in_edges(node, data="position"))[0]
                parent_node = parent_edge[0]
                parent_direction = parent_edge[2]
                child_node = tuple(tree.out_edges(node))[0][1]
                tree.add_edge(parent_node, child_node, position=parent_direction)

            # delete the node
            tree.remove_node(node)


def get_child_ahead(graph: nx.DiGraph, node: object) -> object:
    for (u, v, c) in graph.out_edges(node, data="position"):
        if c == 1:
            return v

    return None


def get_child_behind(graph: nx.DiGraph, node: object) -> object:
    for (u, v, c) in graph.out_edges(node, data="position"):
        if c == -1:
            return v

    return None


def draw_segments(tree: nx.DiGraph) -> None:
    fig = plt.figure()
    axis = plt.gca()
    all_segments = np.concatenate([value for value in dict(nx.get_node_attributes(tree, "colinear_segments")).values()])
    for segment in all_segments:
        axis.plot(*(segment.T), "k-o", linewidth=3, markersize=12)
    plt.show()
