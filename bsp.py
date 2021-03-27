import numpy as np
import networkx as nx

def bisect(segments: np.ndarray, line: np.ndarray):
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

    ahead = numerator > 0
    behind = numerator < 0

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
            all_ahead = np.stack((segments[ahead_mask], bisected_ahead))
        else:
            all_ahead = bisected_ahead
    else:
        all_ahead = segments[ahead_mask]
    if bisected_behind.size != 0:
        if np.any(behind_mask):
            all_behind = np.stack((segments[behind_mask], bisected_behind))
        else: all_behind = bisected_behind
    else:
        all_behind = segments[behind_mask]

    all_colinear = segments[colinear]

    return all_ahead, all_behind, all_colinear


def build_tree(segments: np.ndarray, starting_segment:np.ndarray = None) -> nx.DiGraph:

    def bsp_helper(segments: np.ndarray, division_line: np.ndarray, graph: nx.DiGraph):
        ahead, behind, colinear = bisect(segments, division_line) # get the bisected segments
        node_id = id(division_line) # make your line hashable so it's usable as a node
        graph.add_node(node_id, line=division_line, colinear_segments = colinear) # add the node to the graph
        if behind.size!=0: # if there's any elements behind
            node_behind = bsp_helper(behind, behind[0], graph) # recursively call for all segments behind
            graph.add_edge(node_id, node_behind, position=-1) # add an edge from this node to the behind node
        if ahead.size!=0:
            node_ahead = bsp_helper(ahead, ahead[0], graph) # recursively call for all segments in front
            graph.add_edge(node_id, node_ahead, position=1) # add an edge from this node to the front node
        return node_id # return the hashed id

    graph = nx.DiGraph() # make a new directed graph
    if starting_segment is None:
        starting_segment = segments[0]

    # run the recursive helper function, which should add all nodes and edges
    bsp_helper(segments, starting_segment, graph)
    return nx.relabel.convert_node_labels_to_integers(graph)