import numpy as np
import networkx as nx
import bsp
import matplotlib.pyplot as plt

segments = np.array([
    [[-1.5, 0], [2, 0]],
    [[-2, -1], [-2, 1]],
    [[2, -2], [6, 2]],
    [[-1, -4], [-4, 2]]
])

tree = bsp.build_tree(segments)

fig = plt.figure(figsize=(8,8))
axis = plt.subplot(2,1,1)
axis.grid()
for segment in segments:
    axis.plot(*(segment.T), "o-", color='k', linewidth=3, markersize=12)

ax2 = plt.subplot(2,1,2)
for _,segments in tree.nodes.data('colinear_segments'):
    for segment in segments:
        ax2.plot(*(segment.T), "o-", linewidth=3, markersize=12)

ax2.grid()

ax2.set_xlim(axis.get_xlim())
ax2.set_ylim(axis.get_ylim())
plt.show()