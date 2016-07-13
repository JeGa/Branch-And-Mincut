import numpy as np
import logging
import networkx
import os
from scipy import misc
import maxflow


def loadunaryfile(filename):
    file = open(filename, "r")

    xsize = int(file.readline())
    ysize = int(file.readline())
    labels = int(file.readline())

    data = np.empty((ysize, xsize, labels))

    for x in range(xsize):
        for y in range(ysize):
            for l in range(labels):
                data[y, x, l] = float(file.readline())

    return data


def readimg_normalize(imagename):
    img = misc.imread(os.path.join("data", imagename))
    img = np.array(img, dtype=np.float64) / 255
    return img


def readimg(imagename):
    return misc.imread(os.path.join("data", imagename))


def readimg_grayscale(imagename):
    return misc.imread(os.path.join("data", imagename), 'L')


class Node:
    def __init__(self, y, x):
        self.y = y
        self.x = x

    def pos(self):
        return self.y, self.x


class Nodegrid:
    def __init__(self, ysize, xsize):
        # Create grid of nodes
        self.nodegrid = [[Node(y, x) for x in range(xsize)] for y in range(ysize)]

        self.g = networkx.DiGraph()
        for nodelist in self.nodegrid:
            self.g.add_nodes_from(nodelist)

        # Source node
        self.source = Node(-1, -1)
        self.sink = Node(-1, -1)

        self.g.add_node(self.source)
        self.g.add_node(self.sink)

        self.ysize = ysize
        self.xsize = xsize

    def loop(self, edgecallback, nodecallback):
        """
        Loops over the grid of nodes. Two callback functions are required:

        :param edgecallback: Called for every edge.
        :param nodecallback: Called for every node.
        """
        logging.info("Iterate through graph.")

        for y in range(self.ysize - 1):
            for x in range(self.xsize - 1):
                node_i = self.nodegrid[y][x]

                # Node
                nodecallback(node_i)

                # Right edge
                node_j = self.nodegrid[y][x + 1]
                edgecallback(node_i, node_j)

                # Down edge
                node_j = self.nodegrid[y + 1][x]
                edgecallback(node_i, node_j)

        # Last column
        for y in range(self.ysize - 1):
            node_i = self.nodegrid[y][self.xsize - 1]

            # Node
            nodecallback(node_i)

            # Down edge
            node_j = self.nodegrid[y + 1][self.xsize - 1]
            edgecallback(node_i, node_j)

        # Last row
        for x in range(self.xsize - 1):
            node_i = self.nodegrid[self.ysize - 1][x]

            # Node
            nodecallback(node_i)

            # Right edge
            node_j = self.nodegrid[self.ysize - 1][x + 1]
            edgecallback(node_i, node_j)

        # Last node
        nodecallback(self.nodegrid[self.ysize - 1][self.xsize - 1])

    def loopedges(self, edgecallback):
        logging.info("Iterate through edges.")

        for y in range(self.ysize - 1):
            for x in range(self.xsize - 1):
                node_i = self.nodegrid[y][x]

                # Right edge
                node_j = self.nodegrid[y][x + 1]
                edgecallback(node_i, node_j)

                # Down edge
                node_j = self.nodegrid[y + 1][x]
                edgecallback(node_i, node_j)

        # Last column
        for y in range(self.ysize - 1):
            node_i = self.nodegrid[y][self.xsize - 1]

            # Down edge
            node_j = self.nodegrid[y + 1][self.xsize - 1]
            edgecallback(node_i, node_j)

        # Last row
        for x in range(self.xsize - 1):
            node_i = self.nodegrid[self.ysize - 1][x]

            # Right edge
            node_j = self.nodegrid[self.ysize - 1][x + 1]
            edgecallback(node_i, node_j)

    @staticmethod
    def loopedges_raw(callback, ysize, xsize):
        logging.info("Iterate through edges.")

        for y in range(ysize - 1):
            for x in range(xsize - 1):
                pos_i = (y, x)

                # Right edge
                pos_j = (y, x + 1)
                callback(pos_i, pos_j)

                # Down edge
                pos_j = (y + 1, x)
                callback(pos_i, pos_j)

        # Last column
        for y in range(ysize - 1):
            pos_i = (y, xsize - 1)

            # Down edge
            pos_j = (y + 1, xsize - 1)
            callback(pos_i, pos_j)

        # Last row
        for x in range(xsize - 1):
            pos_i = (ysize - 1, x)

            # Right edge
            pos_j = (ysize - 1, x + 1)
            callback(pos_i, pos_j)

    @staticmethod
    def loopnodes_raw(callback, ysize, xsize):
        logging.info("Iterate through nodes.")
        for y in range(ysize):
            for x in range(xsize):
                callback((y, x))

    def loopnodes(self, callback):
        logging.info("Iterate through nodes.")
        for y in range(self.ysize):
            for x in range(self.xsize):
                callback(self.nodegrid[y][x])

    def add_edge(self, node_i, node_j, capacity):
        self.g.add_edge(node_i, node_j, capacity=capacity)

    def add_source_edge(self, node, capacity):
        self.g.add_edge(self.source, node, capacity=capacity)

    def add_sink_edge(self, node, capacity):
        self.g.add_edge(node, self.sink, capacity=capacity)

    def maxflow(self):
        logging.info("Calculate max flow.")
        value, flows = networkx.maximum_flow(self.g, self.source, self.sink)
        return value, flows

    def mincut(self):
        logging.info("Calculate mincut.")
        value, cut = networkx.minimum_cut(self.g, self.source, self.sink)
        return value, cut

    def getcap(self, node):
        return self.g[self.source][node]["capacity"]

    def hassourcepath(self, node):
        return node in self.g[self.source]

    def draw(self):
        positions = {}
        for nodelist in self.nodegrid:
            for node in nodelist:
                positions[node] = [node.x, node.y]

        pad = 2
        nodesize = 10
        positions[self.source] = [self.xsize / 2 - 0.5, -pad]
        positions[self.sink] = [self.xsize / 2 - 0.5, self.ysize + pad]
        networkx.draw_networkx(self.g, pos=positions,
                               node_size=nodesize, with_labels=False,
                               width=0.5)


class Node_c:
    def __init__(self, nodeid, y, x):
        self.nodeid = nodeid
        self.y = y
        self.x = x


class Nodegrid_c:
    def __init__(self, ysize, xsize):
        self.g = maxflow.GraphFloat()

        self.nodeids = self.g.add_grid_nodes((ysize, xsize))

        self.ysize = ysize
        self.xsize = xsize

    def loop(self, edgecallback, nodecallback):
        """
        Loops over the grid of nodes. Two callback functions are required:

        :param edgecallback: Called for every edge.
        :param nodecallback: Called for every node.
        """
        logging.info("Iterate through graph.")

        for y in range(self.ysize - 1):
            for x in range(self.xsize - 1):
                node_i = self.getNode(y, x)

                # Node
                nodecallback(node_i)

                # Right edge
                node_j = self.getNode(y, x + 1)
                edgecallback(node_i, node_j)

                # Down edge
                node_j = self.getNode(y + 1, x)
                edgecallback(node_i, node_j)

        # Last column
        for y in range(self.ysize - 1):
            node_i = self.getNode(y, self.xsize - 1)

            # Node
            nodecallback(node_i)

            # Down edge
            node_j = self.getNode(y + 1, self.xsize - 1)
            edgecallback(node_i, node_j)

        # Last row
        for x in range(self.xsize - 1):
            node_i = self.getNode(self.ysize - 1, x)

            # Node
            nodecallback(node_i)

            # Right edge
            node_j = self.getNode(self.ysize - 1, x + 1)
            edgecallback(node_i, node_j)

        # Last node
        nodecallback(self.getNode(self.ysize - 1, self.xsize - 1))

    def add_sink_edge(self, node_i, cap):
        self.g.add_tedge(node_i.nodeid, 0, cap)

    def add_source_edge(self, node_i, cap):
        self.g.add_tedge(node_i.nodeid, cap, 0)

    def add_edge(self, node_i, node_j, cap):
        self.g.add_edge(node_i.nodeid, node_j.nodeid, cap, 0)

    def loopnodes(self, callback):
        logging.info("Iterate through nodes.")
        for y in range(self.ysize):
            for x in range(self.xsize):
                callback(self.getNode(y, x))

    def maxflow(self):
        logging.info("Calculate max flow.")
        return self.g.maxflow()

    def getNode(self, y, x):
        return Node_c(self.nodeids[y, x], y, x)

    def getsegment(self, node):
        return self.g.get_segment(node.nodeid)