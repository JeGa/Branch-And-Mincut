import logging
import queue
import time

import matplotlib.pyplot as plt
import maxflow
import numpy as np

import utility


class Rectangle:
    """
    Represents Omega_k.
    """

    def __init__(self, minf, maxf, minb, maxb):
        self.minf = minf
        self.maxf = maxf

        self.minb = minb
        self.maxb = maxb

        self.maxflowvalue = -1

    def setmaxflow(self, value):
        self.maxflowvalue = value

    def getmaxflow(self):
        return self.maxflowvalue

    def split(self):
        """
        Split the rectangle along the longest edge.
        """
        if self.single():
            raise RuntimeError("Not splitable.")

        diff_f = self.maxf - self.minf
        diff_b = self.maxb - self.minb

        if diff_f >= diff_b:
            # Split along f axis.
            half = int((self.maxf - self.minf + 1) / 2)
            rect1 = Rectangle(self.minf, self.minf + half - 1, self.minb, self.maxb)
            rect2 = Rectangle(self.minf + half, self.maxf, self.minb, self.maxb)
        else:
            # Split along b axis.
            half = int((self.maxb - self.minb + 1) / 2)
            rect1 = Rectangle(self.minf, self.maxf, self.minb, self.minb + half - 1)
            rect2 = Rectangle(self.minf, self.maxf, self.minb + half, self.maxb)

        return rect1, rect2

    def single(self):
        if self.minf == self.maxf and self.minb == self.maxb:
            return True
        return False

    def strsize(self):
        f = "f: (" + str(self.minf) + "," + str(self.maxf) + ")"
        b = "b: (" + str(self.minb) + "," + str(self.maxb) + ")"
        return f + "|" + b


class params:
    def __init__(self):
        self.mu = 1
        self.v = 0.1
        self.l1 = 0.0001
        self.l2 = 0.0001


class PQueue(queue.PriorityQueue):
    def __init__(self):
        queue.PriorityQueue.__init__(self)
        self.counter = 0

    def put(self, item, priority):
        queue.PriorityQueue.put(self, (priority, self.counter, item))
        self.counter += 1

    def get(self, *args, **kwargs):
        _, _, item = queue.PriorityQueue.get(self, *args, **kwargs)
        return item


class bmincut:
    def __init__(self, img):
        self.img = img

        self.ysize = img.shape[0]
        self.xsize = img.shape[1]

        self.params = params()

        self.queue = PQueue()

    def addparam(self, rect):
        self.queue.put(rect, rect.getmaxflow())

    def getparam(self):
        return self.queue.get()

    def showimg(self, grid):
        plt.imshow(self.segmentimg(grid))
        plt.draw()

    def segmentimg(self, grid):
        img = np.empty((self.ysize, self.xsize))

        def seg(node_i):
            nonlocal grid
            if grid.getsegment(node_i) == 1:
                # Foreground
                img[node_i.y, node_i.x] = 255
            else:
                # Background
                img[node_i.y, node_i.x] = 0

        grid.loopnodes(seg)

        return img

    def segment_faster(self):
        logging.info("Initialize first square.")

        start = time.perf_counter()

        # Initial square with all possible values.
        rect = Rectangle(0, 255, 0, 255)

        _, _, _ = self.mincut_faster(rect)
        self.addparam(rect)

        img = self.branchandmincut_faster()

        end = time.perf_counter()
        duration = end - start
        logging.info("Took " + str(duration) + " seconds.")

        return img

    def branchandmincut_faster(self):
        j = 0

        while True:
            logging.info("Iteration " + str(j))
            j += 1

            # Get parameter with smallest lower bound.
            rect = self.getparam()

            logging.info("->" + " " + str(rect.getmaxflow()) + " " + rect.strsize())

            if rect.single():
                logging.info("Terminating with:")
                return self.mincut_segment(rect)

            rect1, rect2 = rect.split()

            _, _, _ = self.mincut_faster(rect1)
            self.addparam(rect1)

            _, _, _ = self.mincut_faster(rect2)
            self.addparam(rect2)

        return self.mincut_segment(self.getparam())

    def segment(self):
        logging.info("Initialize first square.")

        start = time.perf_counter()

        # Initial square with all possible values.
        sq = Rectangle(0, 255, 0, 255)

        grid = self.potandflow(sq)
        self.addparam(sq)

        self.showimg(grid)

        grid = self.branchandmincut()

        img = self.segmentimg(grid)

        end = time.perf_counter()
        duration = end - start
        logging.info("Took " + str(duration) + " seconds.")

        return img

    def branchandmincut(self):
        j = 0
        while True:
            logging.info("Iteration " + str(j))
            j += 1

            # Get parameter with smallest lower bound.
            rect = self.getparam()

            logging.info("->" + " " + str(rect.getmaxflow()) + " " + rect.strsize())

            if rect.single():
                # Again ... not sooo nice
                logging.info("Terminating with:")
                return self.potandflow(rect)

            rect1, rect2 = rect.split()

            # logging.info("1) First of split.")
            grid = self.potandflow(rect1)
            # self.showimg(grid)
            self.addparam(rect1)

            # logging.info("2) Second of split.")
            grid = self.potandflow(rect2)
            # self.showimg(grid)
            self.addparam(rect2)

        return self.potandflow(self.getparam())

    def potandflow(self, rect):
        """
        Gets the aggregate potentials and computes the mincut.
        """
        # logging.info("Get aggregate potentials.")
        # potentials = self.aggreg_potentials(rect)
        logging.info("Calculate mincut.")
        maxflow, grid = self.mincut(rect)
        rect.setmaxflow(maxflow)

        return grid

    def aggreg_potentials(self, rect):
        """
        :return: y * x * 2 array of the aggregate potentials F, B
        (There is no pairwise term)
        """
        potentials = np.empty((self.ysize, self.xsize, 2))

        for y in range(self.ysize):
            for x in range(self.xsize):
                potentials[y, x, 1] = self.foreground(rect, y, x)
                potentials[y, x, 0] = self.background(rect, y, x)

        return potentials

    def mincut_segment(self, rect):
        flow, grid, nodeids = self.mincut_faster(rect)
        img = np.empty((self.ysize, self.xsize))

        for y in range(self.ysize):
            for x in range(self.xsize):
                # print(grid.get_segment(nodeids[y, x]))
                if grid.get_segment(nodeids[y, x]) == 1:
                    # Foreground
                    img[y, x] = 255
                else:
                    # Background
                    img[y, x] = 0

        return img

    def mincut_faster(self, rect):
        logging.info("Calculate mincut.")

        # TODO: This is just plain hacked in.
        grid = maxflow.GraphFloat()
        nodeids = grid.add_grid_nodes((self.ysize, self.xsize))

        for y in range(self.ysize - 1):
            for x in range(self.xsize - 1):
                node_i = nodeids[y, x]

                # Node
                # Foreground = 1
                cap = self.foreground(rect, y, x)
                grid.add_tedge(node_i, cap, 0)

                # Background = 0
                cap = self.background(rect, y, x)
                grid.add_tedge(node_i, 0, cap)

                # Right edge
                node_j = nodeids[y, x + 1]
                cap = 1
                grid.add_edge(node_i, node_j, cap, 0)

                # Down edge
                node_j = nodeids[y + 1, x]
                cap = 1
                grid.add_edge(node_i, node_j, cap, 0)

                # Right-down edge
                node_j = nodeids[y + 1, x + 1]
                cap = 1
                grid.add_edge(node_i, node_j, cap, 0)

                node_i = nodeids[y, x + 1]
                node_j = nodeids[y + 1, x]
                cap = 1
                grid.add_edge(node_i, node_j, cap, 0)

        # Last column
        for y in range(self.ysize - 1):
            node_i = nodeids[y, self.xsize - 1]

            # Node

            # Foreground = 1
            cap = self.foreground(rect, y, self.xsize - 1)
            grid.add_tedge(node_i, cap, 0)

            # Background = 0
            cap = self.background(rect, y, self.xsize - 1)
            grid.add_tedge(node_i, 0, cap)

            # Down edge
            node_j = nodeids[y + 1, self.xsize - 1]
            cap = 1
            grid.add_edge(node_i, node_j, cap, 0)

        # Last row
        for x in range(self.xsize - 1):
            node_i = nodeids[self.ysize - 1, x]

            # Node

            # Foreground = 1
            cap = self.foreground(rect, self.ysize - 1, x)
            grid.add_tedge(node_i, cap, 0)

            # Background = 0
            cap = self.background(rect, self.ysize - 1, x)
            grid.add_tedge(node_i, 0, cap)

            # Right edge
            node_j = nodeids[self.ysize - 1, x + 1]
            cap = 1
            grid.add_edge(node_i, node_j, cap, 0)

        # Last node
        node_i = nodeids[self.ysize - 1, self.xsize - 1]

        # Node

        # Foreground = 1
        cap = self.foreground(rect, self.ysize - 1, self.xsize - 1)
        grid.add_tedge(node_i, cap, 0)

        # Background = 0
        cap = self.background(rect, self.ysize - 1, self.xsize - 1)
        grid.add_tedge(node_i, 0, cap)

        flow = grid.maxflow()

        rect.setmaxflow(flow)

        print(flow)

        return flow, grid, nodeids

    def mincut(self, rect):
        """
        Calculate the maxflow using the given aggregate potentials.
        """
        grid = utility.Nodegrid_c(self.ysize, self.xsize)

        def edge(node_i, node_j):
            nonlocal grid

            cap = 1
            grid.add_edge(node_i, node_j, cap)

        def node(node_i):
            nonlocal grid, rect

            # Foreground = 1
            cap = self.foreground(rect, node_i.y, node_i.x)  # potentials[node_i.y, node_i.x, 1]
            grid.add_source_edge(node_i, cap)

            # Background = 0
            cap = self.background(rect, node_i.y, node_i.x)  # potentials[node_i.y, node_i.x, 0]
            grid.add_sink_edge(node_i, cap)

        grid.loop(edge, node)

        return grid.maxflow(), grid

    def foreground(self, rect, y, x):
        I = self.img[y, x]

        min = self.mindiff(rect.minf, rect.maxf, I)
        if min == 0:
            return self.params.v
        return self.params.v + self.params.l1 * np.power(min, 2)

    def background(self, rect, y, x):
        I = self.img[y, x]

        min = self.mindiff(rect.minb, rect.maxb, I)
        if min == 0:
            return 0
        return self.params.l2 * np.power(min, 2)

    def mindiff(self, minbound, maxbound, I):
        if minbound > I:
            min = I - minbound
        elif maxbound < I:
            min = I - maxbound
        else:
            # cfmin <= I and cfmax => I
            min = 0

        return min


def main():
    logging.basicConfig(level=logging.INFO)

    imagename = "garden.png"

    logging.info("Read image.")
    img = utility.readimg_grayscale(imagename)

    seg = bmincut(img)
    img = seg.segment_faster()

    logging.info("Save image.")
    plt.imsave("img_out", img, cmap='gray')

    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
