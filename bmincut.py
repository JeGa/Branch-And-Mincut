import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import queue

import utility


class square:
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
        diff_f = abs(self.minf - self.maxf)
        diff_b = abs(self.minb - self.maxb)

        if diff_f > diff_b:
            half = (self.maxf - self.minf + 1) / 2
            sq1 = square(self.minf, half - 1, self.minb, self.maxb)
            sq2 = square(half, self.maxf, self.minb, self.maxb)
        else:
            half = (self.maxb - self.minb + 1) / 2
            sq1 = square(self.minf, self.maxf, self.minb, half - 1)
            sq2 = square(self.minf, self.maxf, half, self.maxb)

        return sq1, sq2

    def single(self):
        if self.minf == self.maxf and self.minb == self.maxb:
            return True
        return False


class params:
    def __init__(self):
        self.mu = 1
        self.v = 0.1
        self.l1 = 0.0001
        self.l2 = 0.0002


class bmincut:
    def __init__(self, img):
        self.img = img

        self.ysize = img.shape[0]
        self.xsize = img.shape[1]

        self.params = params()

        # TODO: Priority queue
        self.queue = queue.PriorityQueue()

    def addparam(self, sq):
        self.queue.put(sq.getmaxflow, sq)

    def getparam(self):
        return self.queue.get()

    def segment(self):
        # Initial square with all possible values.
        sq = square(0, 255, 0, 255)
        _ = self.potandflow(sq)

        self.addparam(sq)

        segments = self.branchandmincut()

    def branchandmincut(self):
        while True:
            # Get parameter with smallest lower bound.
            sq = self.getparam()

            if sq.single():
                return self.potandflow(sq)

            sq1, sq2 = sq.split()
            _ = self.potandflow(sq1)
            self.addparam(sq1)

            _ = self.potandflow(sq2)
            self.addparam(sq2)

    def potandflow(self, sq):
        potentials = self.aggreg_potentials(sq)
        maxflow, segments = self.maxflow(potentials)
        sq.setmaxflow(maxflow)
        return segments

    def maxflow(self, potentials):
        """
        Calculate the maxflow using the given aggregate potentials.
        """

        grid = utility.Nodegrid(self.ysize, self.xsize)

        def edge(self, node_i, node_j):
            nonlocal grid, potentials

            cap = 1
            grid.add_edge(node_i, node_j, cap)

        def node(self, node_i):
            nonlocal grid, potentials

            # Foreground = 1
            cap = potentials[node_i.y, node_i.x, 1]
            grid.add_sink_edge(node_i, cap)

            # Background = 0
            cap = potentials[node_i.y, node_i.x, 0]
            grid.add_source_edge(node_i, cap)

        grid.loop(edge, node)
        return grid.maxflow()

    def aggreg_potentials(self, sq):
        """
        :return: y * x * 2 array of the aggregate potentials F, B
        (There is no pairwise term)
        """
        potentials = np.empty((self.ysize, self.xsize, 2))

        for y in range(self.ysize):
            for x in range(self.xsize):
                potentials[y, x, 1] = self.foreground(sq, y, x)
                potentials[y, x, 0] = self.background(sq, y, x)

        return potentials

    def foreground(self, sq, y, x):
        cfmin = sq.minf
        cfmax = sq.maxf

        I = self.img[y, x]

        min = self.mindiff(cfmin, cfmax, I)
        return self.params.v + self.params.l1 * np.power(min, 2)

    def background(self, sq, y, x):
        cbmin = sq.minb
        cbmax = sq.maxb

        I = self.img[y, x]

        min = self.mindiff(cbmin, cbmax, I)
        return self.params.l2 * np.power(min, 2)

    def mindiff(self, minbound, maxbound, I):
        if minbound >= I:
            min = I - minbound
        elif maxbound <= I:
            min = I - maxbound
        else:
            # cfmin < I and cfmax > I
            min = I

        return min


def main():
    logging.basicConfig(level=logging.INFO)

    imagename = "garden.png"

    logging.info("Read image.")
    img = utility.readimg_grayscale(imagename)

    seg = bmincut(img)
    seg.segment()

    logging.info("Save image.")
    plt.imsave("img_out", img)

    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
