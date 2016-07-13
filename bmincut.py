import numpy as np
import logging
import time
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
        if self.single():
            raise "Not splitable."

        diff_f = self.maxf - self.minf
        diff_b = self.maxb - self.minb

        if diff_f >= diff_b:
            half = int((self.maxf - self.minf + 1) / 2)
            sq1 = square(self.minf, self.minf + half - 1, self.minb, self.maxb)
            sq2 = square(self.minf + half, self.maxf, self.minb, self.maxb)
        else:
            half = int((self.maxb - self.minb + 1) / 2)
            sq1 = square(self.minf, self.maxf, self.minb, self.minb + half - 1)
            sq2 = square(self.minf, self.maxf, self.minb + half, self.maxb)

        return sq1, sq2

    def single(self):
        if self.minf == self.maxf and self.minb == self.maxb:
            return True
        return False

    def strsize(self):
        f = "(" + str(self.minf) + "," + str(self.maxf) + ")"
        b = "(" + str(self.minb) + "," + str(self.maxb) + ")"
        return f + ";" + b


class params:
    def __init__(self):
        self.mu = 2
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

        # self.queue = queue.PriorityQueue()
        self.queue = PQueue()

    def addparam(self, sq):
        # self.queue.put((sq.getmaxflow(), time.time(), sq))
        self.queue.put(sq, sq.getmaxflow())

    def getparam(self):
        return self.queue.get()  # [2]

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

    def segment(self):
        logging.info("Initialize first square.")

        start = time.perf_counter()

        # Initial square with all possible values.
        sq = square(0, 255, 0, 255)

        grid = self.potandflow(sq)

        self.addparam(sq)

        grid = self.branchandmincut()

        img = self.segmentimg(grid)

        # reachable, non_reachable = cut
        # for i in reachable:
        #    self.img[i.y, i.x] = 0
        # for i in non_reachable:
        #    self.img[i.y, i.x] = 255

        end = time.perf_counter()
        duration = end - start
        logging.info("Took " + str(duration) + "seconds.")

        return img

    def branchandmincut(self):
        j = 0
        while True:
            # for j in range(10):
            logging.info("Iteration " + str(j))
            j += 1

            # Get parameter with smallest lower bound.
            sq = self.getparam()

            logging.info("->" + " " + str(sq.getmaxflow()) + " " + sq.strsize())

            if sq.single():
                # Again ... not sooo nice
                logging.info("Terminating with:")
                return self.potandflow(sq)

            sq1, sq2 = sq.split()

            logging.info("First of split. ----")
            _ = self.potandflow(sq1)
            self.addparam(sq1)

            logging.info("Second of split. ---")
            _ = self.potandflow(sq2)
            self.addparam(sq2)

        return self.potandflow(self.getparam())

    def potandflow(self, sq):
        potentials = self.aggreg_potentials(sq)
        maxflow, grid = self.mincut(potentials)
        sq.setmaxflow(maxflow)

        return grid

    def mincut(self, potentials):
        """
        Calculate the maxflow using the given aggregate potentials.
        """
        # TODO
        # grid = utility.Nodegrid(self.ysize, self.xsize)
        grid = utility.Nodegrid_c(self.ysize, self.xsize)

        def edge(node_i, node_j):
            nonlocal grid, potentials

            cap = 1
            grid.add_edge(node_i, node_j, cap)

        def node(node_i):
            nonlocal grid, potentials

            # Foreground = 1
            cap = potentials[node_i.y, node_i.x, 1]
            grid.add_source_edge(node_i, cap)

            # Background = 0
            cap = potentials[node_i.y, node_i.x, 0]
            grid.add_sink_edge(node_i, cap)

        grid.loop(edge, node)

        # TODO
        # return grid.mincut()
        return grid.maxflow(), grid

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
        I = self.img[y, x]

        min = self.mindiff(sq.minf, sq.maxf, I)
        if min == 0:
            return self.params.v
        return self.params.v + self.params.l1 * np.power(min, 2)

    def background(self, sq, y, x):
        I = self.img[y, x]

        min = self.mindiff(sq.minb, sq.maxb, I)
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
    img = seg.segment()

    logging.info("Save image.")
    plt.imsave("img_out", img)

    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
