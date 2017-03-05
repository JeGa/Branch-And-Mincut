# Branch-And-Mincut

Binary image segmentation using branch and mincut. The goal is to achieve globally optimal segmentation of the Chan-Vese energy function using graph cuts and branch-and-bound.

**Based on the paper:** 
Victor Lempitsky, Andrew Blake, and Carsten Rother. Branch-and-mincut: Global optimization for image segmentation with high-level priors. Journal of Mathematical Imaging and Vision, 44(3):315–329, March 2012

I tried two different libraries for maxflow/mincut calculation. The pure python implementation of networkx [https://networkx.github.io/]() and the faster c module based on Kolmogorovs libary. The main implementation is contained in the bmincut.py file, utility.py contains some helper structures for creating and managing the random field.
