"""
.. module:: Trajectory

Trajectory
*************

:Description: Trajectory

    

:Authors: bejar
    

:Version: 

:Created on: 15/10/2015 10:30 

"""

__author__ = 'bejar'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Trajectory:
    """
    Stores the information from a trajectory

    """

    coords = None  # 2-dimensional array for the points of a trajectory
    distances = None  # accumulated distance for each segment of the trajectory
    cv_hull = None  # 2-dimensional array for the convex hull of the trajectory
    cdmx = None  # Center of mass, coord x
    cdmy = None  # Center of mass, coord y
    exer_id = None  # ID of the exercice

    def __init__(self, vec, exer=''):

        self.coords = vec
        self.exer_id = exer
        # Accumulated distances
        self.distances = np.zeros(self.coords.shape[0])
        for i in range(1, self.coords.shape[0]):
            dx = self.coords[i - 1, 0] - self.coords[i, 0]
            dy = self.coords[i - 1, 1] - self.coords[i, 1]
            self.distances[i] = self.distances[i - 1] + np.sqrt((dx * dx) + (dy * dy))

        self.cv_hull = self.convex_hull()
        self.cdmx = np.sum(self.coords[:, 0]) / (self.coords.shape[0] * 1.0)
        self.cdmy = np.sum(self.coords[:, 1]) / (self.coords.shape[0] * 1.0)

    def cdm(self):
        """
        center of mass of the trajectory
        :return:
        """
        return self.cdmx, self.cdmy

    def geodesic(self):
        """
        Sum of distances of a path (the geodesic distance)

        :param vec: Vector of the get_coordinates of the trajectory
        :return:
        """
        return self.distances[-1]
        # dif = 0.0
        # for i in range(self.coords.shape[0]-1):
        #     dif += np.sqrt(((self.coords[i,0] - self.coords[i+1,0])**2) + ((self.coords[i,1] - self.coords[i+1,1])**2))
        #
        # return dif

    def parametric_select(self, min, max):
        """
        Uses the trajectory in vec parametrically normalizing its distanced to the interval [0-1] and selects from the
        trajectory a segment betwee min and max

        :param vec: array with the trajectory data
        :param dist: array with the incremental sum of distances of the trajectory
        :param min: minimum element of the trajectory
        :param max: maximum element of the trajectory
        :return:
        """
        vec = self.coords.copy()
        dist = self.distances.copy()
        dist /= np.max(dist)

        rows = [j for j in range(vec.shape[0]) if min <= dist[j] <= max]
        sel = Trajectory(vec[rows, :])

        return (sel)

    def euclidean_end_to_end(self):
        """
        Euclidean distance between the beginning and the end of the trajectory
        :return:
        """
        return np.sqrt(
            ((self.coords[-1, 0] - self.coords[0, 0]) ** 2) + ((self.coords[-1, 1] - self.coords[0, 1]) ** 2))

    def straightness(self):
        """
        Straightness coefficient

        returns:
          - Ratio between geodesic and euclidean distance among initial and final point
          - Ratio between geodesic and euclidean distance among initial and final point to the CDM of the curve

        :param vec:
        :return:
        """
        vec = self.coords
        mx = self.cdmx  # np.sum(vec[:,0])/(vec.shape[0]*1.0)
        my = self.cdmy  # np.sum(vec[:,1])/(vec.shape[0]*1.0)

        otocdm = np.sqrt(((vec[0, 0] - mx) ** 2) + ((vec[0, 1] - my) ** 2))
        ftocdm = np.sqrt(((vec[-1, 0] - mx) ** 2) + ((vec[-1, 1] - my) ** 2))

        euc = otocdm + ftocdm

        euc2 = self.euclidean_end_to_end()
        geo = self.geodesic()
        return euc2 / geo, euc / geo

    def convex_hull(self):
        """Computes the convex hull of a set of 2D points.

        Taken from wikipedia

        Input: an iterable sequence of (x, y) pairs representing the points.
        Output: a list of vertices of the convex hull in counter-clockwise order,
          starting from the vertex with the lexicographically smallest get_coordinates.
        Implements Andrew's monotone chain algorithm. O(n log n) complexity.
        """

        # Sort the points lexicographically (tuples are compared lexicographically).
        # Remove duplicates to detect the case we have just one unique point.
        points = [(x, y) for x, y in zip(self.coords[:, 0], self.coords[:, 1])]
        coords = sorted(set(points))

        # Boring case: no points or a single point, possibly repeated multiple times.
        if len(coords) <= 1:
            return coords

        # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
        # Returns a positive value, if OAB makes a counter-clockwise turn,
        # negative for clockwise turn, and zero if the points are collinear.
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        for p in coords:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(coords):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenation of the lower and upper hulls gives the convex hull.
        # Last point of each list is omitted because it is repeated at the beginning of the other list.
        return np.array(lower[:-1] + upper[:-1])

    def convex_hull_area(self):
        """
        Area of a 2D convex Polygon

        :param points:
        :return:
        """
        x = self.cv_hull[:, 0]
        y = self.cv_hull[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def convex_hull_aspect_ratio(self):
        """
        Computes the aspect ration of the convex

        Attention: Min axis simplified to the distance to the closest vertex of the convex hull instead of the shortest
          distance to the convex hull polygon
        :param vec:
        :return:
        """
        # Centroid of the convex hull
        chcdmx = np.sum(self.cv_hull[:, 0]) / (self.cv_hull.shape[0] * 1.0)
        chcdmy = np.sum(self.cv_hull[:, 1]) / (self.cv_hull.shape[0] * 1.0)

        chdist = self.cv_hull - np.array([chcdmx, chcdmy])
        chdist = chdist * chdist
        chdist = np.sum(chdist, axis=1)
        chdist = np.sqrt(chdist)
        axmin = np.min(chdist)
        axmax = np.max(chdist)

        return axmin / axmax

    def convex_hull_perimeter(self):
        """
        Perimeter of the convex hull

        :return:
        """
        cdiff = self.cv_hull - np.roll(self.cv_hull, -1, axis=0)
        cdiff = cdiff * cdiff
        return np.sum(np.sqrt(np.sum(cdiff, axis=1)))

    def max_val(self):
        """
        Returns the maximum value of the get_coordinates
        :return:
        """
        return (np.max(self.coords))

    def min_val(self):
        """
        Returns the minimum value of the get_coordinates
        :return:
        """
        return (np.min(self.coords))

    def plot_trajectory(self, show=False):
        """
        Plots the trajectory and different characteristics

        :return:
        """
        mx, my = self.cdm()

        otocdm = np.sqrt(((self.coords[0, 0] - mx) ** 2) + ((self.coords[0, 1] - my) ** 2))
        ftocdm = np.sqrt(((self.coords[-1, 0] - mx) ** 2) + ((self.coords[-1, 1] - my) ** 2))
        euc = otocdm + ftocdm
        euc2 = np.sqrt(
            ((self.coords[-1, 0] - self.coords[0, 0]) ** 2) + ((self.coords[-1, 1] - self.coords[0, 1]) ** 2))
        geo = self.geodesic()

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(self.exer_id)
        ax = fig.add_subplot(111)
        plt.plot(self.coords[:, 0], self.coords[:, 1])
        plt.scatter(self.coords[:, 0], self.coords[:, 1])
        plt.scatter(mx, my, color='r', marker='+')
        plt.plot([self.coords[0, 0], mx], [self.coords[0, 1], my], color='r')
        plt.plot([self.coords[-1, 0], mx], [self.coords[-1, 1], my], color='r')
        ax.set_aspect('equal', 'datalim')

        ch = self.cv_hull
        plt.plot(ch[:, 0], ch[:, 1], color='g')
        if show:
            plt.show()
            plt.close()
        return fig

    def plot_over_trajectory(self, data):
        """
        Plots data over the trajectory on a 3d Graph

        data is a list of arrays of the same length as the trajectory
        :param data:
        :return:
        """
        colors = 'rgbymc'

        fig = plt.figure(figsize=(60, 20))
        ax = fig.add_subplot(111, projection='3d')

        for i, d in enumerate(data):
            plt.plot(self.coords[:, 0], self.coords[:, 1], d, c=colors[i])
        plt.title(self.exer_id)
        plt.show()
        plt.close()

    def find_begginning_end(self):
        """
        Looks for the beggining of the exercise assuming that it starts when the distance between consecutive steps
        is larger than a threshold
        Heuristically we assume that the mean of incremental steps in the central third of the signal is closer to the
        stable step regime
        :return:
        """

        chunk = int(self.coords.shape[0] / 4.0)
        vdis = np.zeros(self.coords.shape[0])
        for i in range(1, self.coords.shape[0]):
            dx = self.coords[i - 1, 0] - self.coords[i, 0]
            dy = self.coords[i - 1, 1] - self.coords[i, 1]
            vdis[i] = np.sqrt((dx * dx) + (dy * dy))

        # Middle third of the distances
        thresh = np.mean(vdis[chunk:3 * chunk])

        bg = 0
        for i in range(vdis.shape[0]):
            if vdis[i] > thresh:
                bg = i - 1
                break

        nd = 0
        for i in range(vdis.shape[0]):
            if vdis[vdis.shape[0] - i - 1] > thresh:
                nd = vdis.shape[0] - i - 1
                break

        return bg, nd, self.distances


if __name__ == '__main__':
    a = np.array([[0, 0], [2, 0], [2, 2], [1, 3], [0, 2]])
    tr = Trajectory(a)

    print(tr.straightness())
    print(tr.convex_hull_perimeter())
    print(tr.convex_hull_area())
