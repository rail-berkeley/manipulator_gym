import numpy as np
from typing import List, Optional


class WorkspaceChecker:
    """
    Utility class to check and visualize workspace boundaries.
    This also allows clipping points to the nearest boundary.
    """
    def __init__(self, cuboids: List[np.array]):
        """The cuboids define the valid workspace boundaries."""
        assert len(cuboids) > 0, "At least one cuboid is required."
        assert all([cuboid.shape == (2, 3) for cuboid in cuboids]), "Cuboids must be of shape (2, 3)."
        self.cuboids = cuboids

    def within_workspace(self, point: np.array) -> bool:
        """ Check if the point is within the workspace"""
        for cuboid in self.cuboids:
            lower_bound = np.minimum(cuboid[0], cuboid[1])
            upper_bound = np.maximum(cuboid[0], cuboid[1])
            if np.all(point >= lower_bound) and np.all(point <= upper_bound):
                return True
        return False

    def clip_point(self, point: np.array) -> np.array:
        """
        Clip the point within the nearest cuboid boundaries.
        If outside all, return closest boundary point.
        """
        closest_point = None
        min_distance = np.inf

        for cuboid in self.cuboids:
            lower_bound = np.minimum(cuboid[0], cuboid[1])
            upper_bound = np.maximum(cuboid[0], cuboid[1])
            clipped = np.maximum(lower_bound, np.minimum(upper_bound, point))
            distance = np.linalg.norm(clipped - point)

            if distance < min_distance:
                min_distance = distance
                closest_point = clipped

        return closest_point if closest_point is not None else point

    def visualize(self, point: Optional[np.array] = None):
        """ Visualize the cuboids and optionally a point and its clipped version. """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for cuboid in self.cuboids:
            self._plot_cuboid(ax, cuboid)

        if point is not None:
            ax.scatter(*point, color='red', label='Point', s=100)

        if not self.within_workspace(point):
            clipped_point = self.clip_point(point)
            ax.scatter(*clipped_point, color='green', label='Clipped Point', s=100)

        ax.legend()
        plt.show()

    def _plot_cuboid(self, ax, cuboid):
        """ Helper function to plot a cuboid. """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        lower_bound = np.minimum(cuboid[0], cuboid[1])
        upper_bound = np.maximum(cuboid[0], cuboid[1])

        # Generate the list of vertices
        r = [lower_bound, upper_bound]
        vertices = np.array([[r[x][0], r[y][1], r[z][2]] for x in range(2) for y in range(2) for z in range(2)])

        # Define the vertices that form the 12 faces of the cuboid
        faces = [[vertices[j] for j in [0, 1, 3, 2]], [vertices[j] for j in [4, 5, 7, 6]],
                 [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]],
                 [vertices[j] for j in [0, 2, 6, 4]], [vertices[j] for j in [1, 3, 7, 5]]]

        # Create a 3D polygon and add it to the plot
        ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))


if __name__ == "__main__":
    # example usage
    cuboids = [
        np.array([[0, 0, 0], [1, 1, 1]]),
        np.array([[0, 2, 0], [2.2, 3.2, 4.1]])
    ]
    cc = WorkspaceChecker(cuboids=cuboids)
    point = np.array([4, 4, 4])
    # point = np.array([0.5, 0.5, 0.5])
    clipped_point = cc.clip_point(point)

    cc.visualize(point)
