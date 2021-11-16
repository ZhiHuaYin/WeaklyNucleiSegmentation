import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from skimage import draw
from skimage import morphology, io
from tqdm import tqdm


# borrowed from https://gist.github.com/pv/8036995
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool_)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def create_Voronoi_label(img_path):
    label_point = io.imread(img_path)
    h, w = label_point.shape

    points = np.argwhere(label_point > 0)

    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)
    box = Polygon([[0, 0], [0, w], [h, w], [h, 0]])
    region_masks = np.zeros((h, w), dtype=np.int16)
    edges = np.zeros((h, w), dtype=np.bool_)
    count = 1
    for region in regions:
        polygon = vertices[region]
        # Clipping polygon
        poly = Polygon(polygon)
        poly = poly.intersection(box)
        polygon = np.array([list(p) for p in poly.exterior.coords])

        mask = poly2mask(polygon[:, 0], polygon[:, 1], (h, w))
        edge = mask * (~morphology.erosion(mask, morphology.disk(1)))
        edges += edge
        region_masks[mask] = count
        count += 1

    # fuse Voronoi edge and dilated points
    label_point_dilated = morphology.dilation(label_point, morphology.disk(2))
    label_vor = np.zeros((h, w, 3), dtype=np.uint8)
    label_vor[:, :, 0] = morphology.closing(edges > 0, morphology.disk(1)).astype(np.uint8) * 255
    label_vor[:, :, 1] = (label_point_dilated > 0).astype(np.uint8) * 255

    return label_vor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    # parser.add_argument('--height', type=int, default=0)
    # parser.add_argument('--width', type=int, default=0)
    # parser.add_argument('-height', type=int, default=992)
    # parser.add_argument('-width', type=int, default=1440)
    parser.add_argument('-j', type=int, default=0)

    args = parser.parse_args()
    return args


def work(file, root, args):
    if file is not None:
        if not os.path.exists(os.path.join(args.output, file.replace('.png', '_label_vor.png'))):
            # CODE HERE
            try:
                vor_label = create_Voronoi_label(os.path.join(root, file))
                io.imsave(
                    os.path.join(args.output, file.replace('.png', '_label_vor.png')),
                    vor_label, check_contrast=False)
            except Exception as e:
                print(file, e)
                return

            # cluster = create_cluster_label('point.png', './img', '.', '.')
            # io.imsave('point_label_cluster.png', cluster)


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    file_list = []
    for root, dirs, files in os.walk(args.input):
        file_list.extend(files)

    if args.j == 0:
        for file in file_list:
            work(file, root=args.input, args=args)
        return

    if len(file_list) % args.j != 0:
        file_list.extend([None] * (args.j - len(file_list) % args.j))

    with Pool(args.j) as p:
        list(tqdm(
            p.imap_unordered(partial(work, root=args.input, args=args), file_list),
            total=len(file_list), ascii=True
        ))
        p.close()


if __name__ == '__main__':
    main()
