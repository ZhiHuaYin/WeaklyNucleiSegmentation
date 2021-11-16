import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import morphology as ndi_morph
from scipy.ndimage.morphology import distance_transform_edt as dist_tranform
from skimage import morphology, measure, io
from sklearn.cluster import KMeans
from tqdm import tqdm
import cv2

# adjust_list = [
#     'EDF000.png',
#     'EDF005.png',
#     'ISBI_Train_022.png',
#     'ISBI_Train_026.png',
#     'ISBI_Train_027.png',
#     'ISBI_Train_028.png',
#     'ISBI_Train_037.png',
#     'ISBI_Train_043.png'
# ]
adjust_list = [
    'EDF005.png',
    'EDF007.png',
    'ISBI_Train_026.png',
    'ISBI_Train_027.png',
    'ISBI_Train_031.png',
    'ISBI_Train_040.png',
]


def create_cluster_label(img_name, data_dir, label_point_dir, label_vor_dir, process_pad=False):
    if not os.path.exists('{}/{}_label_vor.png'.format(label_vor_dir, img_name[:-4])):
        print('pass {}'.format(img_name))
        return

    # ori_image = io.imread(os.path.join(data_dir, img_name))
    ori_image = cv2.imread(os.path.join(data_dir, img_name), flags=1)

    if process_pad and img_name.startswith('ISBI'):
        ori_image = ori_image[256:768, 256:768, :]

    h, w, _ = ori_image.shape
    label_point = io.imread(os.path.join(label_point_dir, img_name))
    if process_pad and img_name.startswith('ISBI'):
        label_point = label_point[256:768, 256:768]

    # k-means clustering
    dist_embeddings = dist_tranform(255 - label_point).reshape(-1, 1)
    clip_dist_embeddings = np.clip(dist_embeddings, a_min=0, a_max=20)
    color_embeddings = np.array(ori_image, dtype=float).reshape(-1, 3) / 10
    embeddings = np.concatenate((color_embeddings, clip_dist_embeddings), axis=1)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)
    clusters = np.reshape(kmeans.labels_, (h, w))

    overlap_nums = [np.sum((clusters == i) * label_point) for i in range(3)]
    nuclei_idx = np.argmax(overlap_nums)
    remain_indices = np.delete(np.arange(3), nuclei_idx)
    dilated_label_point = morphology.binary_dilation(label_point, morphology.disk(5))
    overlap_nums = [np.sum((clusters == i) * dilated_label_point) for i in remain_indices]
    background_idx = remain_indices[np.argmin(overlap_nums)]

    nuclei_cluster = clusters == nuclei_idx
    background_cluster = clusters == background_idx

    # refine clustering results
    nuclei_labeled = measure.label(nuclei_cluster)
    initial_nuclei = morphology.remove_small_objects(nuclei_labeled, 30)
    refined_nuclei = np.zeros(initial_nuclei.shape, dtype=np.bool_)

    label_vor = io.imread('{}/{}_label_vor.png'.format(label_vor_dir, img_name[:-4]))
    if process_pad and img_name.startswith('ISBI'):
        label_vor = label_vor[256:768, 256:768, :]
    voronoi_cells = measure.label(label_vor[:, :, 0] == 0)
    voronoi_cells = morphology.dilation(voronoi_cells, morphology.disk(2))

    # refine clustering results
    unique_vals = np.unique(voronoi_cells)
    cell_indices = unique_vals[unique_vals != 0]
    N = len(cell_indices)
    for i in range(N):
        cell_i = voronoi_cells == cell_indices[i]
        nucleus_i = cell_i * initial_nuclei

        nucleus_i_dilated = morphology.binary_dilation(nucleus_i, morphology.disk(5))
        nucleus_i_dilated_filled = ndi_morph.binary_fill_holes(nucleus_i_dilated)
        nucleus_i_final = morphology.binary_erosion(nucleus_i_dilated_filled, morphology.disk(7))
        refined_nuclei += nucleus_i_final > 0

    if img_name in adjust_list:
        background_cluster = ~background_cluster

    refined_label = np.zeros((h, w, 3), dtype=np.uint8)
    label_point_dilated = morphology.dilation(label_point, morphology.disk(10))
    refined_label[:, :, 0] = (background_cluster * (refined_nuclei == 0) * (label_point_dilated == 0)).astype(
        np.uint8) * 255
    refined_label[:, :, 1] = refined_nuclei.astype(np.uint8) * 255

    if process_pad and img_name.startswith('ISBI'):
        refined_label = np.stack(
            [np.pad(refined_label[:, :, 0], ((256, 256), (256, 256)), constant_values=255),
             np.pad(refined_label[:, :, 1], ((256, 256), (256, 256)), constant_values=0),
             np.pad(refined_label[:, :, 2], ((256, 256), (256, 256)), constant_values=0)],
            axis=-1)

    return refined_label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-j', type=int, default=0)
    parser.add_argument('--isbi', action='store_true')

    args = parser.parse_args()
    return args


def work(file, root, args):
    if file is not None:
        if not os.path.exists(os.path.join(args.output, file.replace('.png', '_label_cluster.png'))):
            # CODE HERE
            try:
                img_path = root.replace('image_part/1', 'image')
                img_path = img_path.replace('image_part/2', 'image')
                img_path = img_path.replace('image_part/3', 'image')
                img_path = img_path.replace('image_part/4', 'image')
                img_path = img_path.replace('image_part/5', 'image')
                img_path = img_path.replace('image_part/6', 'image')

                label_point_dir = img_path.replace('/image', '/point')
                label_vor_dir = img_path.replace('/image', '/label_vor')

                cluster_label = create_cluster_label(file, root, label_point_dir, label_vor_dir, process_pad=args.isbi)
                if cluster_label is not None:
                    io.imsave(
                        os.path.join(args.output, file.replace('.png', '_label_cluster.png')),
                        cluster_label, check_contrast=False)
            except Exception as e:
                print(file, e)
                return


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
