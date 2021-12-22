import json
import time
import os
import os.path as osp
import tempfile
from collections import defaultdict

import cv2
import mmcv
import numpy as np
import tqdm
from pycocotools import mask, coco

from .builder import DATASETS
from .coco import CocoDataset

from multiprocessing import Pool
from functools import partial


@DATASETS.register_module()
class CocoDataset_Pseudo(CocoDataset):

    @staticmethod
    def bbox_to_segm(box):
        x, y, h, w = box
        ann = []
        ann.extend([x, y])
        ann.extend([x + h // 2, y])
        ann.extend([x + h, y])
        ann.extend([x + h, y + w // 2])
        ann.extend([x + h, y + w])
        ann.extend([x + h // 2, y + w])
        ann.extend([x, y + w])
        ann.extend([x, y + w // 2])
        return [ann]

    @staticmethod
    def mask2polygon(mask):
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            if len(contour) > 4:
                contour_list = contour.flatten().tolist()
                segmentation.append(contour_list)
        return segmentation

    @staticmethod
    def filter_labels(labels, origin_labels, thresh, iou_thresh):
        results = defaultdict(list)
        for det in labels:
            if det['score'] > thresh:
                results[det['image_id']].append(det)
        img_ids = origin_labels.getImgIds()
        labels = []
        for img_id in tqdm.tqdm(img_ids, ascii=True):
            preds = results[img_id]
            anns = origin_labels.imgToAnns[img_id]
            ann_boxes = [ann['bbox'] for ann in anns]
            for pred in preds:
                ious = mask.iou([pred['bbox']], ann_boxes, [0])
                if len(ious) == 0 or np.max(ious) <= iou_thresh:
                    pred['pred_score'] = pred['score']
                    pred.pop('score')
                    labels.append(pred)
        return labels

    @staticmethod
    def process_mask(x, label_mode='polygon'):
        x['area'] = float(mask.area(x['segmentation']))
        if label_mode == 'rle':
            x['iscrowd'] = 1
        elif label_mode == 'polygon':
            x['segmentation'] = CocoDataset_Pseudo.mask2polygon(mask.decode(x['segmentation']))
            if len(x['segmentation']) == 0:
                x['segmentation'] = CocoDataset_Pseudo.bbox_to_segm(x['bbox'])
            x['iscrowd'] = 0
        else:
            raise NotImplementedError
        return x

    def parse_results(self, new_labels, label_mode, result_files, thresh, iou_thresh, label_format='segm', num_j=0):
        label = coco.COCO(self.ann_file)

        print('\n{} Compute iou and filter labels: thresh {}, iou {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), thresh, iou_thresh))
        new_labels = self.filter_labels(new_labels, label, thresh, iou_thresh)

        print('\n{} Add keys for new labels ...'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        id = label.dataset['annotations'][-1]['id'] + 1

        if label_format == 'segm':
            if num_j == 0:
                for i, x in enumerate(tqdm.tqdm(new_labels, ascii=True)):
                    new_labels[i] = self.process_mask(x, label_mode)
            else:
                with Pool(num_j) as p:
                    new_labels = list(tqdm.tqdm(
                        p.imap(partial(self.process_mask, label_mode=label_mode), new_labels),
                        total=len(new_labels), ascii=True
                    ))
                    p.close()

            for x in tqdm.tqdm(new_labels, ascii=True):
                x['id'] = id
                id += 1
        else:
            for x in tqdm.tqdm(new_labels, ascii=True):
                x['area'] = float(x['bbox'][2] * x['bbox'][3])
                x['id'] = id
                id += 1

        label.dataset['annotations'].extend(new_labels)

        os.makedirs(os.path.dirname(result_files['pseudo_label']), exist_ok=True)
        mmcv.dump(label.dataset, result_files['pseudo_label'])
        print('\n{} Save pseudo label file to {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), result_files['pseudo_label']))

        return result_files

    def results2json(self, results, outfile_prefix,
                     label_mode='rle', thresh=0.75, iou_thresh=0., label_format='segm', num_j=0):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['pseudo_label'] = f'{outfile_prefix}.pseudo_label.thresh{thresh:.2f}.iou{iou_thresh:.2f}.json'
            mmcv.dump(json_results, result_files['bbox'])
            return self.parse_results(json_results, label_mode, result_files, thresh, iou_thresh,
                                      label_format='bbox')
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            result_files['pseudo_label'] = f'{outfile_prefix}.pseudo_label.thresh{thresh:.2f}.iou{iou_thresh:.2f}.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
            return self.parse_results(json_results[1], label_mode, result_files, thresh, iou_thresh,
                                      label_format='segm', num_j=num_j)
        elif isinstance(results, str):
            result_files[label_format] = results  # 'segm' or 'bbox'
            result_files['pseudo_label'] = f'{outfile_prefix}.pseudo_label.thresh{thresh:.2f}.iou{iou_thresh:.2f}.json'
            with open(result_files[label_format]) as f:
                new_labels = json.load(f)
            return self.parse_results(new_labels, label_mode, result_files, thresh, iou_thresh, label_format,
                                      num_j=num_j)
        else:
            return super().results2json(results, outfile_prefix)

    def format_results(self, results, jsonfile_prefix=None,
                       label_mode='rle', thresh=0.75, iou_thresh=0., label_format='segm', num_j=0):
        if isinstance(results, str):
            tmp_dir = None
            result_files = self.results2json(results, jsonfile_prefix,
                                             label_mode=label_mode, thresh=thresh,
                                             iou_thresh=iou_thresh, label_format=label_format, num_j=num_j)
            return result_files, tmp_dir

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix,
                                         label_mode=label_mode, thresh=thresh,
                                         iou_thresh=iou_thresh, label_format=label_format)
        return result_files, tmp_dir
