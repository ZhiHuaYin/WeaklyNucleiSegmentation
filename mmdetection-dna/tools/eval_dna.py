import argparse
import itertools
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.api_wrappers import COCOeval as _COCOeval

import datetime
import time


class COCOeval(_COCOeval):
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((16,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=100)
            stats[2] = _summarize(1, iouThr=.75, maxDets=100)
            stats[3] = _summarize(1, iouThr=.9, maxDets=100)
            stats[4] = _summarize(1, iouThr=.5, maxDets=1000)
            stats[5] = _summarize(1, iouThr=.75, maxDets=1000)
            stats[6] = _summarize(1, iouThr=.9, maxDets=1000)
            stats[7] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[10] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[12] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[13] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[14] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[15] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'bbox' or iouType == 'segm':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            raise NotImplementedError
        self.stats = summarize()

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', default='../data/coco_dna/annotations_new/test.json', type=str)
    parser.add_argument('pred', type=str)

    args = parser.parse_args()
    return args


def evaluate(
        gt, json, metrics,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
        metric_items=None):
    coco = COCO(annotation_file=gt)

    eval_results = OrderedDict()
    cocoGt = coco
    for metric in metrics:
        msg = f'\nEvaluating {metric}...'
        print(msg)

        iou_type = 'bbox' if metric == 'proposal' else metric
        try:
            predictions = mmcv.load(json)
            if iou_type == 'segm':
                # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                # When evaluating mask AP, if the results contain bbox,
                # cocoapi will use the box area instead of the mask area
                # for calculating the instance area. Though the overall AP
                # is not affected, this leads to different
                # small/medium/large mask AP results.
                for x in predictions:
                    x.pop('bbox')
                warnings.simplefilter('once')
                warnings.warn(
                    'The key "bbox" is deleted for more accurate mask AP '
                    'of small/medium/large instances since v2.12.0. This '
                    'does not change the overall mAP calculation.',
                    UserWarning)
            cocoDt = cocoGt.loadRes(predictions)
        except IndexError:
            print('The testing results of the whole dataset is empty.')
            break

        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cocoEval.params.catIds = coco.get_cat_ids()
        cocoEval.params.imgIds = coco.get_img_ids()
        if proposal_nums is not None:
            cocoEval.params.maxDets = list(proposal_nums)
        if iou_thrs is not None:
            cocoEval.params.iouThrs = np.array(iou_thrs)
        # mapping of cocoEval.stats
        if iou_type == 'bbox' or iou_type == 'segm':
            coco_metric_names = {
                'mAP': 0,
                'mAP_50@100': 1,
                'mAP_75@100': 2,
                'mAP_90@100': 3,
                'mAP_50@1000': 4,
                'mAP_75@1000': 5,
                'mAP_90@1000': 6,
                'AR@100': 7,
                'AR@300': 8,
                'AR@1000': 9,
                'mAP_s': 10,
                'mAP_m': 11,
                'mAP_l': 12,
                'AR_s@1000': 13,
                'AR_m@1000': 14,
                'AR_l@1000': 15
            }
        else:
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')

        if metric == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                    'AR_m@1000', 'AR_l@1000'
                ]

            for item in metric_items:
                val = float(
                    f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(coco.get_cat_ids()) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(coco.get_cat_ids()):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print('\n' + table.table)

            if metric_items is None:
                if iou_type == 'bbox' or iou_type == 'segm':
                    metric_items = [
                        'mAP', 'mAP_50@100', 'mAP_75@100', 'mAP_90@100',
                        'mAP_50@1000', 'mAP_75@1000', 'mAP_90@1000'
                    ]
                else:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val

            if iou_type == 'bbox' or iou_type == 'segm':
                ap = cocoEval.stats[:7]
                ar = cocoEval.stats[7:10]
                eval_results[f'{metric}_copypaste'] = (
                    f'{100 * ap[0]:.3f} {100 * ap[1]:.3f} {100 * ap[2]:.3f} {100 * ap[3]:.3f} '
                    f'{100 * ap[4]:.3f} {100 * ap[5]:.3f} {100 * ap[6]:.3f} '
                    f'{100 * ar[0]:.3f} {100 * ar[1]:.3f} {100 * ar[2]:.3f}'
                )

    return eval_results


def main():
    args = parse_args()

    metrics = ['bbox', 'segm']
    result = evaluate(args.label, args.pred, metrics)
    print(result)


if __name__ == '__main__':
    main()
