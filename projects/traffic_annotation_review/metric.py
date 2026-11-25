# SPDX-License-Identifier: Apache-2.0
"""MMDetection integration alongside the standard COCO evaluation."""

from mmengine.logging import MMLogger

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS

from .review import ReviewConfig, review_annotations
from .report import write_report


@METRICS.register_module()
class TrafficAnnotationReviewMetric(CocoMetric):
    """Preserve COCO metrics and export candidates for the processed images."""

    def __init__(self, review_dir, review_options=None, **kwargs):
        super().__init__(**kwargs)
        self.review_dir = review_dir
        self.review_config = ReviewConfig(**(review_options or {}))

    def compute_metrics(self, results):
        official_metrics = super().compute_metrics(results)
        predictions, evaluated_ids = [], set()
        for _, pred in results:
            image_id = int(pred['img_id'])
            evaluated_ids.add(image_id)
            for box, score, label in zip(pred['bboxes'], pred['scores'], pred['labels']):
                predictions.append(dict(
                    image_id=image_id, category_id=int(self.cat_ids[int(label)]),
                    bbox=self.xyxy2xywh(box), score=float(score)))
        result = review_annotations(
            self._coco_api.dataset, predictions, config=self.review_config,
            evaluated_ids=evaluated_ids, category_ids=self.cat_ids,
        )
        write_report(self.review_dir, result)
        MMLogger.get_current_instance().info(
            'Annotation review wrote %d review candidates to %s.',
            len(result['findings']), self.review_dir)
        return official_metrics
