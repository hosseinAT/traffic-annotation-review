# SPDX-License-Identifier: Apache-2.0
_base_ = '../../../configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

custom_imports = dict(
    imports=['projects.traffic_annotation_review.metric'],
    allow_failed_imports=False,
)

test_evaluator = dict(
    type='TrafficAnnotationReviewMetric',
    review_dir='work_dirs/traffic_annotation_review/review',
    review_options=dict(min_score=0.65),
)
