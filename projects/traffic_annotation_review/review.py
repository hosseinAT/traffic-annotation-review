# SPDX-License-Identifier: Apache-2.0
"""Match COCO predictions and annotations without changing labels."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real

import numpy as np
from scipy.optimize import linear_sum_assignment


KINDS = (
    'missing_box', 'class_conflict', 'box_alignment', 'duplicate_box',
    'box_out_of_bounds',
)


@dataclass(frozen=True)
class ReviewConfig:
    min_score: float = 0.65
    match_iou: float = 0.50
    candidate_iou: float = 0.20
    class_iou: float = 0.65
    duplicate_iou: float = 0.90
    ignore_ioa: float = 0.50

    def __post_init__(self):
        if not all(isfinite(x) and 0 < x <= 1 for x in asdict(self).values()):
            raise ValueError('All review thresholds must be in (0, 1].')
        if not self.candidate_iou < self.match_iou <= self.class_iou:
            raise ValueError('Require candidate_iou < match_iou <= class_iou.')


@dataclass(frozen=True)
class Box:
    category_id: int
    bbox: tuple[float, float, float, float]
    annotation_id: int | None = None
    score: float | None = None
    ignored: bool = False

    def json(self, classes):
        return dict(
            bbox=list(self.bbox),
            category_id=self.category_id,
            category=classes[self.category_id],
            annotation_id=self.annotation_id,
            score=self.score,
        )


def _integer(row, key):
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f'{key} must be an integer, got {value!r}.')
    return value


def _index(rows, name):
    if not isinstance(rows, list):
        raise ValueError(f'{name} must be a list.')
    index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f'{name} entries must be objects.')
        identifier = _integer(row, 'id')
        if identifier in index:
            raise ValueError(f'Duplicate {name} ID {identifier}.')
        index[identifier] = row
    return index


def _bbox(row):
    coords = row.get('bbox')
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        raise ValueError(f'Expected a COCO xywh box: {coords!r}.')
    if not all(isinstance(x, Real) and not isinstance(x, bool)
               and isfinite(x) for x in coords):
        raise ValueError(f'Nonfinite or nonnumeric box: {coords!r}.')
    x, y, w, h = map(float, coords)
    if min(w, h) <= 0 or not all(isfinite(v) for v in (x + w, y + h, w * h)):
        raise ValueError(f'Invalid box dimensions: {coords!r}.')
    return x, y, w, h


def _coverage_ids(values, images):
    if values is None:
        return None
    image_ids = set(values)
    if not image_ids <= images.keys():
        raise ValueError('Evaluated image IDs must exist in the annotation file.')
    return image_ids


def _parse(document, detections, evaluated_ids, category_ids):
    if not isinstance(document, dict) or not isinstance(detections, list):
        raise ValueError('Supply a COCO annotation object and a prediction list.')
    images = _index(document.get('images'), 'image')
    categories = _index(document.get('categories'), 'category')
    annotations = _index(document.get('annotations'), 'annotation')
    if not categories:
        raise ValueError('The dataset needs at least one category.')
    classes = {}
    for category_id, row in categories.items():
        name = row.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'Category {category_id} needs a name.')
        classes[category_id] = name
    for image_id, row in images.items():
        if not isinstance(row.get('file_name'), str) or not row['file_name']:
            raise ValueError(f'Image {image_id} needs a file_name.')
        if _integer(row, 'width') < 1 or _integer(row, 'height') < 1:
            raise ValueError(f'Image {image_id} has invalid dimensions.')

    evaluated = _coverage_ids(evaluated_ids, images)
    scope = set(classes) if category_ids is None else set(category_ids)
    if not scope or not scope <= classes.keys():
        raise ValueError('Review categories must exist in the annotation file.')
    ground_truth = defaultdict(list)
    for annotation_id, row in annotations.items():
        image_id = _integer(row, 'image_id')
        category_id = _integer(row, 'category_id')
        if image_id not in images or category_id not in classes:
            raise ValueError(f'Annotation {annotation_id} has an unknown image/category.')
        if (evaluated is None or image_id in evaluated) and category_id in scope:
            ground_truth[image_id].append(Box(
                category_id, _bbox(row), annotation_id=annotation_id,
                ignored=bool(row.get('iscrowd', 0) or row.get('ignore', 0))))
    predictions = defaultdict(list)
    for row in detections:
        if not isinstance(row, dict):
            raise ValueError('Prediction entries must be objects.')
        image_id = _integer(row, 'image_id')
        category_id = _integer(row, 'category_id')
        if image_id not in images or category_id not in scope:
            raise ValueError(f'Prediction refers to unknown image/category {image_id}/{category_id}.')
        if evaluated is not None and image_id not in evaluated:
            raise ValueError(f'Prediction for image {image_id} is outside the declared coverage.')
        score = row.get('score')
        if not isinstance(score, Real) or isinstance(score, bool) or not isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Prediction for image {image_id} has an invalid score.')
        predictions[image_id].append(Box(category_id, _bbox(row), score=float(score)))
    return images, classes, ground_truth, predictions, evaluated, scope


def _overlaps(first, second, *, first_area_only=False):
    """IoU, or intersection over first-box area for ignore regions."""
    a = np.asarray([item.bbox for item in first], dtype=np.float64).reshape(-1, 4)
    b = np.asarray([item.bbox for item in second], dtype=np.float64).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum((a[:, :2] + a[:, 2:])[:, None, :],
                    (b[:, :2] + b[:, 2:])[None, :, :])
    intersection = np.maximum(hi - lo, 0).prod(axis=2)
    area_a, area_b = a[:, 2:].prod(axis=1), b[:, 2:].prod(axis=1)
    denominator = area_a[:, None] if first_area_only else (
        area_a[:, None] - intersection + area_b[None, :])
    ratios = np.divide(intersection, denominator, out=np.zeros_like(intersection),
                       where=denominator > 0)
    return np.clip(ratios, 0, 1)


def _same_category(first, second):
    return np.equal.outer([item.category_id for item in first],
                          [item.category_id for item in second])


def _assign(overlaps, valid, used_pred, used_gt):
    """Maximize eligible pairs first, then maximize their summed IoU."""
    if not valid.size:
        return []
    valid = valid.copy()
    valid[list(used_pred), :] = False
    valid[:, list(used_gt)] = False
    bonus = min(valid.shape) + 1
    pairs = []
    for i, j in zip(*linear_sum_assignment(np.where(valid, bonus + overlaps, 0),
                                           maximize=True)):
        if valid[i, j]:
            used_pred.add(int(i))
            used_gt.add(int(j))
            pairs.append((int(i), int(j)))
    return pairs


def _image_findings(image, truth, predictions, classes, config):
    findings = []

    def add(kind, *, annotation=None, predicted=None, other=None, iou=None):
        findings.append(dict(
            kind=kind,
            image_id=image['id'],
            file_name=image['file_name'],
            annotation=annotation.json(classes) if annotation else None,
            prediction=predicted.json(classes) if predicted else None,
            other_annotation=other.json(classes) if other else None,
            iou=float(iou) if iou is not None else None,
        ))

    ordinary = sorted((box for box in truth if not box.ignored),
                      key=lambda box: box.annotation_id)
    ignored = [box for box in truth if box.ignored]
    predictions = sorted((box for box in predictions if box.score >= config.min_score),
                         key=lambda box: (-box.score, box.category_id, box.bbox))
    overlaps = _overlaps(predictions, ordinary)
    same = _same_category(predictions, ordinary)
    used_pred, used_gt = set(), set()
    matched = len(_assign(overlaps, same & (overlaps >= config.match_iou),
                          used_pred, used_gt))

    # A detection in an ignored region may represent a valid but uncounted object.
    inside_ignore = (_overlaps(predictions, ignored, first_area_only=True) >= config.ignore_ioa).any(axis=1)
    used_pred.update(np.flatnonzero(inside_ignore).tolist())
    conflicts = _assign(overlaps, ~same & (overlaps >= config.class_iou),
                        used_pred, used_gt)
    for i, j in conflicts:
        add('class_conflict', annotation=ordinary[j], predicted=predictions[i],
            iou=overlaps[i, j])
    shifted = _assign(overlaps, same & (overlaps >= config.candidate_iou)
                      & (overlaps < config.match_iou), used_pred, used_gt)
    for i, j in shifted:
        add('box_alignment', annotation=ordinary[j], predicted=predictions[i],
            iou=overlaps[i, j])
    protected = (overlaps >= config.candidate_iou).any(axis=1)
    for i, prediction in enumerate(predictions):
        if i not in used_pred and not protected[i]:
            add('missing_box', predicted=prediction)

    gt_overlap = _overlaps(ordinary, ordinary)
    duplicates = np.triu(_same_category(ordinary, ordinary)
                         & (gt_overlap >= config.duplicate_iou), k=1)
    for i, j in zip(*np.where(duplicates)):
        add('duplicate_box', annotation=ordinary[i], other=ordinary[j],
            iou=gt_overlap[i, j])
    for box in ordinary:
        x, y, w, h = box.bbox
        tolerance = 1e-3
        if x < -tolerance or y < -tolerance or x + w > image['width'] + tolerance or y + h > image['height'] + tolerance:
            add('box_out_of_bounds', annotation=box)
    return findings, matched, len(predictions)


def review_annotations(document, detections, *, config=None, evaluated_ids=None, category_ids=None):
    """Return prioritized review candidates and report the observed coverage."""
    config = config or ReviewConfig()
    images, classes, truth, predictions, evaluated, scope = _parse(
        document, detections, evaluated_ids, category_ids)
    findings, matched, kept = [], 0, 0
    for image_id in sorted(images if evaluated is None else evaluated):
        rows, image_matched, image_kept = _image_findings(
            images[image_id], truth[image_id], predictions[image_id], classes, config)
        findings.extend(rows)
        matched += image_matched
        kept += image_kept
    findings.sort(key=lambda row: (
        KINDS.index(row['kind']), -(row['prediction'] or {}).get('score', 0),
        row['image_id'],
        (row['annotation'] or {}).get('annotation_id', 0),
    ))
    for index, finding in enumerate(findings, 1):
        finding['review_id'] = index
    counts = Counter(row['kind'] for row in findings)
    return dict(
        thresholds=asdict(config), category_ids=sorted(scope),
        coverage=dict(images_in_annotations=len(images),
                      images_with_predictions=sum(bool(value) for value in predictions.values()),
                      verified_evaluated_images=len(evaluated) if evaluated is not None else None),
        summary=dict(matched_boxes=matched, high_score_predictions=kept,
                     images_with_findings=len({row['image_id'] for row in findings}),
                     findings={kind: counts[kind] for kind in KINDS}),
        findings=findings,
    )
