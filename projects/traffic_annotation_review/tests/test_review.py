import json
from pathlib import Path

import numpy as np
import pytest

from projects.traffic_annotation_review.review import _assign, review_annotations
from projects.traffic_annotation_review.report import write_report


@pytest.fixture
def data():
    root = Path(__file__).resolve().parents[1] / 'examples'
    return (
        json.loads((root / 'annotations.json').read_text()),
        json.loads((root / 'predictions.bbox.json').read_text()),
    )


def test_review_spots_deliberate_cases_without_relabeling_inputs(data):
    baseline = json.dumps(data)
    result = review_annotations(*data)
    assert json.dumps(data) == baseline
    assert result['summary']['matched_boxes'] == 2
    assert result['summary']['findings'] == {
        'missing_box': 1, 'class_conflict': 1, 'box_alignment': 1,
        'duplicate_box': 1, 'box_out_of_bounds': 1,
    }
    missing = next(row for row in result['findings'] if row['kind'] == 'missing_box')
    assert missing['prediction']['category_id'] == 10
    duplicate = next(row for row in result['findings'] if row['kind'] == 'duplicate_box')
    assert 0 <= duplicate['iou'] <= 1
    assert result['coverage']['verified_evaluated_images'] is None


def test_matching_preserves_two_pairs_when_greedy_takes_best_first():
    overlaps = np.array([[0.95, 0.60], [0.70, 0.20]])
    pairs = _assign(overlaps, overlaps >= 0.5, set(), set())
    assert set(pairs) == {(0, 1), (1, 0)}


def test_missing_proposals_respect_crowd_duplicates_and_low_scores(data):
    result = review_annotations(*data)
    missing = [row for row in result['findings'] if row['kind'] == 'missing_box']
    assert [row['prediction']['bbox'] for row in missing] == [[300, 90, 14, 36]]


def test_native_coverage_includes_empty_predictions(data):
    truth, _ = data
    result = review_annotations(truth, [], evaluated_ids=[22, 44])
    assert result['coverage'] == {
        'images_in_annotations': 4, 'images_with_predictions': 0,
        'verified_evaluated_images': 2,
    }
    assert result['summary']['findings']['duplicate_box'] == 0
    assert result['summary']['findings']['box_out_of_bounds'] == 1


def test_export_rejects_mismatched_category_schema(data):
    truth, preds = data
    preds[0]['category_id'] = 2
    with pytest.raises(ValueError, match='unknown image/category'):
        review_annotations(truth, preds)


def test_invalid_truth_area_is_actionable(data):
    truth, preds = data
    truth['annotations'][0]['bbox'][2] = 0
    with pytest.raises(ValueError, match='Invalid box dimensions'):
        review_annotations(truth, preds)


def test_report_exports_escaped_markup_and_review_decision_column(data, tmp_path):
    truth, preds = data
    truth['images'][0]['file_name'] = '<photo>.jpg'
    result = review_annotations(truth, preds)
    write_report(tmp_path, result)
    html = (tmp_path / 'review.html').read_text(encoding='utf-8')
    csv = (tmp_path / 'review.csv').read_text(encoding='utf-8')
    assert '&lt;photo&gt;.jpg' in html and '<photo>.jpg' not in html
    assert 'review_decision' in csv
    assert json.loads((tmp_path / 'review.json').read_text(encoding='utf-8')) == result
