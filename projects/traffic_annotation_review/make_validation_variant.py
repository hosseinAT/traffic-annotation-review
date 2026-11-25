# SPDX-License-Identifier: Apache-2.0
"""Inject five documented edits into the 12-image COCO validation fixture."""

import argparse
import json
from pathlib import Path


def build_variant(input_file, output_file):
    document = json.loads(input_file.read_bytes())
    annotation = {row['id']: row for row in document['annotations']}
    images = {row['id']: row for row in document['images']}
    targets = {365622: 1584, 165069: 2006, 1328232: 5001, 169115: 5037, 172843: 6040}
    if any(ann_id not in annotation or annotation[ann_id]['image_id'] != image_id
           for ann_id, image_id in targets.items()):
        raise ValueError('Use the clean 12-image subset from prepare_coco_subset.')

    document['annotations'].remove(annotation[365622])
    annotation[165069]['category_id'] = next(
        row['id'] for row in document['categories'] if row['name'] == 'car')
    shifted = annotation[1328232]
    shifted['bbox'][0] += shifted['bbox'][2] * 0.55
    duplicate = {**annotation[169115], 'id': max(annotation) + 1}
    document['annotations'].append(duplicate)
    beyond = annotation[172843]
    beyond['bbox'][2] = images[beyond['image_id']]['width'] - beyond['bbox'][0] + 8
    with output_file.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, ensure_ascii=False)
    return [
        ['missing_box', 1584, 365622],
        ['class_conflict', 2006, 165069],
        ['box_alignment', 5001, 1328232],
        ['duplicate_box', 5037, 169115],
        ['box_out_of_bounds', 6040, 172843],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_variant(args.annotations, args.out), indent=2))


if __name__ == '__main__':
    main()
