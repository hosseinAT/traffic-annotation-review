# SPDX-License-Identifier: Apache-2.0
"""Prepare a small, reproducible COCO batch for integration verification."""

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


SOURCES = {
    'coco': 'https://images.cocodataset.org/val2017/',
    'mirror': 'https://huggingface.co/datasets/merve/coco/resolve/main/val2017/',
}
TRAFFIC = {'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'traffic light'}


def prepare(annotation_file, output, count, local_images=None, image_source='coco'):
    if count < 1:
        raise ValueError('count must be positive')
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    original = annotation_file.read_bytes()
    document = json.loads(original)
    category_names = {row['name']: row['id'] for row in document['categories']}
    road_ids = {category_names[name] for name in TRAFFIC}
    image_categories = defaultdict(set)
    for row in document['annotations']:
        if not row.get('iscrowd'):
            image_categories[row['image_id']].add(row['category_id'])
    selected = sorted(
        (image for image in document['images']
         if category_names['person'] in image_categories[image['id']]
         and road_ids & image_categories[image['id']]),
        key=lambda row: row['id'],
    )[:count]
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} images meet the sample criteria.')
    image_ids = {row['id'] for row in selected}
    subset = {
        **document,
        'images': selected,
        'annotations': [row for row in document['annotations']
                        if row['image_id'] in image_ids],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp_dir:
        staging = Path(temp_dir) / 'subset'
        (staging / 'val2017').mkdir(parents=True)
        (staging / 'annotations').mkdir()

        def copy_image(row):
            filename = row['file_name']
            if not re.fullmatch(r'\d{12}\.jpg', filename):
                raise ValueError(f'Unexpected COCO val2017 file name: {filename}')
            target = staging / 'val2017' / filename
            if local_images is not None:
                shutil.copyfile(local_images / filename, target)
            else:
                with urllib.request.urlopen(SOURCES[image_source] + filename,
                                            timeout=30) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
            with Image.open(target) as image:
                if image.size != (row['width'], row['height']):
                    raise ValueError(f'Image dimensions disagree with COCO: {filename}')
                image.verify()

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(copy_image, selected))
        (staging / 'annotations' / 'instances_val2017.json').write_text(
            json.dumps(subset, ensure_ascii=False), encoding='utf-8')
        selection = dict(
            annotation_sha256=hashlib.sha256(original).hexdigest(),
            selection='Lowest COCO IDs with a non-crowd person and at least one traffic category',
            image_ids=sorted(image_ids),
            annotation_count=len(subset['annotations']),
            image_source=SOURCES[image_source] if local_images is None else 'local val2017 folder',
        )
        (staging / 'selection.json').write_text(
            json.dumps(selection, indent=2) + '\n', encoding='utf-8')
        staging.rename(output)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--local-images', type=Path)
    parser.add_argument('--image-source', choices=SOURCES, default='coco')
    args = parser.parse_args()
    print(json.dumps(prepare(args.annotations, args.out_dir, args.count,
                             args.local_images, args.image_source), indent=2))


if __name__ == '__main__':
    main()
