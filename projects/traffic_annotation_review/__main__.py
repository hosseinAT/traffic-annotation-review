# SPDX-License-Identifier: Apache-2.0
"""Run the annotation review on COCO annotations and an exported bbox.json."""

import argparse
import hashlib
import json
from pathlib import Path

from .review import ReviewConfig, review_annotations
from .report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    defaults = ReviewConfig()
    for key, value in vars(defaults).items():
        parser.add_argument('--' + key.replace('_', '-'), type=float,
                            default=value)
    args = parser.parse_args()
    try:
        generated = {(args.out_dir / f'review.{ext}').resolve()
                     for ext in ('json', 'csv', 'html')}
        if any(p.resolve() in generated for p in (args.annotations, args.predictions)):
            raise ValueError('Choose an output directory that preserves both input files.')
        config = ReviewConfig(**{key: getattr(args, key) for key in vars(defaults)})
        annotation_bytes = args.annotations.read_bytes()
        prediction_bytes = args.predictions.read_bytes()
        result = review_annotations(
            json.loads(annotation_bytes), json.loads(prediction_bytes), config=config)
        result['inputs'] = dict(
            annotations_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
            predictions_sha256=hashlib.sha256(prediction_bytes).hexdigest())
        write_report(args.out_dir, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'{len(result["findings"])} review candidates written to {args.out_dir}')


if __name__ == '__main__':
    main()
