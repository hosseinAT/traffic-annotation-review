# SPDX-License-Identifier: Apache-2.0
"""Export the review queue as JSON, CSV and an offline HTML report."""

import csv
import io
import json
import tempfile
from html import escape
from pathlib import Path

from .review import KINDS


LABELS = {
    'missing_box': 'Möglicherweise fehlende Box',
    'class_conflict': 'Abweichende Klasse',
    'box_alignment': 'Abweichende Box',
    'duplicate_box': 'Mögliche doppelte Box',
    'box_out_of_bounds': 'Box außerhalb des Bildes',
}


def _write_file(path, text):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=path.parent, mode='w', encoding='utf-8', newline='',
                delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(text)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _csv_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def _rows_csv(findings):
    output = io.StringIO(newline='')
    columns = [
        'review_id', 'kind', 'image_id', 'file_name', 'annotation_id',
        'reference_class', 'reference_xywh', 'prediction_class',
        'prediction_xywh', 'prediction_score', 'iou', 'other_annotation_id',
        'review_decision',
    ]
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    for row in findings:
        gt = row['annotation'] or {}
        pred = row['prediction'] or {}
        other = row['other_annotation'] or {}
        cells = dict(
            review_id=row['review_id'], kind=row['kind'],
            image_id=row['image_id'], file_name=row['file_name'],
            annotation_id=gt.get('annotation_id'),
            reference_class=gt.get('category'),
            reference_xywh=json.dumps(gt['bbox']) if gt else '',
            prediction_class=pred.get('category'),
            prediction_xywh=json.dumps(pred['bbox']) if pred else '',
            prediction_score=pred.get('score'), iou=row['iou'],
            other_annotation_id=other.get('annotation_id'),
            review_decision='',
        )
        writer.writerow({key: _csv_cell(value) for key, value in cells.items()})
    return output.getvalue()


def _box_label(box):
    if box is None:
        return '—'
    coords = ', '.join(f'{x:.1f}' for x in box['bbox'])
    annotation = f" · ID {box['annotation_id']}" if box['annotation_id'] is not None else ''
    confidence = f" · Score {box['score']:.3f}" if box['score'] is not None else ''
    return f"{escape(box['category'])}{annotation}{confidence}<br><small>xywh {coords}</small>"


def _html(result, display_limit=500):
    counts = result['summary']['findings']
    total = len(result['findings'])
    cards = ''.join(f'<li><strong>{counts[kind]}</strong>{LABELS[kind]}</li>' for kind in KINDS)
    rows = []
    for row in result['findings'][:display_limit]:
        other = _box_label(row['other_annotation']) if row['other_annotation'] else ''
        overlap = f"{row['iou']:.3f}" if row['iou'] is not None else '—'
        rows.append(
            f"<tr><td>{row['review_id']}</td><td>{LABELS[row['kind']]}</td>"
            f"<td>{escape(row['file_name'])}<br><small>ID {row['image_id']}</small></td>"
            f"<td>{_box_label(row['annotation'])}</td>"
            f"<td>{other or _box_label(row['prediction'])}</td><td>{overlap}</td></tr>")
    body = '\n'.join(rows) or '<tr><td colspan="6">Keine Prüfvorschläge im Bericht.</td></tr>'
    coverage = result['coverage']
    evaluated = coverage['verified_evaluated_images']
    if evaluated is None:
        coverage_text = (f"{coverage['images_with_predictions']} Bilder mit Vorhersagen bei "
                         f"{coverage['images_in_annotations']} Einträgen. "
                         "JSON ohne separate Bildliste liefert keine vollständige Laufabdeckung.")
    else:
        coverage_text = f"{evaluated} ausgewertete Bilder bei {coverage['images_in_annotations']} Einträgen."
    subtitle = f'{total} Prüfvorschläge · {result["summary"]["images_with_findings"]} betroffene Bilder'
    notes = (f'Die Tabelle zeigt {min(total, display_limit)} von {total} Zeilen. '
             'Die CSV enthält alle Vorschläge und eine Spalte für die manuelle Entscheidung.')
    return f'''<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prüfbericht für Bildannotationen</title>
<style>
* {{box-sizing:border-box}} body {{margin:0;background:#eef3f7;color:#152333;font:16px/1.55 system-ui,sans-serif}}
main {{max-width:1300px;margin:auto;padding:36px 24px 64px}}
h1 {{margin:0;font-size:clamp(1.8rem,4vw,2.8rem);line-height:1.15}} h2 {{font-size:1.25rem}}
header {{background:#173454;color:white;border-radius:16px;padding:30px}}
header p {{color:#d6e6f5;margin:10px 0 0}} a {{color:#144777}}
ul {{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;padding:0;list-style:none}}
li {{background:white;border-radius:12px;padding:13px 17px}} li strong {{display:block;font-size:1.6rem}}
section {{background:white;border-radius:12px;padding:20px}} .table {{overflow:auto}}
table {{width:100%;border-collapse:collapse;text-align:left}}
th, td {{padding:11px 10px;vertical-align:top;border-bottom:1px solid #dbe5ed}}
th {{background:#edf5fc}} small {{color:#4a6073}} td {{overflow-wrap:anywhere}}
</style></head><body><main>
<header><h1>Prüfbericht für Bildannotationen</h1><p>{subtitle}</p><p>{coverage_text}</p></header>
<ul aria-label="Anzahl je Prüfgrund">{cards}</ul>
<section><h2>Prüfliste</h2>
<p>Vergleiche die markierten Boxen und Klassen im Originalbild. Modellvorhersagen sind Hinweise für die Prüfung.</p>
<p>{notes} <a href="review.csv">CSV öffnen</a> · <a href="review.json">JSON öffnen</a></p>
<div class="table"><table><thead><tr><th>Nr.</th><th>Prüfgrund</th><th>Bild</th><th>Annotation</th>
<th>Vorhersage oder zweite Annotation</th><th>IoU</th></tr></thead>
<tbody>{body}</tbody></table></div></section>
</main></body></html>'''


def write_report(out_dir, result):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_file(out / 'review.json',
                json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    _write_file(out / 'review.csv', _rows_csv(result['findings']))
    _write_file(out / 'review.html', _html(result))
