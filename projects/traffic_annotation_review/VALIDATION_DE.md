# Lokale Prüfung der Extension

Die Prüfung vom **24. November 2026** verwendete Python 3.11.18,
PyTorch 2.1.0 CPU, MMCV 2.1.0, MMEngine 0.10.7 und MMDetection 3.3.0.

| Prüfung | Ausgeführter Test | Ergebnis |
|---|---|---|
| Unit-Tests | Mehrdeutige Zuordnung, Crowd-Ausnahmen, Klassen-IDs, Eingabefehler und Exporte | 7 bestanden |
| Offline-Funktionstest | Mitgelieferte JSON-Dateien mit festgelegten Boxen | 5 erwartete Prüfvorschläge |
| Integration auf realen Bildern | Offizielles RTMDet-Tiny-Modell auf 12 Verkehrsbildern aus COCO val2017 mit 122 ursprünglichen Annotationen | Mit und ohne Review-Erweiterung bitgleiche Vorhersagedateien und gleiche Standard-COCO-Kennzahlen |
| Kontrollierte Korrekturprobe | Dieselben 12 Bilder mit fünf absichtlich veränderten Annotationen in einer Kopie | Alle 5 eingebauten Fälle als Prüfvorschläge gefunden |
| Vergleich der Zugangswege | Dieselben kontrollierten Labels und exportierten Modellvorhersagen per CLI | Dieselben fünf Vorschläge wie in MMDetection |

Die unveränderte Teilmenge ergab `bbox_mAP 0.460`, `bbox_mAP_50 0.600` und
`bbox_mAP_75 0.539` in beiden Läufen. Der Review-Evaluator meldete bei den
gewählten Schwellen 25 berücksichtigte Vorhersagen und keine Prüfvorschläge.
Die zwölf Bilder dienen als Integrationstest. Für Aussagen zur allgemeinen
Erkennungsleistung oder zur Qualität eines Datensatzes braucht es eine
repräsentative und manuell geprüfte Stichprobe.

In der kontrollierten Kopie änderte sich die mAP auf 0.416. Die
Vorhersagen blieben bitgleich. Dieser Vergleich zeigt, wie die Änderungen an
den Referenzboxen die Auswertung beeinflussen. Die fünf Beispielobjekte wurden
nach einem ersten Lauf aus zuverlässig zugeordneten Fällen ausgewählt.
Die erreichten fünf Treffer sind eine Softwareprüfung für diese Fälle.

| Bild-ID | Ausgangsannotation | Absichtliche Änderung | Befund |
|---:|---:|---|---|
| 1584 | 365622 | Bus-Box entfernt | Möglicherweise fehlende Box |
| 2006 | 165069 | Bus als Auto markiert | Abweichende Klasse |
| 5001 | 1328232 | Personen-Box um 55 Prozent ihrer Breite verschoben | Abweichende Box |
| 5037 | 169115 | Bus-Box mit neuer ID dupliziert | Mögliche doppelte Box |
| 6040 | 172843 | Rechte Kante um acht Pixel über die Bildgrenze gesetzt | Box außerhalb des Bildes |

[Prüfbericht der kontrollierten Variante](validation/controlled_report/review.html)
· [Prüfbericht der unveränderten Variante](validation/original_report/review.html)
· [Messdaten und Prüfsummen](validation/run_record.json)

## Datenherkunft und Wiederholung

Die Liste der 12 Bild-IDs und die Verteilung der Ausgaben stehen im
Messprotokoll. Der Originaldatensatz COCO 2017 val wurde für den Versuch lokal
aus öffentlich verfügbaren Kopien geladen. Die Bilder und der
Modell-Checkpoint werden bei der Wiederholung separat bezogen. Die
Beispielausschnitte enthalten nur Koordinaten, Klassen und Berichtszahlen.
[COCO](https://cocodataset.org/#download)
· [Annotationskopie](https://huggingface.co/datasets/pcuenq/coco2017-instances/blob/main/instances_val2017.json)
· [Bildkopien](https://huggingface.co/datasets/merve/coco/tree/main/val2017)

`instances_val2017.json` SHA-256
`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.

RTMDet-Tiny-Checkpoint SHA-256
`78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb`.
Beide MMDetection-Konfigurationen meldeten beim Laden dieselben zwei
zusätzlichen Präprozessorschlüssel `data_preprocessor.mean` und `std`.
Der vollständige Testlauf und der Vergleich der Vorhersagedateien waren
erfolgreich.
[Offizieller RTMDet-Checkpoint](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth)

Für den Modelllauf wird MMDetection 3.3.0 separat installiert. Kopiere
`projects/traffic_annotation_review/` an dieselbe Stelle im Framework und
führe die Befehle in dessen Hauptordner aus.

```bash
python -m projects.traffic_annotation_review.prepare_coco_subset \
  --annotations /path/to/instances_val2017.json --local-images /path/to/val2017 \
  --out-dir data/coco_traffic_subset
python -m projects.traffic_annotation_review.make_validation_variant \
  --annotations data/coco_traffic_subset/annotations/instances_val2017.json \
  --out data/coco_traffic_subset/annotations/controlled_edits.json
```

Der Modelllauf mit den gezielt veränderten Annotationen schreibt die
Vorhersagen und den Prüfbericht nach `work_dirs/traffic_review/`.

```bash
python -m tools.test projects/traffic_annotation_review/configs/rtmdet_tiny_review.py \
  /path/to/rtmdet_tiny.pth --work-dir work_dirs/traffic_review \
  --cfg-options test_dataloader.num_workers=0 test_dataloader.persistent_workers=False \
  test_dataloader.dataset.data_root=data/coco_traffic_subset \
  test_dataloader.dataset.ann_file=annotations/controlled_edits.json \
  test_evaluator.ann_file=data/coco_traffic_subset/annotations/controlled_edits.json \
  test_evaluator.review_dir=work_dirs/traffic_review/review \
  test_evaluator.outfile_prefix=work_dirs/traffic_review/predictions
```

Für die Vergleichsmessung mit unveränderter Annotation ist
`configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py` die Originalkonfiguration.
Verwende dabei `instances_val2017.json` in beiden Annotationspfaden.
In diesem Lauf entfällt die Option `test_evaluator.review_dir`.
Das Modell wurde mit vorhandenen offiziellen Gewichten verwendet.
