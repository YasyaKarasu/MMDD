# AbeBooks retrieval viewer

An offline HTML/JavaScript viewer for the existing 2×2 experiment. The Python
entrypoint exports saved traces and serves only the viewer, exported JSON, and
images registered by asset ID. It does not load models or API configuration.

From an isolated working directory, using the MMDD environment:

```bash
cd /tmp
conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/retrieval_viewer.py build
conda run --no-capture-output -n MMDD python /home/oycy/MMDD/src/retrieval_viewer.py serve --host 0.0.0.0 --port 8770
```

`--run-root` selects an experiment with baseline/hubs/columns/both subdirectories.
`--output` overrides the exported data directory (default: `<run-root>/viewer`).
The running instance is at **http://10.130.141.43:8770**; its PID, launch command,
logs and browser check receipt are saved in that viewer directory.

The page provides:

- All 150 queries: train (120), dev (15), test (15).
- Four dataset versions; Raw and selected/full-epoch SUP/KD on dev/test.
- Raw train traces only, because those are the saved natural train pools.
- Query → text/image first-hop ranking → second-hop target ranking, with scores.
- Pair-specific gold-evidence flags, D1 retention, C150 admission, Teacher ranks.
- Complete visible query/target tables and full evidence text/images.
- Direct query→target top100 and a separate gold-label inspection view.
- Shareable URLs preserving dataset, split, model, query, evidence and view.

Rank semantics: first-hop ranks are separate for text and image. Second-hop
ranks belong to the selected evidence. D1 retention happens before C150
admission. Gold annotations are for the current query–target pair; a nongold
asset is not automatically known to be irrelevant. Existing recovery values
are labels for inspection, not newly executed attribute-recovery results.

The server includes an optional local Chinese font route (`/font.otf`). The
current instance has `viewer-sc.otf` downloaded from the Noto CJK project's
`Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf`, distributed under the SIL
Open Font License. Its license is saved as `FONT-LICENSE.txt` beside the font.
The large font and generated datasets stay in the experiment output directory.
Browsers with installed Chinese fonts use those first.

Validation:

```bash
cd /tmp
conda run --no-capture-output -n MMDD python -m pytest /home/oycy/MMDD/tests/test_retrieval_viewer.py -q
```

Browser checks use Chromium/Playwright against the LAN IP, including dataset
and model switching, missing-gold deep links, filters, table dialogs, and a
390-pixel mobile viewport. The page has no CDN or external API dependency.
