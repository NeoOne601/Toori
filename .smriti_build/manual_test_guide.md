# Sprint 4 Manual Test Guide

## Goal

Verify that Smriti storage configuration is safe, understandable, and usable from the desktop Settings UI.

## Preconditions

1. Start the runtime:

```bash
cd /Users/macuser/toori
TOORI_DATA_DIR=.toori python3 -m uvicorn cloud.api.main:app --host 127.0.0.1 --port 7777
```

2. Start the desktop app:

```bash
cd /Users/macuser/toori/desktop/electron
npm start
```

3. Have at least one folder with a few image or video files available for watch-folder testing.

## Test 1: Choose Smriti storage location

1. Open `Settings`.
2. Scroll to `Smriti Storage`.
3. Click `Browse` next to `Primary Smriti data directory`.
4. Pick a new empty folder.
5. Save the settings.

Expected:
- The UI reports that storage settings were saved.
- The chosen folder is preserved after refresh.
- Subdirectories for frames/thumbs are created if needed.

## Test 2: Set storage budget and warning behavior

1. In `Smriti Storage`, set a small `Storage budget (GB)`, for example `0.1`.
2. Save.
3. Click `Refresh Usage`.

Expected:
- The usage panel shows the configured budget.
- The usage bar updates and displays warning colors when usage crosses thresholds.

## Test 3: Add and remove watched folders

1. Click `+ Add Folder`.
2. Choose a media folder.
3. Wait for the status panel to refresh.
4. Remove the same folder with `Remove`.

Expected:
- Added folders appear immediately in the watch-folder list.
- Each folder shows existence, indexed count, pending count, and watch status.
- Removing the folder removes it from the list and persists across refresh.

## Test 4: Watch-folder persistence across restart

1. Add a folder and confirm it appears in the list.
2. Quit the desktop app and stop the runtime.
3. Start both again.
4. Open `Settings > Smriti Storage`.

Expected:
- The watch folder is still listed.
- If the folder still exists, it becomes active again automatically.

## Test 5: Disk usage breakdown

1. Open `Settings > Smriti Storage`.
2. Compare the usage panel before and after adding or indexing media.

Expected:
- The UI shows values for frames, thumbnails, database, FAISS, and templates.
- The total usage is human-readable and updates on refresh.

## Test 6: Safe prune actions

1. Click `Remove records for deleted files`.
2. Click `Clear failed ingestions`.
3. Verify the status message after each action.

Expected:
- The action completes without crashing.
- Status text reports records removed and bytes freed.

## Test 7: Dangerous clear-all protection

1. Click `Clear All Smriti Data`.
2. Try to confirm without entering `CONFIRM_CLEAR_ALL`.
3. Enter `CONFIRM_CLEAR_ALL` and run it.

Expected:
- The destructive action is blocked until the exact confirmation string is present.
- The UI clearly warns that only Smriti-managed data is removed.
- Original source media outside Smriti storage is not deleted.

## Test 8: Browser fallback for folder entry

1. Run the web UI with `npm run web` instead of Electron.
2. Open `Settings > Smriti Storage`.
3. Try `Browse` for a watch folder or storage directory.

Expected:
- Browser mode falls back to a manual path prompt.
- Saving still works when a valid absolute path is entered.

## Regression checks

Run these after manual validation:

```bash
pytest -q cloud/api/tests cloud/jepa_service/tests cloud/search_service/tests cloud/monitoring/tests tests/test_readme.py
cd /Users/macuser/toori/desktop/electron && npm run typecheck && npm run build
pytest -q cloud/jepa_service/tests/test_cwma_ecgd_setu2.py -k "telescope_described_as_cylindrical"
python3.11 -W error::DeprecationWarning -c "from cloud.runtime.app import create_app; create_app(); print('CLEAN')"
```

## Test Suite 6: Sprint 5 Production Gate

### Test 6.1: Full Production Gate

```bash
pytest -v cloud/api/tests/test_smriti_production.py
```

Expected:
- All production-gate tests pass.
- `test_telescope_behind_person_not_described_as_body_part` passes.
- No test in this file is skipped or removed.

### Test 6.2: Data Migration Dry Run

```bash
mkdir -p ~/Desktop/smriti_migrate_test/smriti/frames
mkdir -p ~/Desktop/smriti_migrate_test/smriti/thumbs
python3.11 - <<'PY'
import sqlite3
from pathlib import Path
root = Path.home() / "Desktop/smriti_migrate_test/smriti"
db = root / "smriti.sqlite3"
conn = sqlite3.connect(db)
conn.execute("CREATE TABLE IF NOT EXISTS demo (id TEXT PRIMARY KEY, value TEXT)")
conn.execute("INSERT OR REPLACE INTO demo VALUES (?, ?)", ("row_1", "ok"))
conn.commit()
conn.close()
(root / "sag_templates.json").write_text("[]")
PY
curl -s -X POST http://127.0.0.1:7777/v1/smriti/storage/migrate \
  -H "Content-Type: application/json" \
  -d '{"target_data_dir": "/tmp/smriti_migration_target", "dry_run": true}' | python3 -m json.tool
ls /tmp/smriti_migration_target 2>&1
```

Expected:
- The API returns `success: true` and `dry_run: true`.
- The response reports files and bytes that would be moved.
- `/tmp/smriti_migration_target` does not exist after dry run.

### Test 6.3: Data Migration Live Run

```bash
curl -s -X POST http://127.0.0.1:7777/v1/smriti/storage/migrate \
  -H "Content-Type: application/json" \
  -d '{"target_data_dir": "/tmp/smriti_migration_live", "dry_run": false}' | python3 -m json.tool
ls /tmp/smriti_migration_live/
ls ~/Desktop/smriti_migrate_test/smriti/
curl -s http://127.0.0.1:7777/v1/smriti/storage | python3 -m json.tool
```

Expected:
- The migration reports success and moves files.
- The target directory is created.
- Source data remains intact after migration.
- Runtime storage config points to the new data directory.

### Test 6.4: Setu-2 Feedback Loop

```bash
MEDIA_ID=$(curl -s -X POST http://127.0.0.1:7777/v1/smriti/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "person", "top_k": 1}' | \
  python3 -c "import json,sys; r=json.load(sys.stdin); print(r['results'][0]['media_id'] if r['results'] else 'NONE')")

echo "Media ID: $MEDIA_ID"

if [ "$MEDIA_ID" != "NONE" ]; then
  curl -s -X POST http://127.0.0.1:7777/v1/smriti/recall/feedback \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"person\", \"media_id\": \"$MEDIA_ID\", \"confirmed\": true, \"session_id\": \"manual_test\"}" | python3 -m json.tool
fi
```

Expected:
- Valid media feedback returns `updated: true` and a non-zero `w_mean`.
- Missing media feedback returns `updated: false` without an HTTP error.

### Test 6.5: Mandala Worker

Check in the desktop/browser UI:
- Mandala opens without freezing the main thread.
- Cluster positions animate in and respond to hover/click.
- Zoom controls work.
- Keyboard navigation works.
- If the corpus is empty, the empty-state message is visible.

Expected:
- A `mandala-force-worker` worker appears in browser devtools when clusters exist.
- No WebGL dependency is required.

### Test 6.6: Deepdive Interactive Patches

Check in the desktop/browser UI:
- Open a recall result in Deepdive.
- Press `E` to toggle the JEPA patch overlay.
- Click a patch and inspect the detail popover.
- Press `F` to toggle fullscreen.
- Press `Escape` to close.

Expected:
- Focus moves into the modal on open and returns to the trigger on close.
- Patch details show anchor, stratum, confidence, and hallucination risk when available.
- Semantic neighbors render when the corpus has related media.

### Test 6.7: Person Journal Co-Occurrence Graph

Check in the desktop/browser UI:
- Load a person journal for a name that exists in indexed media.
- Verify the journal still shows the timeline.
- Verify a co-occurrence graph renders when atlas data is present.

Expected:
- The graph does not crash or flicker.
- No empty white canvas appears when there is no atlas data.

### Test 6.8: Accessibility Verification

Check in the desktop/browser UI:
- Keyboard-only navigation reaches Smriti sections, recall results, and Deepdive controls.
- Focus rings are visible.
- Live status changes are announced to assistive tech.
- Reduced-motion behavior removes non-essential animation when the system setting is enabled.

Expected:
- Canvas elements are `aria-hidden` where appropriate.
- Modal focus is trapped and restored correctly.

### Test 6.9: Full Regression

```bash
pytest -q cloud/api/tests cloud/jepa_service/tests cloud/search_service/tests cloud/monitoring/tests tests/test_readme.py
cd /Users/macuser/toori/desktop/electron && npm run typecheck && npm run build
cat /Users/macuser/toori/.smriti_build/state.json | python3 -m json.tool
```

Expected:
- Python suite passes with at least `195` tests.
- Desktop typecheck and build both succeed.
- `.smriti_build/state.json` reports Sprint 5 completion honestly.

## Troubleshooting: Sprint 5

- If `mandala-force-worker.ts` is missing from devtools, verify the file exists and the worker import path is correct.
- If Deepdive focus does not return to the opener, verify the cleanup path restores the previously focused element.
- If migration returns 422, verify the request uses `target_data_dir`.
- If feedback returns `updated: false` for valid media, verify the media has an embedding available through Smriti or observations.
- If the journal graph is blank, inspect the `atlas` payload in the journal response.
