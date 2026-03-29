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
