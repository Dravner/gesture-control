# Standalone desktop build — 2026-10-09

Delivered app: `/Users/a1111/schcool/diplom/gesture-control/dist/Жестовое управление.app`.

- Native Apple Silicon arm64, built on macOS 15.6.1 using Python 3.14.3, PyInstaller 6.22.3, Qt/PySide6 Essentials 6.11.2.
- Bundle occupies approximately 661 MiB on disk; regular-file payload 686,598,088 bytes before the final minimum-OS metadata correction.
- Minimum supported metadata: **macOS 15.0**. Native audit found Homebrew libmpdec/libsqlite3 and shiboken6 built with `minos 15.0`; compatibility with macOS 14 is therefore not claimed. Tested OS: 15.6.1. Intel and universal2 builds are not provided.
- Stable identifier `org.dravner.gesturecontrol`, Russian display name, original locally generated hand icon, camera purpose text and high-resolution capability in Info.plist.
- `codesign --verify --deep --strict` passes. Signature is ad-hoc, without Developer ID, team identity or notarization. No Gatekeeper/notarization claim or TCC bypass is made.

The bundle includes its own native executable, Python.framework, Qt libraries/plugins, NumPy/OpenCV, PyObjC wrappers for system Vision/Quartz/Foundation, MediaPipe Tasks shared library, PyTorch and optional MLP/temporal inference. It requires neither the source checkout nor `.venv` nor Python.app on the destination Mac. `otool -L` of the entry executable refers only to macOS system libraries/frameworks.

Exactly nine allowlisted model/metadata assets are included: 8,645,203 bytes in total, with individual SHA-256 hashes recorded in `build-manifest.json`. Project gesture profiles, settings, calibration, session recordings, research data and the large external/reference trees are excluded. Model provenance is recorded in `packaging/MODEL_PROVENANCE.md`, which is also included in the app resources.

At first normal GUI launch, only missing built-in model files are copied from read-only app resources to `~/Library/Application Support/GestureControl/models`. The GUI receives `~/Library/Application Support/GestureControl` as its writable root. Existing models, profiles and settings are preserved. Apple Vision is the default for a new macOS user root; later saved tracker choices persist. Camera capture and OS input do not start automatically.

## Verification

- `tests/test_app_paths.py`: **3 passed**, verifying missing-resource errors, frozen resource path selection, explicit temporary home paths, preservation of custom files and exclusion of project private data.
- Source and frozen native diagnostics: **PASS**, importing Qt, Apple Vision/Quartz/Foundation, MediaPipe, OpenCV, NumPy and Torch; running MediaPipe inference on a synthetic blank image; loading/running finite MLP and joint/two_stage/window temporal predictions.
- These probes use temporary roots/caches; they do not open a camera, send OS events, run GUI, or create home-directory profiles. `frozen-diagnostics.json` records `camera_opened=false` and `os_events_posted=false`.
- `resource-audit.json`: exactly nine allowed assets, bundled Python.framework, no project profiles/external/research directories.
- `native-compatibility.json`: deployment versions from 375 native Mach-O files.
- Final strict code signature verification passes after correcting minimum OS metadata to 15.0 and re-signing. The build spec also declares 15.0 for future builds.

GUI/camera/privacy testing is handled separately by the main task; the package agent has not opened the GUI or camera.

## Build warnings and resolved environment issues

The first build reached binary collection but failed because PyInstaller tried to write its cache under home Application Support, outside the sandbox. The reproducible script now sets `PYINSTALLER_CONFIG_DIR=build/pyinstaller-cache` and `MPLCONFIGDIR=build/matplotlib-cache`; the clean final build succeeds. The initial log is retained only as diagnostic evidence.

Sandboxed MediaPipe construction aborted because its Metal/GL service was unavailable. The same source and frozen native model probes pass outside the sandbox with the normal Apple graphics context. This is an execution-environment limitation, not a missing bundled model/library.

PyInstaller emits warnings for optional TensorBoard, pycparser generated tables, legacy scipy `_cdflib`, Linux/Windows/CUDA/HIP ctypes libraries, and the absolute system ApplicationServices path. The application uses the installed macOS ApplicationServices framework and CPU inference. Native probes pass; these optional cross-platform references are not advertised features. MediaPipe emits standard feedback-tensor and square-ROI warnings on the synthetic blank-image probe.

Rebuild: `.venv/bin/python tools/build_app.py`. The script prepares the icon and resource manifest, records installed dependency versions, performs a clean native build and verifies its signature. Official build references are linked in `packaging/README.md`.

## Native UI and live camera smoke

The frozen app was launched through the native UI. All three settings tabs and custom-master entry were visible. Cursor sensitivity was changed to1.20 using the locale comma separator, saved and reopened with the value preserved; numerical defaults were then restored. OS input remained off. Camera permission was confirmed enabled in System Settings; after the first permission request, a second start delivered1185packets and the GUI remained responsive. Capture/result/display aggregates are in `live-smoke-summary.json`: median27.23ms,p95 33.93ms capture-to-display, measured after capture.read returns. This excludes sensor exposure/driver time and OS command latency. The unlabelled session included observed pointer/drag commands, but no recognition accuracy is inferred. Camera was stopped, buttons released. No raw personal session, coordinates or images are included in this report/Git.
