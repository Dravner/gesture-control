# macOS desktop build

Build on an Apple Silicon Mac with the project `.venv` and requirements installed:

```sh
.venv/bin/python -m pip install --index-url https://pypi.org/simple pyinstaller==6.22.3
.venv/bin/python -m pytest tests/test_app_paths.py -q
.venv/bin/python tools/build_app.py
```

Output: `dist/Жестовое управление.app`. The native `Contents/MacOS/GestureControl` executable embeds the Python runtime; it does not launch Python.app, use a shell wrapper, or require this checkout/venv on the destination computer. It packages PySide6 Essentials, MediaPipe, OpenCV, NumPy, PyTorch for optional checkpoint modes, and PyObjC bindings for the system Apple Vision/Quartz/Foundation frameworks.

The current build targets arm64 with macOS 15.0 minimum metadata. Actual compatibility must be tested on the target OS; the build report records the host macOS and all dependency versions. Intel Macs need a separate native x86_64 environment and adjusted target architecture. Universal2 is not claimed.

Bundled models remain in read-only app resources. On first launch only missing allowlisted model files are copied to `~/Library/Application Support/GestureControl/models`. Profiles, settings, calibration, session recordings and other mutable files are created under that user application directory. Existing files are never replaced by the bootstrap. Project user data is not distributed. Copying this app to another directory does not change the user data location.

Camera access and system input are off at startup. macOS privacy authorization is obtained by the normal OS permission flow when the user enables the corresponding function. This build neither bypasses nor resets TCC. The stable bundle identifier is `org.dravner.gesturecontrol`; the camera purpose is declared in Info.plist.

PyInstaller applies ad-hoc signing by default. `GESTURE_CODESIGN_IDENTITY` optionally supplies an already available signing identity for builds. The build checks `codesign --verify --deep --strict`. Ad-hoc signing is not Developer ID signing or notarization; public distribution requires its own Developer ID/notarization process.

A camera-free dependency/model probe is available:

```sh
'dist/Жестовое управление.app/Contents/MacOS/GestureControl' --diagnostics
```

It uses a temporary directory, imports the native dependencies, opens the bundled Hand Landmarker, and loads the MLP and streaming checkpoints. It does not start camera capture, launch GUI, send OS input, or write profiles to the home directory. Actual camera/permission behavior remains a manual test.

`reports/desktop-package-20261009/` contains the build environment, resource hashes, PyInstaller log, and code signature details.

Official references:
- https://pyinstaller.org/en/stable/feature-notes.html#macos-multi-arch-support
- https://pyinstaller.org/en/stable/usage.html
- https://pyinstaller.org/en/stable/runtime-information.html
