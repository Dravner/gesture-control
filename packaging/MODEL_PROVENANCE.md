# Included model provenance

Only the nine allowlisted files from `models/` are included. A build manifest records exact byte counts and SHA-256 hashes. No gesture profiles, session recordings, private calibration, research datasets, videos, annotations, notebooks or full reference repositories are bundled.

- `hand_landmarker.task`: Google MediaPipe Hand Landmarker model asset. Official model documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker . Official distribution URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task . Existing project asset is bundled; this build does not redownload or replace it. MediaPipe framework is Apache-2.0; model terms should be checked separately before public redistribution.
- `mlp_hagrid_6classes.pth`, `mlp_hagrid_2d_no_gesture.pth`, `mlp_hagrid_subsample.pth`: existing locally trained compact project checkpoints based on public HaGRID landmark datasets; see project README and training reports. HaGRID source: https://github.com/hukenovs/hagrid . No source images are bundled.
- `streaming_joint.pth`, `streaming_selected.pth`, `streaming_two_stage.pth`, `streaming_window.pth`: existing project temporal checkpoints trained/evaluated with public IPN Hand streams; source: https://github.com/GibranBenitez/IPN-hand . No source videos are bundled.
- `selection.json`: existing project method selection metadata.

These notes describe provenance, not a new license grant. Third-party Python package licenses remain within their installed package metadata where collected by PyInstaller. Apple Vision/Quartz use system frameworks through PyObjC wrappers, rather than redistributed Apple framework binaries.
