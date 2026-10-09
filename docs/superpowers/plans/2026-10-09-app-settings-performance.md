# Application settings and performance implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. The document task and packaging have separate file scopes.

**Goal:** Preserve the user-validated gesture controller, provide a standalone macOS GUI with persistent tuning and actions, and reduce avoidable scheduling/startup latency with measured safety checks.

**Architecture:** Existing camera/latest-frame/tracker/controller/Quartz separation remains. Typed interaction settings and a dedicated settings dialog keep tuning out of the monolithic main UI. Coalesced queued Qt notification consumes completed packets promptly; a slow timer remains a health watchdog. Frozen bundle assets are separate from persistent user data.

**Tech Stack:** Python3.14, PySide6, Apple Vision/PyObjC, optional MediaPipe/PyTorch, PyInstaller arm64. Native RGB resolution stays the default because downsample evidence is only a proxy on public640×480 footage.

## Global constraints

- Baseline6021c2c remains recoverable and pushed to origin/dev; no user session/profile data in Git.
- No extrapolated cursor on tracking loss; no OS actions from camera/tracker workers; settings changes release input.
- Defaults preserve currently accepted sensitivity/scroll/pinch semantics. Faster reacquisition must have independent jitter/gap/false-switch tests.
- Article18–22pages uses methodical formatting and explicitly separates proxies, synthetic tests and one-person qualitative acceptance.

## Tasks

- [ ] Typed InteractionSettings with bounded finite values, atomic persistence, defaults, compatible existing preferences.
- [ ] Dedicated Qt settings tabs for cursor/smoothing, scroll and gesture timings/amplitude, safe action mappings, custom enrollment entry.
- [ ] Lazy optional Torch/temporal loading in GUI; preserve explicit legacy Engine API and experimental inference.
- [ ] Coalesced latest-packet GUI notification and health timer; stalled callbacks and close races cannot post input.
- [ ] Motion-compatible relative pointer acquisition with short pose confirmation; explicit scroll/navigation handoff; synthetic frame-rate and identity/gap tests.
- [ ] Standalone .app resources/userdata paths and build, native GUI QA, import/permission disclosure if new identity requires access.
- [ ] New latency/timing/false-switch experiments, full tests, native camera review, software documentation, second commit/push.
- [ ] Integrate final evidence into article, render and inspect every page, deliver~20pageDOCX.
