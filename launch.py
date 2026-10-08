"""Launch the native UI; replay is always restricted to the event log."""
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description='Потоковое жестовое управление')
    parser.add_argument('--video', type=Path, help='Видеофайл для безопасного воспроизведения')
    parser.add_argument('--screenshot', type=Path, help='Сохранить снимок интерфейса без камеры')
    parser.add_argument('--tracker',choices=['mediapipe','apple_vision'],help='Выбрать трекер при запуске; камера и OS-ввод автоматически не включаются')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    from gesture_system.gui import run
    return run(Path(__file__).resolve().parent, args.video, args.screenshot,tracker_backend=args.tracker)


if __name__ == '__main__':
    raise SystemExit(main())
