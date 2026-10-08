"""Summarize recorded live TRAIN/TRIAL without replaying or changing profiles."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(path):
    path = Path(path)
    metadata_path, frames_path = path / 'session.json', path / 'frames.npz'
    metadata = json.loads(metadata_path.read_text())
    frames = np.load(frames_path, allow_pickle=False)
    tasks = metadata.get('tasks', [])
    trials = [task for task in tasks if task['mode'] == 'trial' and task['status'] != 'cancelled']
    groups = {}
    for task in trials:
        result = groups.setdefault(task['expected_id'], {
            'name': task['name'], 'attempts': 0, 'recognized_attempts': 0,
            'target_events': 0, 'duplicate_events': 0, 'other_events': 0, 'tasks': []})
        count = int(task['matched_count'])
        result['attempts'] += 1
        result['recognized_attempts'] += int(count > 0)
        result['target_events'] += count
        result['duplicate_events'] += max(count - 1, 0)
        result['other_events'] += sum(event['gesture'] != task['expected_id'] for event in task['detections'])
        result['tasks'].append(task)
    timing = {}
    for key in ('inference_ms', 'capture_to_result_ms', 'capture_to_display_ms'):
        if key in frames.files:
            values = frames[key]
            values = values[np.isfinite(values)]
            if len(values):
                timing[key] = {'n': len(values), 'p50': float(np.median(values)),
                               'p95': float(np.percentile(values, 95))}
    return {
        'scope': 'Recorded online decisions, not replay. Trial windows are user-prompted; '
                 'attempt recognition is not unsegmented event F1 or usability. '
                 'No background false-positive rate without independent background annotation. '
                 'Timing excludes camera exposure and OS response.',
        'session': str(path.resolve()),
        'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (metadata_path, frames_path)},
        'source': metadata['source'], 'settings': metadata['settings'],
        'frames': len(frames['timestamps']), 'hand_frames': int(frames['present'].sum()),
        'world_frames': int(frames['world_present'].sum()) if 'world_present' in frames.files else 0,
        'train_tasks': [task for task in tasks if task['mode'] == 'train'],
        'trials': groups, 'timing': timing,
        'executed_os_events': sum(bool(event['executed']) for event in metadata['events']),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.session)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('train_tasks', 'trials')}, ensure_ascii=False, indent=2))
