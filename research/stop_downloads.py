"""Stop only our official archive downloaders after eight complete RGB members."""
import subprocess,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
IDS=['1HylyDnApIRNMloREqvKYWWhB88q7Z1Cg','1tqR2FF8OlXGmYACw3TSxyS6QGDit_s3a','1Dk7l-jAAvNLlb0faypAco88XYLWPa9g_','1x0mDr-QHQtDkfcQAm9bKdWBR7Lj3fLcV','1PCjldH6hmVYV7EObPf-jtNWCtDK60o6n']
archive_dir=ROOT/'research/data/archives';stopped=set()
while len(stopped)<5:
    for i,id in enumerate(IDS,1):
        marker=archive_dir/f'videos0{i}.subset.json'
        if marker.exists() and i not in stopped:
            subprocess.run(['pkill','-f',f'[Pp]ython.* -m gdown {id}'],check=False)
            stopped.add(i);print('Stopped archive',i,flush=True)
    time.sleep(15)
