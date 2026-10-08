"""Reproduce the fixed official IPN Hand archive subset (~1 GB compressed).
Never extracts incomplete videos. Child downloads stop after eight complete videos.
"""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.extract import extract_complete,run
IDS=['1HylyDnApIRNMloREqvKYWWhB88q7Z1Cg','1tqR2FF8OlXGmYACw3TSxyS6QGDit_s3a','1Dk7l-jAAvNLlb0faypAco88XYLWPa9g_','1x0mDr-QHQtDkfcQAm9bKdWBR7Lj3fLcV','1PCjldH6hmVYV7EObPf-jtNWCtDK60o6n']

def main():
    import gdown
    base=ROOT/'research/data';archives=base/'archives';archives.mkdir(parents=True,exist_ok=True)
    annotations=base/'annotations/annotations'
    if not (annotations/'Annot_List.txt').exists():
        gdown.download_folder(id='1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD',output=str(base/'annotations')+'/',use_cookies=False)
    workers={}
    for i,id in enumerate(IDS,1):
        if (archives/f'videos0{i}.subset.json').exists():continue
        output=archives/f'videos0{i}.tgz'
        workers[i]=subprocess.Popen([sys.executable,'-m','gdown',id,'--no-cookies','-q','-O',str(output)],cwd=ROOT)
    try:
        while workers:
            for path in sorted(archives.glob('*.part'))+sorted(archives.glob('*.tgz')):extract_complete(path,base/'videos')
            for i,worker in list(workers.items()):
                if (archives/f'videos0{i}.subset.json').exists():
                    worker.terminate();worker.wait(timeout=10);del workers[i];print('Eight complete videos acquired from archive',i,flush=True)
                elif worker.poll() is not None:raise RuntimeError(f'Archive {i} failed or ended before subset was complete')
            if workers:time.sleep(20)
    finally:
        for worker in workers.values():worker.terminate()
    run()
    provenance=dict(dataset='IPN Hand',project_url='https://gibranbenitez.github.io/IPN_Hand/',license='CC BY 4.0',video_folder='https://drive.google.com/drive/folders/1O4Fn_jbAEcKIksXHmQIMcHTyaDw4wt9x',annotation_folder='https://drive.google.com/drive/folders/1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD',archive_ids=IDS,selection='First eight complete video members from each of five original archives, selected before recognition; archives intentionally partial',grouping='First two underscore-separated filename components; official metadata has 50 distinct keys with four videos each')
    (base/'provenance.json').write_text(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
