"""Prepare a reproducible image dataset by copying selected JPG files."""
from pathlib import Path
import argparse, json, shutil, yaml

def main(config: Path):
    cfg=yaml.safe_load(config.read_text(encoding='utf-8')); p=cfg['paths']; limit=cfg['training'].get('max_train_per_class'); raw=Path(p['raw']); out=Path(p['processed'])
    if out.exists(): shutil.rmtree(out)
    counts={}
    for split in ('train','valid','test'):
        classes=sorted(x for x in (raw/split).iterdir() if x.is_dir())
        for cls in classes:
            files=sorted(cls.glob('*.jpg'))
            if split=='train' and limit: files=files[:int(limit)]
            dest=out/split/cls.name; dest.mkdir(parents=True,exist_ok=True)
            for f in files: shutil.copy2(f,dest/f.name)
            counts[f'{split}/{cls.name}']=len(files)
    (out/'dataset_stats.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    train_classes = {k.split('/', 1)[1] for k in counts if k.startswith('train/')}
    print(json.dumps({'processed': str(out), 'images': sum(counts.values()), 'classes': len(train_classes)}))
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--config',type=Path,default=Path('params.yaml')); main(a.parse_args().config)
