"""Create and persist one augmented copy of each training image."""
from pathlib import Path
import argparse, random, shutil, yaml
from PIL import Image, ImageEnhance, ImageOps

def main(config: Path):
    cfg=yaml.safe_load(config.read_text(encoding='utf-8')); src=Path(cfg['paths']['processed']); out=Path(cfg['paths']['augmented'])
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(src,out)
    rng=random.Random(cfg['seed'])
    for f in sorted((src/'train').glob('*/*.*')):
        with Image.open(f) as im:
            im=im.convert('RGB')
            if rng.random()<0.5: im=ImageOps.mirror(im)
            im=im.rotate(rng.uniform(-15,15),resample=Image.Resampling.BILINEAR)
            im=ImageEnhance.Brightness(im).enhance(rng.uniform(.85,1.15))
            dest=out/'train'/f.parent.name/(f.stem+'_aug.jpg'); dest.parent.mkdir(parents=True,exist_ok=True); im.save(dest,quality=90)
    print(f'augmented dataset saved to {out}')
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--config',type=Path,default=Path('params.yaml')); main(a.parse_args().config)
