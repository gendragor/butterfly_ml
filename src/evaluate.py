"""Evaluate saved neural-network checkpoints on the untouched test split."""
from pathlib import Path
import sys
_VENDOR_NN = Path(__file__).resolve().parents[1] / 'vendor_nn'
if _VENDOR_NN.exists(): sys.path.insert(0, str(_VENDOR_NN))
import argparse, json, yaml, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def build(name, classes):
    if name == 'simplecnn':
        return nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,classes))
    if name == 'mobilenet_v3_small':
        m=models.mobilenet_v3_small(weights=None); m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,classes); return m
    m=models.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,classes); return m

def main(config: Path):
    c=yaml.safe_load(config.read_text(encoding='utf-8')); t=c['training']; root=Path(c['paths']['augmented'] if t['use_augmentation'] else c['paths']['processed']); tf=transforms.Compose([transforms.Resize((t['image_size'],t['image_size'])),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); ds=datasets.ImageFolder(root/'test',tf); out=Path(c['paths']['models']); results={}
    for name,opt in c['models'].items():
        if not opt.get('enabled',True): continue
        ckpt=torch.load(out/f'{name}.pt',map_location='cpu',weights_only=False); model=build(name,len(ds.classes)); model.load_state_dict(ckpt['model']); model.eval(); correct=total=0
        with torch.no_grad():
            for x,y in DataLoader(ds,batch_size=t['batch_size'],num_workers=0): correct+=(model(x).argmax(1)==y).sum().item(); total+=len(y)
        results[name]={'test_accuracy':correct/total,'test_samples':total,'checkpoint_epoch':ckpt['epoch']}
    report=Path(c['paths']['reports']); report.mkdir(parents=True,exist_ok=True); (report/'test_metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8'); print(json.dumps(results,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--config',type=Path,default=Path('params.yaml')); main(a.parse_args().config)
