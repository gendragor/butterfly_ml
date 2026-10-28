"""Train three configurable CNN classifiers on CPU/GPU."""
from pathlib import Path
import argparse, json, random, time, yaml, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# CPU-friendly training: no pretrained downloads are required, so the project is reproducible offline.

def build(name, classes):
    if name=='simplecnn':
        return nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,classes))
    if name=='mobilenet_v3_small':
        m=models.mobilenet_v3_small(weights=None); m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,classes); return m
    if name=='resnet18':
        m=models.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,classes); return m
    raise ValueError(name)

def main(config: Path):
    c=yaml.safe_load(config.read_text(encoding='utf-8')); t=c['training']; root=Path(c['paths']['augmented'] if t['use_augmentation'] else c['paths']['processed']); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); random.seed(c['seed']); torch.manual_seed(c['seed'])
    tf=transforms.Compose([transforms.Resize((t['image_size'],t['image_size'])),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); train=datasets.ImageFolder(root/'train',tf); valid=datasets.ImageFolder(root/'valid',tf); test=datasets.ImageFolder(root/'test',tf); classes=len(train.classes); out=Path(c['paths']['models']); out.mkdir(parents=True,exist_ok=True); report=Path(c['paths']['reports']); report.mkdir(parents=True,exist_ok=True)
    metrics={}
    for name,opt in c['models'].items():
        if not opt.get('enabled',True): continue
        model=build(name,classes).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate']); loss_fn=nn.CrossEntropyLoss(); best=0.; stale=0; history=[]; start=time.time()
        for epoch in range(1,t['epochs']+1):
            model.train(); correct=total=0
            for x,y in DataLoader(train,batch_size=t['batch_size'],shuffle=True,num_workers=0):
                x,y=x.to(device),y.to(device); optimizer.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); optimizer.step(); correct+=(logits.argmax(1)==y).sum().item(); total+=len(y)
            model.eval(); vc=vt=0
            with torch.no_grad():
                for x,y in DataLoader(valid,batch_size=t['batch_size'],num_workers=0): vc+=(model(x.to(device)).argmax(1)==y.to(device)).sum().item(); vt+=len(y)
            va=vc/vt; history.append({'epoch':epoch,'train_accuracy':correct/total,'valid_accuracy':va})
            if va>best: best=va; stale=0; torch.save({'model':model.state_dict(),'classes':train.classes,'epoch':epoch},out/f'{name}.pt')
            else: stale+=1
            if stale>=t['patience']: break
        metrics[name]={'best_valid_accuracy':best,'epochs_trained':len(history),'seconds':round(time.time()-start,1),'history':history}
    report.joinpath('metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--config',type=Path,default=Path('params.yaml')); main(a.parse_args().config)
