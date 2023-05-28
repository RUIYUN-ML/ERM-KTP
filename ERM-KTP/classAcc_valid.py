import torch
import torch.nn as nn
from dataLoader import DataLoader
from model.resnet_20 import resnet20
from model.resnet_50 import resnet50
from model.resnext import resnext50
from parser_init import *
from torch.autograd import Variable

args = parser_init()

def classACC_valid(model, dataloaders, num_classes, args):
  model.eval()
  if args.ifmask:
    # L1-density
    #print(model.module.lmask.get_density())

    #Cosine similarity
    matrix = model.module.lmask.mask.transpose(0,1)
    csi = 0
    for idx in range(matrix.size(0)):
      x = matrix[idx].view(1,-1)
      for idy in range(matrix.size(0)):
        if idx != idy:
          y = matrix[idy].view(1,-1)
          csi += torch.cosine_similarity(x, y, dim=-1)
    #print(csi)

  remain_train_acc = 0
  unlearn_train_acc = 0
  remain_val_acc = 0
  unlearn_val_acc = 0

  with torch.no_grad():
    for phase in ['train', 'val']:
      correct = list(0. for i in range(num_classes))
      total = list(0. for i in range(num_classes))
      for iteration, data in enumerate(dataloaders[phase]):
          
          images, labels = data
          images = Variable(images.cuda())
          labels = Variable(labels.cuda())

          output = model(images)

          prediction = torch.argmax(output, 1)
          res = (prediction == labels)
          for label_idx in range(len(labels)):
              label_single = labels[label_idx]
              correct[label_single] += res[label_idx].item()
              total[label_single] += 1
      acc_str = 'Accuracy: %f\n'%(sum(correct)/sum(total))

      for acc_idx in range(len(correct)):
            try:
              acc = correct[acc_idx]/total[acc_idx]
            except:
              acc = 0
            finally:
              acc_str += '\tclassID:%d\tacc:%f\t \n'%(acc_idx+1, acc)

      if phase == 'train':
        remain_train_acc = (sum(correct[args.num_unlearn:]))/(sum(total[args.num_unlearn:]))
        unlearn_train_acc = (sum(correct[0:args.num_unlearn]))/(sum(total[0:args.num_unlearn]))
      elif phase == 'val':
        remain_val_acc = (sum(correct[args.num_unlearn:]))/(sum(total[args.num_unlearn:]))
        unlearn_val_acc = (sum(correct[0:args.num_unlearn]))/(sum(total[0:args.num_unlearn]))
  
  return remain_train_acc, unlearn_train_acc, remain_val_acc, unlearn_val_acc




if __name__ == '__main__':

  if args.dataset == 'cifar-10':
    num_classes = 10
  elif args.dataset =='cifar-100':
    num_classes = 100
  elif args.dataset =='imagenet':
    num_classes = 200


  checkpoint_path = args.load_checkpoint

  if args.model == 'resnet50':
      model = resnet50(num_classes=num_classes, ifmask=args.ifmask, pretrained=False)
  if args.model == 'resnet20':
      model = resnet20(ifmask=args.ifmask, num_classes=num_classes)
  elif args.model == 'resnext50':
      model = resnext50(num_classes=num_classes, ifmask=args.ifmask)

  model = torch.nn.DataParallel(model)
  checkpoint = torch.load(checkpoint_path)

  ori_checkpoint = torch.load(checkpoint_path)

  model.load_state_dict(checkpoint['model_state_dict'])
  model = model.cuda()

  loader = DataLoader(args.dataset,batch_size=args.batch_size, seed=args.seed)
  dataloaders, dataset_sizes = loader.load_data()

  criterion = nn.CrossEntropyLoss()
  print('epoch ' + str(checkpoint['epoch']))

  print(classACC_valid(model, dataloaders, num_classes, args))