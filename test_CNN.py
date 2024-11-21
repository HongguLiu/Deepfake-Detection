import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
def main():
	args = parse.parse_args()
	test_list = args.test_list
	batch_size = args.batch_size
	model_path = args.model_path
	torch.backends.cudnn.benchmark=True
	test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
	test_dataset_size = len(test_dataset)
	corrects = 0
	acc = 0
	#model = torchvision.models.densenet121(num_classes=2)
	model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	model.load_state_dict(torch.load(model_path),strict = True)
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model = model.cuda()
	model.eval()
	with torch.no_grad():
		for (image, labels) in test_loader:
			image = image.cuda()
			labels = labels.cuda()
			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)
			corrects += torch.sum(preds == labels.data).to(torch.float32)
			print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
		acc = corrects / test_dataset_size
		print('Test Acc: {:.4f}'.format(acc))



if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--batch_size', '-bz', type=int, default=32)
	parse.add_argument('--test_list', '-tl', type=str, default='./last_labels.txt')
	parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/ffpp_c23.pth')
	main()
	print('Hello world!!!')