import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import wandb
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torchvision import transforms

from network.models import model_selection
from dataset.mydataset import MyDataset

def calculate_class_weights(dataset):
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    total_count = len(labels)
    class_weights = [total_count / class_counts[i] for i in range(len(class_counts))]
    return torch.FloatTensor(class_weights).cuda()

def main():
    args = parse.parse_args()
    name = args.name
    train_list = args.train_list
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wandb.init(project="Deepfake-Xception", name='train_Xception_with_kfold')
    wandb.config.update(args)

    # Data augmentation for real class
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    # Datasets with augmentation for real class
    full_dataset = MyDataset(txt_path=train_list,
                             transform=xception_default_data_transforms['train'],
                             real_transform=augmentation_transforms)

    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(full_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_val_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f'FOLD {fold + 1}/{k_folds}')
        print('-' * 20)

        # Prepare data loaders for this fold
        train_subsampler = Subset(full_dataset, train_idx)
        val_subsampler = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        model = nn.DataParallel(model)

        best_model_wts = model.state_dict()
        for epoch in range(epoches):
            print(f'Epoch {epoch + 1}/{epoches}')
            print('-' * 10)

            model.train()
            train_loss, train_corrects = 0.0, 0.0
            all_labels, all_preds = [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_corrects += torch.sum(preds == labels).item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            train_loss /= len(train_subsampler)
            train_acc = train_corrects / len(train_subsampler)
            train_f1 = f1_score(all_labels, all_preds, average='macro')

            print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
            wandb.log({"Train Loss": train_loss, "Train Acc": train_acc, "Train F1": train_f1, "Epoch": epoch + 1})

            # Validation phase
            model.eval()
            val_loss, val_corrects = 0.0, 0.0
            all_val_labels, all_val_preds = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_corrects += torch.sum(preds == labels).item()
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(preds.cpu().numpy())

            val_loss /= len(val_subsampler)
            val_acc = val_corrects / len(val_subsampler)
            val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')

            print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
            wandb.log({"Val Loss": val_loss, "Val Acc": val_acc, "Val F1": val_f1, "Epoch": epoch + 1})

            # Save best model in the current fold
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = model.state_dict()

            scheduler.step()

        # Load the best model weights for this fold
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), os.path.join(output_path, f"best_model_fold_{fold + 1}.pth"))
        print(f'Saved best model of fold {fold + 1} with accuracy: {best_val_acc:.4f}')

    print(f'Best Validation Accuracy across all folds: {best_val_acc:.4f}')
    wandb.finish()

if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='deepfake_xception_kfold')
    parse.add_argument('--train_list', '-tl', type=str, default='./labels2.txt')
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--epoches', '-e', type=int, default=50)
    parse.add_argument('--model_name', '-mn', type=str, default='xception_model.pth')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/ffpp_c23.pkl')
    main()
