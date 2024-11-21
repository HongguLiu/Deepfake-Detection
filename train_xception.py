import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import wandb
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from collections import Counter
from torchvision import transforms
from network.models import model_selection
from dataset.mydataset import MyDataset

print("Script started")

# µ¥ÀÌÅÍ º¯È¯ ¼³Á¤
xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
        transforms.RandomResizedCrop(size=(299, 299), scale=(0.7, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}


# Å¬·¡½º °¡ÁßÄ¡ °è»ê
def calculate_class_weights(dataset):
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    total_count = len(labels)
    class_weights = [total_count / class_counts[i] for i in range(len(class_counts))]
    return torch.FloatTensor(class_weights).cuda()


# ¸ÞÀÎ ÇÔ¼ö
def main():
    args = parse.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ¼³Á¤ °¡Á®¿À±â
    name = args.name
    train_list = args.train_list
    epoches = args.epoches
    batch_size = args.batch_size
    dropout = args.dropout
    learning_rate = args.learning_rate
    step_size = args.step_size
    gamma = args.gamma
    k_folds = args.k_folds
    model_path = args.model_path
    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wandb.init(project="Deepfake-Xception", name=name)
    wandb.config.update(args)

    # µ¥ÀÌÅÍ¼Â ÁØºñ
    full_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'])
    class_weights = calculate_class_weights(full_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ¸ðµ¨ ÃÊ±âÈ­
    model = model_selection(modelname='xception', num_out_classes=2, dropout=dropout).to(device)
    model = nn.DataParallel(model)

    # Æ¯Á¤ ·¹ÀÌ¾î µ¿°á
    for name, param in model.named_parameters():
        if "conv1" in name or "conv2" in name:
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scaler = torch.cuda.amp.GradScaler()

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_val_acc = 0.0

    print("Initializing KFold")

    # K-Fold ±³Â÷ °ËÁõ
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f'FOLD {fold + 1}/{k_folds}')
        print('-' * 40)

        train_subsampler = Subset(full_dataset, train_idx)
        val_subsampler = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                drop_last=True)

        print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")

        # ¿¡Æø ÇÐ½À
        for epoch in range(epoches):
            print(f'Epoch {epoch + 1}/{epoches}')
            print('-' * 20)

            model.train()
            train_loss, train_corrects = 0.0, 0.0
            all_labels, all_preds = [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                _, preds = torch.max(outputs, 1)
                train_loss += loss.item()
                train_corrects += torch.sum(preds == labels).item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            train_loss /= len(train_subsampler)
            train_acc = train_corrects / len(train_subsampler)
            train_f1 = f1_score(all_labels, all_preds, average='macro')
            print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
            wandb.log({"Train Loss": train_loss, "Train Acc": train_acc, "Train F1": train_f1, "Epoch": epoch + 1})

            # °ËÁõ ´Ü°è
            model.eval()
            val_loss, val_corrects = 0.0, 0.0
            all_val_labels, all_val_preds = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.amp.autocast(device_type='cuda'):
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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = model.state_dict()

            scheduler.step()

        model.load_state_dict(best_model_wts)
        torch.save(model.module.state_dict(), os.path.join(output_path, f"best_model_fold_{fold + 1}.pkl"))
        print(f'Saved best model of fold {fold + 1} with accuracy: {best_val_acc:.4f}')

    print(f'Best Validation Accuracy across all folds: {best_val_acc:.4f}')
    wandb.finish()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='deepfake_xception_last1117')
    parse.add_argument('--train_list', '-tl', type=str, default='./last2_labels.txt')
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--epoches', '-e', type=int, default=100)
    parse.add_argument('--model_name', '-mn', type=str, default='xception_model.pkl')
    parse.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
    parse.add_argument('--learning_rate', '-lr', type=float, default=0.00001, help="Learning rate for optimizer")
    parse.add_argument('--step_size', type=int, default=5, help="Step size for LR scheduler")
    parse.add_argument('--gamma', type=float, default=0.5, help="Gamma value for LR scheduler")
    parse.add_argument('--k_folds', type=int, default=5, help="Number of folds for KFold cross-validation")
    parse.add_argument('--model_path', '-mp', type=str, default='/home/hail/Documents/jjw/output/deepfake_xception_last3/best_model_fold_5.pkl',
                       help="Path to model weights to load if continuing training")
    main()
    print("Script finished")
