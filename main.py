import argparse
import os
import csv
import shutil
import random
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as t
from sklearn.model_selection import train_test_split
from utils import conf as config
from models import XceptionWithCBAM



def save_checkpoint(path, state_dict, epoch=0, arch="", acc1=0):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if torch.is_tensor(v):
            v = v.cpu()
        new_state_dict[k] = v

    torch.save({
        "epoch": epoch,
        "arch": arch,
        "acc1": acc1,
        "state_dict": new_state_dict,
    }, path)



class SGDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list  # list of (image_path, label)
        self.transform = transform

        self.default_transform = t.Compose([
            t.ToTensor(),
            t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image_path, label = self.data_list[index]

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            img = Image.fromarray(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))

        if self.transform is not None:
            result = self.transform(image=np.array(img))
            img = result["image"]
        else:
            img = self.default_transform(img)

        return img, label

    def __len__(self):
        return len(self.data_list)


def load_all_data(data_root):
    data_list = []
    for label_name, label in [("0_real", 0), ("1_fake", 1)]:
        folder_path = os.path.join(data_root, label_name)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(folder_path, fname)
                data_list.append((fpath, label))
    return data_list


def split_dataset(data_root, train_ratio=0.25, test_ratio=0.75, transform=None, seed=42):
    assert abs(train_ratio +  test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    data = load_all_data(data_root)
    train_data, test_data = train_test_split(data, train_size=train_ratio, random_state=seed, stratify=[label for _, label in data])
    
    return (
        SGDataset(train_data, transform=transform),
        SGDataset(test_data, transform=transform)
    )

def main():
    torch.backends.cudnn.benchmark = True

    train_dataset, test_dataset = split_dataset(config.data_root, train_ratio=0.25, test_ratio=0.75)

    kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                  shuffle=True, pin_memory=True)
    
    if args.mode == "test":
            test_loader = DataLoader(test_dataset, **kwargs)
            model = XceptionWithCBAM(num_classes=2)
            assert hasattr(config, "resume") and os.path.isfile(config.resume), "Checkpoint required for testing"
            ckpt = torch.load(config.resume, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            model = model.cuda()
            model = nn.DataParallel(model)

            criterion = nn.CrossEntropyLoss()
            model.eval()
            loss_record = []
            acc_record = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss_record.append(loss.item())

                    preds = torch.argmax(outputs.data, 1)
                    iter_acc = torch.sum(preds == labels).item() / len(preds)
                    acc_record.append(iter_acc)

            epoch_loss = np.mean(loss_record)
            epoch_acc = np.mean(acc_record)
            print("Test: loss=%.6f, acc=%.6f" % (epoch_loss, epoch_acc))
            return  # 结束 test 模式



    train_loader = DataLoader(train_dataset, **kwargs)

    print(len(train_loader),len(train_loader.dataset))
    # Model initialization
    model = XceptionWithCBAM(num_classes=2)

    if hasattr(config, "resume") and os.path.isfile(config.resume):
        ckpt = torch.load(config.resume, map_location="cpu")
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("acc1", 0.0)
        model.load_state_dict(ckpt["state_dict"])
    else:
        start_epoch = 0
        best_acc = 0.0

    model = model.cuda()
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)

    os.makedirs(config.save_dir, exist_ok=True)

    for epoch in range(config.n_epoches):
        if epoch < start_epoch:
            scheduler.step()
            continue

        print("Epoch {}".format(epoch + 1))

        model.train()

        loss_record = []
        acc_record = []

        for count, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            # print(outputs.size())
            # print(labels.size())
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_loss = loss.item()
            loss_record.append(iter_loss)

            preds = torch.argmax(outputs.data, 1)
            iter_acc = torch.sum(preds == labels).item() / len(preds)
            acc_record.append(iter_acc)

            if count and count % 100 == 0:
                print("T-Iter %d: loss=%.6f, acc=%.6f"
                      % (count, iter_loss, iter_acc))

        epoch_loss = np.mean(loss_record)
        epoch_acc = np.mean(acc_record)
        print("Training: loss=%.6f, acc=%.6f" % (epoch_loss, epoch_acc))

        model.eval()
        loss_record = []
        acc_record = []

        with torch.no_grad():


            epoch_loss = np.mean(loss_record)
            epoch_acc = np.mean(acc_record)
            print("Validation: loss=%.6f, acc=%.6f" % (epoch_loss, epoch_acc))

            scheduler.step()
            ckpt_path = os.path.join(config.save_dir, "ckpt-%d.pth" % epoch)
            save_checkpoint(
                ckpt_path,
                model.state_dict(),
                epoch=epoch + 1,
                acc1=epoch_acc)

            if epoch_acc > best_acc:
                print("Best accuracy!")
                shutil.copy(ckpt_path,
                            os.path.join(config.save_dir, "best.pth"))
                best_acc = epoch_acc

            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument('--mode', type=str, default='train', help='train or test')

    args = parser.parse_args()
    main(args)
