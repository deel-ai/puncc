import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

DATA_DIR    = "/datasets/shared_datasets/imagenet/ILSVRC/Data/CLS-LOC/val"
OUT_FILE    = "imagenet_val_softmax_scores.npz"
BATCH_SIZE  = 128
NUM_WORKERS = 4

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cudnn.benchmark = True

    dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = torchvision.models.resnet152(pretrained=True, progress=True).cuda()
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Computing softmax scores"):
            logits = model(images.cuda())
            all_scores.append(F.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    np.savez_compressed(OUT_FILE, scores=all_scores, labels=all_labels)
    print(f"Saved to {OUT_FILE}  —  scores {all_scores.shape}, labels {all_labels.shape}")
