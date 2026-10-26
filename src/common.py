"""
Shared code for the Traffic Sign Recognition project.
Used by train.py, app.py and the Colab notebook.

Contents
  - CLASS_NAMES            : the 43 GTSRB classes
  - get_transforms()       : preprocessing + augmentation
  - get_datasets()         : GTSRB (auto-download) / Kaggle folder / fake data
  - SignCNN                : custom CNN built from scratch (Conv-BN-ReLU blocks)
  - build_resnet18()       : transfer learning model
  - build_mobilenet()      : lightweight edge model (MobileNetV3-Small)
  - GradCAM                : visual explanation of predictions (Module 5)
  - fgsm_attack()          : "fooling" a CNN with adversarial noise (Module 5)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

IMG_SIZE = 64
NUM_CLASSES = 43
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5t", "Right-of-way at next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5t prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road",
    "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing for vehicles over 3.5t",
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def get_transforms(train: bool):
    """Images are kept in [0,1]; normalisation happens INSIDE the model
    (see Normalized wrapper). This makes Grad-CAM display and FGSM simpler.

    NOTE: no horizontal flip -- flipping turns 'Turn left' into 'Turn right'!
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomApply([transforms.RandomAffine(
                degrees=12, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)], p=0.8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


class _IntFolderTarget:
    """Kaggle GTSRB folders are named '0'..'42' but ImageFolder sorts them as
    strings ('0','1','10',...). This maps ImageFolder index -> true class id."""
    def __init__(self, classes):
        self.map = {i: int(c) for i, c in enumerate(classes)}

    def __call__(self, y):
        return self.map[y]


def get_datasets(root="data", source="torchvision", val_frac=0.1, seed=42):
    """Returns (train_ds, val_ds, test_ds).

    source = 'torchvision' : auto-downloads GTSRB (~350 MB) -- easiest, use on Colab
    source = 'folder'      : Kaggle layout  root/Train/<0..42>/*.png  and
                             root/Test/<0..42>/*.png (see README for a helper)
    source = 'fake'        : random images, only for smoke-testing the pipeline
    """
    tr_t, te_t = get_transforms(True), get_transforms(False)

    if source == "torchvision":
        full_train = datasets.GTSRB(root, split="train", download=True, transform=tr_t)
        full_train_eval = datasets.GTSRB(root, split="train", download=True, transform=te_t)
        test_ds = datasets.GTSRB(root, split="test", download=True, transform=te_t)
    elif source == "folder":
        full_train = datasets.ImageFolder(os.path.join(root, "Train"), transform=tr_t)
        full_train.target_transform = _IntFolderTarget(full_train.classes)
        full_train_eval = datasets.ImageFolder(os.path.join(root, "Train"), transform=te_t)
        full_train_eval.target_transform = _IntFolderTarget(full_train_eval.classes)
        test_ds = datasets.ImageFolder(os.path.join(root, "Test"), transform=te_t)
        test_ds.target_transform = _IntFolderTarget(test_ds.classes)
    elif source == "fake":
        n = 2000
        full_train = datasets.FakeData(n, (3, IMG_SIZE, IMG_SIZE), NUM_CLASSES, transform=tr_t)
        full_train_eval = datasets.FakeData(n, (3, IMG_SIZE, IMG_SIZE), NUM_CLASSES, transform=te_t)
        test_ds = datasets.FakeData(500, (3, IMG_SIZE, IMG_SIZE), NUM_CLASSES,
                                    transform=te_t, random_offset=10_000)
    else:
        raise ValueError(source)

    # train / validation split (validation uses non-augmented transform)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(full_train), generator=g).tolist()
    n_val = int(len(idx) * val_frac)
    val_ds = torch.utils.data.Subset(full_train_eval, idx[:n_val])
    train_ds = torch.utils.data.Subset(full_train, idx[n_val:])
    return train_ds, val_ds, test_ds


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Normalized(nn.Module):
    """Wraps a model so it takes raw [0,1] images and normalises internally."""
    def __init__(self, net, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def conv_block(c_in, c_out, drop):
    # Two 3x3 convs (VGG-style) + BatchNorm (Module 4) + MaxPool + Dropout (Module 3)
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(drop),
    )


class SignCNN(nn.Module):
    """Custom CNN from scratch. 64x64x3 -> 43 classes. ~1.2M parameters."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.block1 = conv_block(3, 32, 0.1)     # 64 -> 32
        self.block2 = conv_block(32, 64, 0.2)    # 32 -> 16
        self.block3 = conv_block(64, 128, 0.3)   # 16 -> 8
        self.pool = nn.AdaptiveAvgPool2d(4)      # 8x8 -> 4x4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        # He (Kaiming) initialisation -- "better weight initialisation" (Module 4)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        return self.classifier(self.pool(x))


def build_resnet18(num_classes=NUM_CLASSES, pretrained=True):
    """ResNet-18 pretrained on ImageNet, final layer replaced (transfer learning)."""
    w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=w)
    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(net.fc.in_features, num_classes))
    return net


def build_mobilenet(num_classes=NUM_CLASSES, pretrained=True):
    """MobileNetV3-Small: depthwise-separable convs, built for phones / edge devices."""
    w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_small(weights=w)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
    return net


# Model registry. "<arch>_robust" = same architecture trained with adversarial training.
ARCHS = {"cnn": "Custom CNN (from scratch)",
         "resnet18": "ResNet-18 (transfer learning)",
         "mobilenet": "MobileNetV3-Small (edge model)"}


def arch_of(name):
    return name[:-len("_robust")] if name.endswith("_robust") else name


def display_name(name):
    base = ARCHS[arch_of(name)]
    return base + " + adversarial training" if name.endswith("_robust") else base


def build_model(name, pretrained=True):
    arch = arch_of(name)
    if arch == "cnn":
        return Normalized(SignCNN())
    if arch == "resnet18":
        return Normalized(build_resnet18(pretrained=pretrained))
    if arch == "mobilenet":
        return Normalized(build_mobilenet(pretrained=pretrained))
    raise ValueError(name)


def cam_target_layer(model):
    """Last convolutional stage -- used by Grad-CAM."""
    net = model.net
    if isinstance(net, SignCNN):
        return net.block3[5]
    if hasattr(net, "layer4"):
        return net.layer4[-1]
    return net.features[-1]          # MobileNetV3


def load_model(name, path, device="cpu"):
    model = build_model(name, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# Grad-CAM  (Selvaraju et al., 2017) -- "Visualizing CNNs", Module 5
# --------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = self.grads = None
        layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W) in [0,1]. Returns (heatmap HxW in [0,1], class_idx)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # GAP of gradients
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))      # weighted sum of maps
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx


def overlay_cam(img_chw, cam, alpha=0.45):
    """img_chw: tensor (3,H,W) in [0,1]; cam: HxW numpy. Returns HxWx3 uint8."""
    import matplotlib.cm as cm
    img = img_chw.permute(1, 2, 0).cpu().numpy()
    heat = cm.jet(cam)[..., :3]
    out = (1 - alpha) * img + alpha * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# FGSM adversarial attack (Goodfellow et al., 2015) -- "Fooling CNNs", Module 5
# --------------------------------------------------------------------------
def fgsm_attack(model, x, y, eps):
    """x in [0,1]. Returns adversarial images x + eps*sign(grad_x loss)."""
    if eps == 0:
        return x
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    model.zero_grad()
    loss.backward()
    return (x + eps * x.grad.sign()).clamp(0, 1).detach()


def count_flops(model, device="cpu"):
    """Multiply-accumulate based FLOPs for one 64x64 image (None if unsupported)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        return int(fc.get_total_flops())
    except Exception:
        return None
