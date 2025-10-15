"""IoU"""

import numpy as np
import torch
import imageio
import skimage.transform as sktf
import os

SCANNET20_CLASS_LABELS = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
)

COCOMAP_CLASS_LABELS = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "shelves",
    "counter",
    "curtain",
    "ceiling",
    "refridgerator",
    "television",
    "person",
    "toilet",
    "sink",
    "lamp",
    "bag",
)

COLORMAP = [
    (0.0, 0.0, 0.0),
    (174.0, 199.0, 232.0),
    (152.0, 223.0, 138.0),
    (31.0, 119.0, 180.0),
    (255.0, 187.0, 120.0),
    (188.0, 189.0, 34.0),
    (140.0, 86.0, 75.0),
    (255.0, 152.0, 150.0),
    (214.0, 39.0, 40.0),
    (197.0, 176.0, 213.0),
    (148.0, 103.0, 189.0),
    (196.0, 156.0, 148.0),
    (23.0, 190.0, 207.0),
    (247.0, 182.0, 210.0),
    (219.0, 219.0, 141.0),
    (255.0, 127.0, 14.0),
    (158.0, 218.0, 229.0),
    (44.0, 160.0, 44.0),
    (112.0, 128.0, 144.0),
    (227.0, 119.0, 194.0),
    (213.0, 92.0, 176.0),
    (94.0, 106.0, 211.0),
    (82.0, 84.0, 163.0),
    (100.0, 85.0, 144.0),
    (66.0, 188.0, 102.0),
    (140.0, 57.0, 197.0),
    (202.0, 185.0, 52.0),
    (51.0, 176.0, 203.0),
    (200.0, 54.0, 131.0),
    (92.0, 193.0, 61.0),
    (78.0, 71.0, 183.0),
    (172.0, 114.0, 82.0),
    (91.0, 163.0, 138.0),
    (153.0, 98.0, 156.0),
    (140.0, 153.0, 101.0),
    (100.0, 125.0, 154.0),
    (178.0, 127.0, 135.0),
    (146.0, 111.0, 194.0),
    (96.0, 207.0, 209.0),
]

def confusion_matrix(pred_ids, gt_ids, num_classes):
    """calculate the confusion matrix."""

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)

    # the sum of each row (axis=1) is predicted truth, the sum of each column (axis=0) is ground truth
    confusion = (
        np.bincount(pred_ids * (num_classes + 1) + gt_ids, minlength=(num_classes + 1) ** 2)
        .reshape((num_classes + 1, num_classes + 1))
        .astype(np.ulonglong)
    )
    return confusion[:, 1:] # do not calculate unlabeled points (the first column)

def get_iou(label_id, confusion):
    """calculate IoU."""

    # true positives
    tp = np.longlong(confusion[label_id + 1, label_id])
    # false positives
    fp = np.longlong(confusion[label_id + 1, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = tp + fp + fn
    if denom == 0:
        return float("nan")
    return float(tp) / denom, tp, denom


def evaluate_confusion(scene_name, confusion, stdout=False, dataset="scannet20"):
    if stdout:
        print("evaluating", confusion.sum(), "points...")

    if "scannet20" in dataset:
        CLASS_LABELS = SCANNET20_CLASS_LABELS
    elif "cocomap" in dataset:
        CLASS_LABELS = COCOMAP_CLASS_LABELS
    else:
        raise NotImplementedError
    N_CLASSES = len(CLASS_LABELS)
    print("num_classes:", N_CLASSES)

    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if confusion.sum(axis=0)[i] == 0:  # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / confusion.sum(axis=0)[i]
        count += 1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    mean_iou /= count
    mean_acc /= count
    if stdout:
        print(f"Scene: {scene_name}")
        print("classes          IoU")
        print("----------------------------")
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                print(
                    "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2],
                    )
                )
            except:
                print(label_name + " error!")
                continue
        print("Mean IoU", mean_iou)
        print("Mean Acc", mean_acc)

    with open("eval_result.log", "a") as fp:
        fp.write(f"Scene: {scene_name}\n")
        fp.write("classes,IoU\n")
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                fp.write(
                    "{0:<14s}: {1:>5.3f}  ({2:>6d}/{3:<6d})\n".format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2],
                    )
                )
            except:
                fp.write(label_name + ",error\n")
        fp.write("mean IoU,{}\n".format(mean_iou))
        fp.write("mean Acc,{}\n\n".format(mean_acc))
    return mean_iou, mean_acc

def get_text_requests(dataset_name="scannet20"):
    if isinstance(dataset_name, list):
        labelset = dataset_name
    elif dataset_name == "scannet20":
        labelset = list(SCANNET20_CLASS_LABELS)
    elif dataset_name == "cocomap":
        labelset = list(COCOMAP_CLASS_LABELS)

    # add unlabeled label and palette
    labelset = ["other"] + labelset

    palette = torch.tensor(COLORMAP[:len(labelset)+1]).cuda().flatten()

    return palette, labelset

def get_mapped_label(height, width, image_path, label_mapping):
    label_path = str(image_path).replace("color", "label-filt").replace(".jpg", ".png")
    if not os.path.exists(label_path):
        return None

    label_img = np.array(imageio.imread(label_path))
    label_img = sktf.resize(label_img, [height, width], order=0, preserve_range=True)
    mapped = np.copy(label_img)
    for k, v in label_mapping.items():
        mapped[label_img == k] = v
    label_img = mapped.astype(np.uint8)

    return label_img

def render_palette(label, palette, rgb_palette=True):
    shape = label.shape
    label = label.reshape(-1)
    new_3d = torch.zeros((label.shape[0], 3)).cuda()
    u_index = torch.unique(label)
    for index in u_index:
        if rgb_palette:
            new_3d[label == index] = torch.tensor(
                [
                    palette[index * 3] / 255.0,
                    palette[index * 3 + 1] / 255.0,
                    palette[index * 3 + 2] / 255.0,
                ]
            ).cuda()
        else:
            new_3d[label == index] = torch.tensor(
                [
                    palette[index * 3 + 2] / 255.0,
                    palette[index * 3 + 1] / 255.0,
                    palette[index * 3] / 255.0,
                ]
            ).cuda()

    return new_3d.reshape(*shape, 3).permute(2, 0, 1)