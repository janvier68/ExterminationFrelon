# super_augmentation_yolo_v3_focus.py
import os, glob, random, shutil
import cv2
import numpy as np
import albumentations as A
import argparse
from tqdm import tqdm

# ---------- I/O labels ----------
def read_yolo_labels(txt_path):
    boxes, classes = [], []
    if not os.path.exists(txt_path): 
        return boxes, classes
    with open(txt_path) as f:
        for l in f:
            p = l.strip().split()
            if len(p) == 5:
                c, xc, yc, w, h = p
                boxes.append([float(xc), float(yc), float(w), float(h)])
                classes.append(int(c))
    return boxes, classes

def write_yolo_labels(txt_path, boxes, classes):
    with open(txt_path, "w") as f:
        for b, c in zip(boxes, classes):
            f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

# ---------- effets additionnels ----------
def add_black_squares(img, max_squares=3):
    h, w = img.shape[:2]
    n = random.randint(0, max_squares)
    for _ in range(n):
        sz = random.randint(int(0.05*min(h,w)), int(0.2*min(h,w)))
        x1 = random.randint(0, max(1, w - sz))
        y1 = random.randint(0, max(1, h - sz))
        cv2.rectangle(img, (x1,y1), (x1+sz, y1+sz), (0,0,0), thickness=-1)
    return img

# ---------- focus sur les insectes (crop & zoom/dézoom) ----------
def crop_around_boxes(img, boxes, classes, min_box_area=0.0001, max_scale=6.0):
    """
    img : HxWx3
    boxes : YOLO normalisé [xc, yc, w, h]
    On découpe un crop centré sur 1 à N boîtes, avec un facteur de zoom contrôlé
    pour que les insectes ne soient pas microscopiques dans l'image finale.
    """
    h, w = img.shape[:2]
    if not boxes:
        return img, boxes, classes  # rien à faire

    # filtrer les boîtes trop petites (optionnel)
    keep_b, keep_c = [], []
    for b, c in zip(boxes, classes):
        if b[2] * b[3] >= min_box_area:
            keep_b.append(b)
            keep_c.append(c)
    if not keep_b:
        # si on a tout filtré, on garde les boîtes originales
        keep_b, keep_c = boxes, classes

    boxes = keep_b
    classes = keep_c

    # choisir 1 à 3 boîtes à couvrir par le crop
    k = min(len(boxes), random.randint(1, 3))
    sel_indices = random.sample(range(len(boxes)), k)

    # bbox en pixels englobant les boîtes choisies
    x_min, y_min = 1e9, 1e9
    x_max, y_max = -1e9, -1e9
    for i in sel_indices:
        xc, yc, bw, bh = boxes[i]
        xc_px, yc_px = xc * w, yc * h
        bw_px, bh_px = bw * w, bh * h
        x1 = xc_px - bw_px/2
        y1 = yc_px - bh_px/2
        x2 = xc_px + bw_px/2
        y2 = yc_px + bh_px/2
        x_min = min(x_min, x1)
        y_min = min(y_min, y1)
        x_max = max(x_max, x2)
        y_max = max(y_max, y2)

    # rectangle englobant en pixels
    box_w = x_max - x_min
    box_h = y_max - y_min

    # facteur d'agrandissement du crop par rapport à la bbox
    # si les insectes sont très petits, box_w/box_h seront petits, donc scale limité
    # zoom/dézoom léger : entre 1.5 et max_scale (mais clampé par la taille de l'image)
    scale = random.uniform(1.5, max_scale)

    # taille cible du crop
    crop_w = box_w * scale
    crop_h = box_h * scale

    # forcer un crop pas trop petit (évite 640x640 super zoomés sur 3 pixels)
    min_side = 0.3 * min(w, h)
    crop_w = max(crop_w, min_side)
    crop_h = max(crop_h, min_side)

    # rendre le crop +/- carré pour faciliter le resize 640x640
    side = max(crop_w, crop_h)
    crop_w = crop_h = side

    # centre du rectangle englobant
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    # coordonnées du crop en pixels
    x1 = cx - crop_w / 2
    y1 = cy - crop_h / 2
    x2 = cx + crop_w / 2
    y2 = cy + crop_h / 2

    # clamp dans l'image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # recalcule la taille finale réelle du crop après clamp
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 1 or crop_h <= 1:
        return img, boxes, classes  # crop degénéré, on laisse tomber

    # crop image
    crop_img = img[int(y1):int(y2), int(x1):int(x2)]

    # adapter les boîtes au nouveau repère
    new_boxes, new_classes = [], []
    for b, c in zip(boxes, classes):
        xc, yc, bw, bh = b
        xc_px, yc_px = xc * w, yc * h
        bw_px, bh_px = bw * w, bh * h

        # centre dans le crop
        xc_c = xc_px - x1
        yc_c = yc_px - y1
        if xc_c < 0 or xc_c > crop_w or yc_c < 0 or yc_c > crop_h:
            continue  # centre en dehors du crop => on ignore

        # YOLO normalisé dans le crop
        xc_n = xc_c / crop_w
        yc_n = yc_c / crop_h
        bw_n = bw_px / crop_w
        bh_n = bh_px / crop_h

        # filtrage minimal (évite des boîtes ridicules)
        if bw_n <= 0 or bh_n <= 0:
            continue

        # clamp léger
        xc_n = min(max(xc_n, 0.0), 1.0)
        yc_n = min(max(yc_n, 0.0), 1.0)
        bw_n = min(max(bw_n, 0.0), 1.0)
        bh_n = min(max(bh_n, 0.0), 1.0)

        new_boxes.append([xc_n, yc_n, bw_n, bh_n])
        new_classes.append(c)

    if not new_boxes:
        # si on a perdu toutes les boîtes, on retourne l'original pour ne pas jeter l'échantillon
        return img, boxes, classes

    return crop_img, new_boxes, new_classes

# ---------- pipelines () ----------
def make_aug_pipeline():
    return A.Compose([
        # Transformations géométriques
        A.OneOf([
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Affine(scale=(0.9, 1.2), translate_percent=(0.0, 0.1), p=0.5)
        ], p=0.7),

        # Changement lumière / contraste / gamma (léger)
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.4
        ),
        A.RandomGamma(
            gamma_limit=(90, 110),
            p=0.2
        ),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.2,
        clip=True
    ))



def mosaic_4(items):
    """
    items: liste de tuples (img, boxes, classes)
    Mosaic 2x2 : on met 4 images (recadrées) dans une grille 2x2.
    Toutes les images sont redimensionnées à la taille de la première.
    Les boîtes YOLO sont ajustées au nouveau repère (2*W x 2*H).
    """
    assert len(items) >= 1

    base_h, base_w = items[0][0].shape[:2]

    # homogénéiser tailles
    proc = []
    for img, boxes, cls in items:
        if img.shape[:2] != (base_h, base_w):
            img = cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
        proc.append((img, boxes, cls))

    # on veut exactement 4 "cases"
    while len(proc) < 4:
        proc.append(random.choice(proc))
    proc = proc[:4]

    out_h, out_w = base_h * 2, base_w * 2
    out = np.zeros((out_h, out_w, 3), dtype=proc[0][0].dtype)

    out_boxes, out_cls = [], []

    for q, (img, boxes, cls) in enumerate(proc):
        # quadrants: 0=haut-gauche, 1=haut-droite, 2=bas-gauche, 3=bas-droite
        qx = q % 2  # 0 ou 1
        qy = q // 2  # 0 ou 1
        x_off = qx * base_w
        y_off = qy * base_h

        out[y_off:y_off+base_h, x_off:x_off+base_w] = img

        # Ajustement YOLO:
        # x' = (x*base_w + x_off) / (2*base_w) = x/2 + qx/2
        # y' = (y*base_h + y_off) / (2*base_h) = y/2 + qy/2
        # w' = (w*base_w) / (2*base_w) = w/2
        # h' = (h*base_h) / (2*base_h) = h/2
        for b, c in zip(boxes, cls):
            x, y, w_, h_ = b
            x_new = x * 0.5 + qx * 0.5
            y_new = y * 0.5 + qy * 0.5
            w_new = w_ * 0.5
            h_new = h_ * 0.5

            # clamp
            x_new = min(max(x_new, 0.0), 1.0)
            y_new = min(max(y_new, 0.0), 1.0)
            w_new = min(max(w_new, 0.0), 1.0)
            h_new = min(max(h_new, 0.0), 1.0)

            if w_new <= 0 or h_new <= 0:
                continue

            out_boxes.append([x_new, y_new, w_new, h_new])
            out_cls.append(c)

    return out, out_boxes, out_cls


# ---------- composition/mix ----------
def mix_images(img1, boxes1, cls1, img2, boxes2, cls2):
    """Mélange gauche/droite après avoir mis les deux images à la même taille."""
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)

    cut_x = random.randint(int(w*0.3), int(w*0.7))
    mix = np.zeros_like(img1)
    mix[:, :cut_x] = img1[:, :cut_x]
    mix[:, cut_x:] = img2[:, cut_x:]

    boxes, classes = [], []
    # Conserver uniquement les boîtes dont le centre est du bon côté
    for b, c in zip(boxes1, cls1):
        if b[0] < cut_x / w:
            boxes.append(b); classes.append(c)
    for b, c in zip(boxes2, cls2):
        if b[0] > cut_x / w:
            boxes.append(b); classes.append(c)

    return mix, boxes, classes

def concat_side_by_side(items):
    """
    items: liste de tuples (img, boxes, classes)
    Renvoie: img_concat, boxes_concat, classes_concat
    Règle: toutes les images sont redimensionnées à la taille de la première.
    Les boîtes YOLO sont ajustées pour la nouvelle largeur (k*w).
    """
    assert len(items) >= 2
    base_h, base_w = items[0][0].shape[:2]

    # homogénéiser tailles
    proc = []
    for img, boxes, cls in items:
        if img.shape[:2] != (base_h, base_w):
            img = cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
        proc.append((img, boxes, cls))

    k = len(proc)
    out_w, out_h = base_w * k, base_h
    out = np.zeros((out_h, out_w, 3), dtype=proc[0][0].dtype)

    out_boxes, out_cls = [], []
    for i, (img, boxes, cls) in enumerate(proc):
        x_off = i * base_w
        out[:, x_off:x_off+base_w] = img

        # Ajustement YOLO:
        # x' = (x*base_w + x_off) / (k*base_w)  => x'/=k puis + (i/k)
        # y' = y
        # w' = w / k
        # h' = h
        for b, c in zip(boxes, cls):
            x, y, w_, h_ = b
            x_new = x / k + i / k
            w_new = w_ / k
            x_new = min(max(x_new, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            out_boxes.append([x_new, y, w_new, h_])
            out_cls.append(c)

    return out, out_boxes, out_cls


# ---------- pipeline principal ----------
def process_all(in_images="images", in_labels="labels", out_dir="dataset_final", target_count=1000):
    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(sum([
        glob.glob(os.path.join(in_images, ext)) 
        for ext in ["*.jpg","*.png","*.jpeg","*.bmp"]
    ], []))
    if not img_paths: 
        return

    aug = make_aug_pipeline()
    out_img_dir = os.path.join(out_dir, "all/images")
    out_lbl_dir = os.path.join(out_dir, "all/labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    total = 0
    idx = 0
    
    # barre de progression
    pbar = tqdm(total=target_count, desc="Data augmentation")
    
    while total < target_count:
        img_path = img_paths[idx % len(img_paths)]
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(in_labels, base + ".txt")
        img = cv2.imread(img_path)
        boxes, cls = read_yolo_labels(lbl_path)
        if img is None or not boxes:
            idx += 1
            continue

        # 1) focus sur les insectes (crop + zoom/dézoom)
        img_f, boxes_f, cls_f = crop_around_boxes(img, boxes, cls)
        img, boxes, cls = img_f, boxes_f, cls_f

        # 2) mix / concat (éventuel) sur ces images déjà "focus"
        r = random.random()
        if r < 0.2 and len(img_paths) > 1:
            # mix 2 images
            img2_path = random.choice(img_paths)
            base2 = os.path.splitext(os.path.basename(img2_path))[0]
            lbl2_path = os.path.join(in_labels, base2 + ".txt")
            img2 = cv2.imread(img2_path)
            b2, c2 = read_yolo_labels(lbl2_path)
            if img2 is not None and b2:
                img2_f, b2_f, c2_f = crop_around_boxes(img2, b2, c2)
                img, boxes, cls = mix_images(img, boxes, cls, img2_f, b2_f, c2_f)

        elif r < 0.4 and len(img_paths) > 2:
            # concat N images (2 à 4)
            k = random.randint(2, 4)
            sel_paths = random.sample(img_paths, k-1)  # -1 car img courant inclus
            items = [(img, boxes, cls)]
            ok = True
            for p in sel_paths:
                bname = os.path.splitext(os.path.basename(p))[0]
                lpath = os.path.join(in_labels, bname + ".txt")
                im = cv2.imread(p)
                if im is None:
                    ok = False
                    break
                b_, c_ = read_yolo_labels(lpath)
                if not b_:
                    continue
                im_f, b_f, c_f = crop_around_boxes(im, b_, c_)
                items.append((im_f, b_f, c_f))
            items = [it for it in items if it[1]]  # garder ceux avec boxes
            if ok and len(items) >= 2:
                img, boxes, cls = concat_side_by_side(items)

        elif r < 0.6 and len(img_paths) > 3:
            # mosaic 2x2 (4 images max)
            k = random.randint(2, 4)
            sel_paths = random.sample(img_paths, k-1)  # -1 car img courant inclus
            items = [(img, boxes, cls)]
            ok = True
            for p in sel_paths:
                bname = os.path.splitext(os.path.basename(p))[0]
                lpath = os.path.join(in_labels, bname + ".txt")
                im = cv2.imread(p)
                if im is None:
                    ok = False
                    break
                b_, c_ = read_yolo_labels(lpath)
                if not b_:
                    continue
                im_f, b_f, c_f = crop_around_boxes(im, b_, c_)
                if b_f:
                    items.append((im_f, b_f, c_f))
            items = [it for it in items if it[1]]
            if ok and len(items) >= 2:
                img, boxes, cls = mosaic_4(items)
                
        if not boxes:
            idx += 1
            continue

        # 3) Albumentations sur image et boîtes (bruit léger)
        augm = aug(image=img, bboxes=boxes, class_labels=cls)
        img_o, b_o, c_o = augm["image"], augm["bboxes"], augm["class_labels"]
        if not b_o:
            idx += 1
            continue

        # 4) carrés noirs éventuels (désactivés par défaut)
        # img_o = add_black_squares(img_o, max_squares=3)

        # 5) redimension final 640x640
        img_o = cv2.resize(img_o, (640, 640), interpolation=cv2.INTER_LINEAR)
        # Les bboxes sont déjà normalisées dans le repère de img_o, donc rien à changer.

        out_name = f"{base}_aug_{total}.jpg"
        cv2.imwrite(os.path.join(out_img_dir, out_name), img_o)
        write_yolo_labels(os.path.join(out_lbl_dir, out_name.replace(".jpg", ".txt")), b_o, c_o)

        total += 1
        pbar.update(1)
        idx += 1
    pbar.close()
    split_dataset(out_dir)

# ---------- split 80/10/10 ----------
def split_dataset(root_dir, ratios=(0.8, 0.1, 0.1)):
    img_dir = os.path.join(root_dir, "all/images")
    lbl_dir = os.path.join(root_dir, "all/labels")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(ratios[0]*n)
    n_val = int(ratios[1]*n)

    sets = {"train": imgs[:n_train],
            "valid": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:]}

    for s, subset in sets.items():
        os.makedirs(os.path.join(root_dir, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, s, "labels"), exist_ok=True)
        for im_path in subset:
            base = os.path.basename(im_path)
            lbl_path = os.path.join(lbl_dir, base.replace(".jpg", ".txt"))
            shutil.copy(im_path, os.path.join(root_dir, s, "images", base))
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(root_dir, s, "labels", base.replace(".jpg", ".txt")))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default="origineLabel",
        help="Dossier d'entrée contenant images/ et labels/"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="dataAugmented",
        help="Dossier de sortie"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3000,
        help="Nombre d'images augmentées à générer"
    )

    args = parser.parse_args()

    process_all(
        in_images=os.path.join(args.in_dir, "images"),
        in_labels=os.path.join(args.in_dir, "labels"),
        out_dir=args.out_dir,
        target_count=args.count
    )

"""
python augmente.py --in_dir origineLabel --out_dir dataAugmented --count 3000
"""
