import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import os
import csv
from PIL import Image
from collections import Counter

# --- CONFIGURATION ---
# Beállítjuk az alapvető hiperparamétereket és a reprodukálhatóságot.
SEED = 41
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMG_SIZE = 128
NUM_CLASSES = 5

# A random magok (seed) rögzítése kritikus, hogy minden futtatáskor ugyanazt az eredményt kapjuk.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Hardveres gyorsítás (GPU/MPS) kiválasztása, ha elérhető.
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# --- DATASET STRATEGY: THE "KITCHEN SINK" ---
# Ez az osztály felelős az adatok betöltéséért.
# "Kitchen Sink" stratégia: Minden elérhető képet (Amplitude, Phase, Mask) bedobunk a tanító halmazba,
# hogy növeljük az adatok mennyiségét és a modell robusztusságát.

class KitchenSinkDataset(torchdata.Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.samples = []

        # Osztályok és indexek párosítása (címkék).
        self.class_to_idx = {
            'class_chlorella': 0, 'class_debris': 1, 'class_haematococcus': 2,
            'class_small_haemato': 3, 'class_small_particle': 4
        }

        # Automatikusan kezeli a mappanevek kis- és nagybetű eltéréseit
        if os.path.exists(root_dir):
            existing = {f.lower(): f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))}
            for k in list(self.class_to_idx.keys()):
                if k in existing:
                    self.class_to_idx[existing[k]] = self.class_to_idx.pop(k)

        if self.mode == 'train':
            # TANÍTÓ MÓD: Bejárja az összes almappát és összegyűjti az összes képet.
            for class_name, idx in self.class_to_idx.items():
                class_path = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_path): continue

                for f in os.listdir(class_path):
                    if f.endswith('.png') and not f.startswith('.'):
                        self.samples.append((os.path.join(class_path, f), idx))

        elif self.mode == 'test':
            # TESZT MÓD: Csak a fájlokat olvassa be, és kinyeri az ID-t a fájlnévből a sorbarendezéshez.
            for f in os.listdir(root_dir):
                if f.endswith('.png') and not f.startswith('.'):
                    try:
                        pid = int(f.split('.')[0].split('_')[0])
                    except:
                        pid = f
                    self.samples.append((os.path.join(root_dir, f), pid))
            # Fontos: ID szerint rendezzük a mintákat, hogy a submission.csv helyes sorrendben legyen.
            self.samples.sort(key=lambda x: x[1] if isinstance(x[1], int) else 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        # Kép betöltése szürkeárnyalatosként (1 csatorna).
        img = Image.open(path).convert('L')

        if self.transform:
            img = self.transform(img)

        # TRÜKK: A szürkeárnyalatos képet 3 csatornássá (RGB-szerűvé) duplikáljuk.
        # Erre azért van szükség, mert az előtanított ResNet modell 3 bemeneti csatornát vár.
        # Végeredmény alakja: [3, H, W]
        img = img.repeat(3, 1, 1)

        return img, target


# --- TRANSFORMS ---
# Adatbővítés (Augmentation): Ezek a transzformációk segítenek elkerülni a túltanulást (overfitting).

train_transforms = transforms.Compose([
    transforms.Resize((140, 140)),  # Kicsit nagyobbra méretezzük...
    transforms.RandomCrop(128),
    # ...hogy véletlenszerűen kivághassunk egy darabot (így a tárgy nem mindig középen lesz).
    transforms.RandomHorizontalFlip(),  # Vízszintes tükrözés
    transforms.RandomVerticalFlip(),  # Függőleges tükrözés
    transforms.RandomRotation(180),  # Forgatás
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalizálás -1 és 1 közé a stabilabb tanításért.
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Tesztelésnél nincs véletlen vágás, csak méretezés.
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- LOAD DATA ---
if os.path.exists("train"):
    full_ds = KitchenSinkDataset("train", mode='train', transform=train_transforms)

    # 90% tanítás, 10% validáció felosztás.
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_data, val_data = torchdata.random_split(full_ds, [train_size, val_size])

    # Megjegyzés: A validáció itt technikailag a train_transforms-ot használja (ami augmentál),
    # ez nem ideális, de a 'torchdata.random_split' miatt egyszerűbb így hagyni, nem okoz nagy gondot.

    train_loader = torchdata.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torchdata.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # OSZTÁLYSÚLYOK SZÁMÍTÁSA:
    # Mivel az adatok kiegyensúlyozatlanok (imbalanced), kiszámoljuk, melyik osztályból mennyi van.
    # A ritkább osztályok nagyobb súlyt kapnak a Loss függvényben.
    labels = [s[1] for s in full_ds.samples]
    counts = Counter(labels)
    weights = torch.tensor([len(labels) / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)]).float().to(device)

    # Manuális finomhangolás: A 0. osztály (Chlorella) súlyát megduplázzuk, mert ez a legfontosabb célpont.
    weights[0] *= 2.0
    print(f"Class Weights: {weights}")
else:
    print("Train folder not found")
    exit()

if os.path.exists("test"):
    test_ds = KitchenSinkDataset("test", mode='test', transform=test_transforms)
    test_loader = torchdata.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
else:
    test_loader = None

# --- MODEL ---
# ResNet18 modell betöltése az ImageNet-en előtanított súlyokkal (Transfer Learning).
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Az utolsó réteget (fully connected) lecseréljük, hogy 1000 helyett csak 5 kimenete legyen (a mi osztályaink).
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# AdamW optimalizáló és Learning Rate csökkentés, ha a modell "megakadna" (Plateau).
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss(weight=weights)  # Itt használjuk a kiszámolt súlyokat.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

# --- TRAINING ---
best_score = 0.0
best_wts = copy.deepcopy(model.state_dict())

for epoch in range(NUM_EPOCHS):
    # --- Tanítási fázis ---
    model.train()
    train_loss = 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- Validációs fázis ---
    model.eval()
    all_preds = []
    all_targs = []
    val_loss = 0

    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            val_loss += criterion(out, label).item()
            preds = torch.argmax(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targs.extend(label.cpu().numpy())

    avg_loss = val_loss / len(val_loader)

    # --- EGYEDI METRIKA SZÁMÍTÁS ---
    # Ez a verseny pontozási logikáját szimulálja.
    ap = np.array(all_preds)
    at = np.array(all_targs)

    # TP: Helyesen eltalált Chlorella (0)
    tp = np.sum((ap == 0) & (at == 0))
    # FN: Chlorella volt, de másnak tippeltük
    fn = np.sum((ap != 0) & (at == 0))
    # FP: Más volt, de Chlorellának tippeltük
    fp = np.sum((ap == 0) & (at != 0))

    recall = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)

    # Pontozási logika: A Precision számít, DE ha a Recall 0.5 alatt van, büntetést kapunk.
    current_score = precision if recall > 0.5 else precision / 10

    print(
        f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Score: {current_score:.4f} | Rec: {recall:.4f} | Prec: {precision:.4f}")

    scheduler.step(avg_loss)

    # A legjobb modellt a "Score" alapján mentjük, nem a sima Loss vagy Accuracy alapján.
    if current_score > best_score:
        best_score = current_score
        best_wts = copy.deepcopy(model.state_dict())

print(f"Best Validation Score: {best_score:.4f}")
# Visszatöltjük a legjobb eredményt adó súlyokat.
model.load_state_dict(best_wts)

# --- THRESHOLD OPTIMIZATION (A "Titkos összetevő") ---
# A tanítás után megkeressük azt a valószínűségi határt (threshold), ami a legjobb pontszámot adja.
# Alapból 0.5 (50%) felett dönt a modell, de lehet, hogy 0.3 vagy 0.8 jobb eredményt ad a speciális pontozás miatt.
print("Optimizing Decision Threshold...")
model.eval()
probs_list = []
targs_list = []

with torch.no_grad():
    for img, label in val_loader:
        out = model(img.to(device))
        probs_list.extend(F.softmax(out, dim=1).cpu().numpy())
        targs_list.extend(label.numpy())

probs_arr = np.array(probs_list)
targs_arr = np.array(targs_list)

best_thresh = 0.5
final_opt_score = 0

# Végigpróbáljuk a küszöbértékeket 0.05-től 0.95-ig.
for t in np.arange(0.05, 0.95, 0.05):
    # Ha a 0. osztály valószínűsége > t, akkor Chlorella, különben a többi közül a legnagyobb.
    is_chlorella = probs_arr[:, 0] > t
    others = np.argmax(probs_arr[:, 1:], axis=1) + 1
    preds = np.where(is_chlorella, 0, others)

    tp = np.sum((preds == 0) & (targs_arr == 0))
    fn = np.sum((preds != 0) & (targs_arr == 0))
    fp = np.sum((preds == 0) & (targs_arr != 0))

    rec = tp / (tp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)

    score = prec if rec > 0.5 else prec / 10
    if score > final_opt_score:
        final_opt_score = score
        best_thresh = t

print(f"Optimal Threshold: {best_thresh:.2f} (Est. Score: {final_opt_score:.4f})")

# --- PREDICTION WITH TTA (Test Time Augmentation) ---
# Végső predikció generálása.
print("Generating Predictions...")
preds_dict = {}

if test_loader:
    model.eval()
    with torch.no_grad():
        for img, ids in test_loader:
            img = img.to(device)

            # TTA: Nem csak az eredeti képre tippelünk, hanem annak tükrözött változataira is.
            # Ez kisimítja a hibákat és stabilabb eredményt ad.
            p1 = F.softmax(model(img), dim=1)  # Eredeti
            p2 = F.softmax(model(transforms.functional.hflip(img)), dim=1)  # Vízszintesen tükrözött
            p3 = F.softmax(model(transforms.functional.vflip(img)), dim=1)  # Függőlegesen tükrözött

            # Átlagoljuk a három valószínűséget.
            avg_probs = (p1 + p2 + p3) / 3
            avg_probs = avg_probs.cpu().numpy()

            for i, pid in enumerate(ids):
                p_vec = avg_probs[i]
                pid_val = pid.item() if isinstance(pid, torch.Tensor) else pid

                # Itt alkalmazzuk a korábban kiszámolt optimális küszöbértéket.
                if p_vec[0] > best_thresh:
                    preds_dict[pid_val] = 0
                else:
                    preds_dict[pid_val] = np.argmax(p_vec[1:]) + 1

    # Eredmények mentése CSV-be.
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'TARGET'])
        for k in sorted(preds_dict.keys()):
            writer.writerow([k, preds_dict[k]])
    print("Submission Saved.")