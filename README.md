# AI Employee Monitoring (CCTV-style)

Office camera feed se har worker ko detect + track karta hai aur ID ke saath status dikhata hai
(bilkul reference images ki tarah):

- 🟢 **ID 1 - WORKING** — person desk zone me hai
- 🔴 **ID x - PHONE USAGE** — person box ke andar cell phone detect hua
- 🔴 **ID x - ABSENT** — person ID K seconds se camera me nahi aaya (last position par red box)
- 🟠 **ID x - IDLE** — person visible magar zone ke bahar
- 🟡 `phone` — phone par yellow box
- Top-left panel: sab IDs ki list + `Working: N  Alerts: M`

## Folder structure

```
camera model/
├── app.py              # Streamlit app (main file)
├── engine.py           # detection + tracking + status logic + overlay drawing
├── config.json         # desk zones (normalized 0-1 points) + settings
├── kaggle_train.ipynb  # Kaggle par model fine-tune karne ka notebook
├── requirements.txt
├── README.md
└── models/
    └── best.pt         # (optional) Kaggle se trained model yahan rakhein
```

## Local test

```bash
cd "camera model"
pip install -r requirements.txt
streamlit run app.py
```

Model file `models/best.pt` na ho to app khud `yolov8n.pt` download kar lega
(COCO me `person` + `cell phone` already detect hoti hain, is liye custom training
optional hai — accuracy improve karne ke liye karein).

## 1) Kaggle par train

1. [kaggle.com](https://www.kaggle.com) → New Notebook.
2. `kaggle_train.ipynb` ke cells paste karein (ya file upload karein).
3. **Settings → Accelerator → GPU T4** select karein.
4. Dataset: apna office/person+phone dataset (Roboflow → Export → YOLOv8) attach karein
   aur `DATA_YAML` path update karein. (Dataset na ho to Cell 3-5 skip karein.)
5. Training ke baad `/kaggle/working/best.pt` download karein.

## 2) GitHub par upload

```bash
cd "camera model"
git init
git add .
git commit -m "AI Employee Monitoring model"
git branch -M main
git remote add origin https://github.com/USERNAME/employee-monitoring.git
git push -u origin main
```

`best.pt` ko repo ke `models/` folder me daal dein (yolov8n ~6MB, GitHub limit 100MB — theek hai).

## 3) Streamlit par deploy

1. [share.streamlit.io](https://share.streamlit.io) → **Create app**.
2. GitHub repo + branch select karein, **Main file path:** `app.py`.
3. Deploy! (requirements.txt automatically use ho ga)

RTSP/IP camera URL aur video upload dono app ke andar supported hain.

## Zones (desks) badalna

`config.json` me `zones` ke `points` **normalized (0–1)** hain — kisi bhi resolution par kaam karte hain.
Apne camera view ke hisaab se desk ke corners ki values update kar dein.
Agar `zones: []` khali rakha jaye to har visible person WORKING count ho ga (zone check skip).
