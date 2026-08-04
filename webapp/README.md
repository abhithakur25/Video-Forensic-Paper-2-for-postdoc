# Web application — OM²AHL-BiG Video Forgery Detection

Browser UI for **Paper 2** (intra-frame video forgery detection), styled after the
ForensiQ / AT SCHOOL.IN design references (dark theme, upload drop-zone, results cards).

## What it does

| Stage | Desktop (`GUI.py`) | Web (`webapp/app.py`) |
|---|---|---|
| Select media | `select_data_event` | Upload video/image |
| Preprocessing | Viola–Jones ROI | Haar cascade face ROI |
| GradCAM | MobileNetV2 GradCAM | Lightweight Laplacian heat overlay |
| ResNet-stat | ResNet101 + mean/var/std/skew/kurt | Local mean/var/std maps |
| SIFT-VGG | VGG16 + SIFT | ORB keypoint overlay |
| Shape-ResNet | Shape descriptor | Contour structure map |
| Verdict | Full BiLSTM-GBM (offline) | Demo forensic score (+ paper metrics page) |

Full OM²AHL-BiG training stays offline (`driver.py evaluate` / `Main.py`) because
Render free tier cannot host ResNet101+VGG16+BiLSTM+CoSH.

## Local run

```powershell
cd webapp
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:8080
```

## Deploy on Render

1. Connect this GitHub repo to Render.
2. Use `webapp/render.yaml` or set:
   - **Root Directory:** `webapp`
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `gunicorn --workers 1 --threads 2 --timeout 180 app:app`
3. Open the Render URL → **Video Detection** tab.

## API

- `GET /` — UI
- `GET /api/health` — liveness
- `POST /api/analyze` — multipart field `file` (video or image) → JSON stages + verdict
