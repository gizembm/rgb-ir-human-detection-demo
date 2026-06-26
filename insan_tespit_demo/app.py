import os
import time
import base64
import numpy as np
import cv2
import torch
from flask import Flask, render_template, request, jsonify
from ensemble_boxes import weighted_boxes_fusion

app = Flask(__name__)

# =========================================================
# MODEL YOLLARI
# =========================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

MODEL_PATHS = {
    "YOLOv5s IR":   os.path.join(MODEL_DIR, "yolov5s_ir.pt"),
    "YOLOv5s RGB":  os.path.join(MODEL_DIR, "yolov5s_rgb.pt"),
    "Early Fusion": os.path.join(MODEL_DIR, "ef_gray_ir_blend.pt"),
    "YOLOv26s IR":  os.path.join(MODEL_DIR, "yolov26s_ir.pt"),
    "YOLOv26s RGB": os.path.join(MODEL_DIR, "yolov26s_rgb.pt"),
    "EF Fine-Tuned (NII-CU)": os.path.join(MODEL_DIR, "ef_finetuned_niicu.pt"),
}

# Adaptif sistem parametreleri
LOW_THR      = 60
HIGH_THR     = 90
WBF_WEIGHTS  = [1.0, 2.0]
WBF_IOU_THR  = 0.5
WBF_SKIP_THR = 0.35

# Model performans bilgileri (LLVIP test seti sonuçları)
MODEL_INFO = {
    "YOLOv5s IR": {
        "precision": 0.954, "recall": 0.902, "map5": 0.959, "map595": 0.626,
        "params": "7.0M", "gflops": "15.8", "giris": "IR görüntüsü",
        "aciklama": (
            "LLVIP termal IR veri seti üzerinde 60 epoch eğitilen baseline model. "
            "Gece ve düşük ışık koşullarında insan ısısını tespit ederek çalışır; aydınlatmadan bağımsızdır. "
            "RGB modeline kıyasla düşük ışıkta belirgin üstünlük sağlar (mAP@0.5: 0.959 vs 0.902)."
        )
    },
    "YOLOv5s RGB": {
        "precision": 0.898, "recall": 0.838, "map5": 0.902, "map595": 0.502,
        "params": "7.0M", "gflops": "15.8", "giris": "RGB görüntüsü",
        "aciklama": (
            "Görünür spektrum (RGB) görüntüler üzerinde eğitilen baseline model. "
            "Gündüz ve iyi aydınlatmalı ortamlarda güçlü performans sunar. "
            "Gece veya düşük ışık koşullarında performans belirgin biçimde düşer; bu koşullarda IR veya füzyon modelleri tercih edilmelidir."
        )
    },
    "Early Fusion": {
        "precision": 0.950, "recall": 0.919, "map5": 0.969, "map595": 0.650,
        "params": "7.0M", "gflops": "15.8", "giris": "RGB + IR görüntüsü",
        "aciklama": (
            "RGB ve IR görüntüleri giriş seviyesinde birleştiren özgün üç kanallı erken füzyon modeli. "
            "Kanal yapısı: [Gray | IR | 0.4xGray+0.6xIR]. Ablasyon çalışmasında en yüksek genel kutu hassasiyetini (mAP@0.5:0.95=0.650) elde etmiştir. "
            "Tek model kullandığı için geç füzyona kıyasla daha düşük işlem maliyeti sunar."
        )
    },
    "YOLOv26s IR": {
        "precision": 0.966, "recall": 0.890, "map5": 0.952, "map595": 0.650,
        "params": "9.5M", "gflops": "20.5", "giris": "IR görüntüsü",
        "aciklama": (
            "Daha büyük mimari (9.5M parametre, 20.5 GFLOPs) ile termal IR üzerinde eğitilen model. "
            "Baseline modeller arasında en yüksek mAP@0.5:0.95 değerini (0.650) elde etmiştir. "
            "YOLOv5s'e kıyasla daha yüksek hesaplama maliyeti taşır; edge deployment için YOLOv5s daha uygundur."
        )
    },
    "YOLOv26s RGB": {
        "precision": 0.948, "recall": 0.907, "map5": 0.954, "map595": 0.590,
        "params": "9.5M", "gflops": "20.5", "giris": "RGB görüntüsü",
        "aciklama": (
            "YOLOv26s mimarisiyle görünür spektrum üzerinde eğitilen model. "
            "YOLOv5s RGB ile karşılaştırmalı mimari analizi için kullanılmıştır. "
            "mAP@0.5:0.95 açısından YOLOv5s RGB'yi geride bırakır (0.590 vs 0.502), ancak hesaplama maliyeti daha yüksektir."
        )
    },
    "Adaptif": {
        "precision": 0.840, "recall": 0.951, "map5": 0.945, "map595": 0.615,
        "params": "7.0M×2", "gflops": "15.8×2", "giris": "RGB + IR görüntüsü",
        "aciklama": (
            "Sahne parlaklığına göre otomatik model seçimi yapan sistem. "
            "Parlaklık < 60 ise IR modeli, 60-90 arasında WBF tabanlı geç füzyon, 90 üzerinde RGB modeli devreye girer. "
            "LLVIP test setinde görüntülerin %68'i IR, %25'i füzyon, %7'si RGB moduyla işlenmiştir. "
            "Insan kaçırmamanın kritik olduğu senaryolarda yüksek recall (0.951) sunar."
        )
    },
    "EF Fine-Tuned (NII-CU)": {
        "precision": 0.984, "recall": 0.971, "map5": 0.986, "map595": 0.647,
        "params": "7.0M", "gflops": "15.8", "giris": "RGB + IR görüntüsü",
        "aciklama": (
            "LLVIP Early Fusion modelinin NII-CU IHA veri setine 30 epoch fine-tune edilmiş hali. "
            "Doğrudan aktarımda mAP@0.5=0.141 iken fine-tuning sonrası 0.986'ya yükselmiştir. "
            "Drone/IHA bakış açısından çekilen görüntülerde ve farklı termal kamera karakteristiklerinde daha iyi genelleme sağlar."
        )
    },
}

# =========================================================
# MODEL ÖNBELLEKLEME
# =========================================================
_model_cache = {}

def load_model(key):
    if key not in _model_cache:
        import pathlib
        # Windows'ta PosixPath → WindowsPath düzeltmesi
        pathlib.PosixPath = pathlib.WindowsPath

        path = MODEL_PATHS[key]
        print(f"[Model] {key} yükleniyor: {path}")

        if "YOLOv26s" in key:
            from ultralytics import YOLO as UltralyticsYOLO
            _model_cache[key] = UltralyticsYOLO(path)
        else:
            m = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=path, force_reload=False
            )
            _model_cache[key] = m

        print(f"[Model] {key} yüklendi.")
    return _model_cache[key]


# =========================================================
# GÖRÜNTÜ İŞLEME YARDIMCI FONKSİYONLARI
# =========================================================
def normalize_uint8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def decode_image(file_bytes):
    """Flask'tan gelen dosya byte'larını OpenCV BGR görüntüsüne çevirir."""
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_image(img_bgr):
    """BGR görüntüyü base64 string'e çevirir (HTML'de göstermek için)."""
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def create_early_fusion(rgb_bgr, ir_bgr):
    """
    RGB ve IR görüntülerden 3 kanallı erken füzyon görüntüsü oluşturur.
    Kanal yapısı (eğitimdeki ile birebir aynı):
      Kanal 1: RGB → Gri ton
      Kanal 2: IR → Gri ton
      Kanal 3: 0.4×Gray + 0.6×IR (blend)
    """
    gray  = normalize_uint8(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY))
    ir    = normalize_uint8(cv2.cvtColor(ir_bgr,  cv2.COLOR_BGR2GRAY))

    if ir.shape != gray.shape:
        ir = cv2.resize(ir, (gray.shape[1], gray.shape[0]))

    blend = normalize_uint8(cv2.addWeighted(gray, 0.4, ir, 0.6, 0))
    return np.stack([gray, ir, blend], axis=-1)


def get_preds_v5(model, img_bgr, conf):
    """YOLOv5 modelinden normalize koordinatlarda tahmin alır."""
    model.conf = conf
    results    = model(img_bgr)
    preds      = results.xyxy[0].cpu().numpy()
    h, w       = img_bgr.shape[:2]

    boxes, scores, labels = [], [], []
    for x1, y1, x2, y2, score, cls in preds:
        boxes.append([x1/w, y1/h, x2/w, y2/h])
        scores.append(float(score))
        labels.append(int(cls))
    return boxes, scores, labels


def get_preds_v26(model, img_bgr, conf):
    """YOLOv26s modelinden normalize koordinatlarda tahmin alır."""
    results = model.predict(img_bgr, conf=conf, verbose=False)
    h, w    = img_bgr.shape[:2]

    boxes, scores, labels = [], [], []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1/w, y1/h, x2/w, y2/h])
            scores.append(float(box.conf[0]))
            labels.append(int(box.cls[0]))
    return boxes, scores, labels


def get_preds(model, img_bgr, conf, key):
    """Model tipine göre doğru tahmin fonksiyonunu çağırır."""
    if "YOLOv26s" in key:
        return get_preds_v26(model, img_bgr, conf)
    return get_preds_v5(model, img_bgr, conf)


def draw_boxes(img_bgr, boxes_norm, scores, color=(0, 255, 0)):
    """Tespit kutularını görüntüye çizer."""
    h, w = img_bgr.shape[:2]
    out  = img_bgr.copy()
    for i, (box, score) in enumerate(zip(boxes_norm, scores)):
        x1, y1 = int(box[0]*w), int(box[1]*h)
        x2, y2 = int(box[2]*w), int(box[3]*h)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"#{i+1} {score:.2f}",
                    (x1, max(y1-6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def add_overlay(img_bgr, label, n_det, brightness=None):
    """Görüntünün üstüne bilgi şeridi ekler."""
    out  = img_bgr.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0,0), (w,46), (15,15,15), -1)
    cv2.putText(out, f"{label}  |  Tespit: {n_det} kisi",
                (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)
    if brightness is not None:
        cv2.putText(out, f"Parlaklik: {brightness:.1f}",
                    (8,40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1)
    return out


def estimate_brightness(rgb_bgr):
    return float(np.mean(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)))


# =========================================================
# ANA TESPİT FONKSİYONU
# =========================================================
def detect(model_name, conf, rgb_bgr, ir_bgr):
    """
    Model seçimine göre tespit yapar.
    Döndürür: (result_img_bgr, detections_list, elapsed_ms, brightness, selected_mode)
    """
    brightness    = None
    selected_mode = model_name

    # ── IR MODELLERİ ──────────────────────────────────────
    if model_name in ["YOLOv5s IR", "YOLOv26s IR"]:
        if ir_bgr is None:
            raise ValueError("IR görüntüsü gerekli.")
        model = load_model(model_name)
        t0    = time.time()
        boxes, scores, _ = get_preds(model, ir_bgr, conf, model_name)
        ms    = (time.time() - t0) * 1000
        out   = draw_boxes(ir_bgr, boxes, scores, (0, 165, 255))
        out   = add_overlay(out, model_name, len(scores))

    # ── RGB MODELLERİ ─────────────────────────────────────
    elif model_name in ["YOLOv5s RGB", "YOLOv26s RGB"]:
        if rgb_bgr is None:
            raise ValueError("RGB görüntüsü gerekli.")
        model = load_model(model_name)
        t0    = time.time()
        boxes, scores, _ = get_preds(model, rgb_bgr, conf, model_name)
        ms    = (time.time() - t0) * 1000
        out   = draw_boxes(rgb_bgr, boxes, scores, (255, 80, 0))
        out   = add_overlay(out, model_name, len(scores))

    # ── ERKEN FÜZYON ──────────────────────────────────────
    elif model_name in ["Early Fusion", "EF Fine-Tuned (NII-CU)"]:
        if rgb_bgr is None or ir_bgr is None:
            raise ValueError("Early Fusion için hem RGB hem IR gerekli.")
        fused = create_early_fusion(rgb_bgr, ir_bgr)
        model = load_model(model_name)
        t0    = time.time()
        boxes, scores, _ = get_preds(model, fused, conf, "Early Fusion")
        ms    = (time.time() - t0) * 1000
        out   = draw_boxes(rgb_bgr, boxes, scores, (0, 220, 0))
        out   = add_overlay(out, model_name, len(scores))

    # ── ADAPTİF SİSTEM ────────────────────────────────────
    elif model_name == "Adaptif":
        if rgb_bgr is None or ir_bgr is None:
            raise ValueError("Adaptif sistem için hem RGB hem IR gerekli.")
        brightness = estimate_brightness(rgb_bgr)

        t0 = time.time()

        if brightness < LOW_THR:
            model = load_model("YOLOv5s IR")
            boxes, scores, _ = get_preds(model, ir_bgr, conf, "YOLOv5s IR")
            out   = draw_boxes(ir_bgr, boxes, scores, (0, 165, 255))
            selected_mode = f"IR (Parlaklık {brightness:.1f} < {LOW_THR})"

        elif brightness > HIGH_THR:
            model = load_model("YOLOv5s RGB")
            boxes, scores, _ = get_preds(model, rgb_bgr, conf, "YOLOv5s RGB")
            out   = draw_boxes(rgb_bgr, boxes, scores, (255, 80, 0))
            selected_mode = f"RGB (Parlaklık {brightness:.1f} > {HIGH_THR})"

        else:
            rgb_model = load_model("YOLOv5s RGB")
            ir_model  = load_model("YOLOv5s IR")
            b_rgb, s_rgb, l_rgb = get_preds(rgb_model, rgb_bgr, conf, "YOLOv5s RGB")
            b_ir,  s_ir,  l_ir  = get_preds(ir_model,  ir_bgr,  conf, "YOLOv5s IR")
            boxes, scores, _ = weighted_boxes_fusion(
                [b_rgb, b_ir], [s_rgb, s_ir], [l_rgb, l_ir],
                weights=WBF_WEIGHTS, iou_thr=WBF_IOU_THR,
                skip_box_thr=WBF_SKIP_THR
            )
            boxes  = list(boxes)
            scores = list(scores)
            out    = draw_boxes(rgb_bgr, boxes, scores, (0, 220, 220))
            selected_mode = f"WBF Füzyon ({LOW_THR}≤P≤{HIGH_THR}, P={brightness:.1f})"

        ms  = (time.time() - t0) * 1000
        out = add_overlay(out, f"Adaptif → {selected_mode.split('(')[0].strip()}",
                          len(scores), brightness)

    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    # Tespit detayları
    h, w = out.shape[:2]
    detections = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        detections.append({
            "id":    i + 1,
            "score": round(float(score), 3),
            "box": {
                "x1": round(box[0]*w), "y1": round(box[1]*h),
                "x2": round(box[2]*w), "y2": round(box[3]*h),
            }
        })

    return out, detections, round(ms, 1), brightness, selected_mode


# =========================================================
# FLASK ROUTE'LARI
# =========================================================
@app.route("/")
def index():
    return render_template("index.html", model_info=MODEL_INFO)


@app.route("/model_info/<model_name>")
def model_info(model_name):
    """Model seçimi değişince bilgi kartını döndürür."""
    info = MODEL_INFO.get(model_name, {})
    return jsonify(info)


@app.route("/detect", methods=["POST"])
def detect_route():
    """Tespit isteğini işler ve sonuçları döndürür."""
    model_name = request.form.get("model", "Early Fusion")
    conf       = float(request.form.get("conf", 0.25))

    # Görüntüleri oku
    rgb_bgr = ir_bgr = None

    if "rgb_image" in request.files and request.files["rgb_image"].filename:
        rgb_bgr = decode_image(request.files["rgb_image"].read())

    if "ir_image" in request.files and request.files["ir_image"].filename:
        ir_bgr = decode_image(request.files["ir_image"].read())

    try:
        out_img, detections, elapsed_ms, brightness, selected_mode = detect(
            model_name, conf, rgb_bgr, ir_bgr
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Model performans bilgisi
    info = MODEL_INFO.get(model_name, {})

    return jsonify({
        "image":         encode_image(out_img),
        "detections":    detections,
        "elapsed_ms":    elapsed_ms,
        "brightness":    round(brightness, 1) if brightness is not None else None,
        "selected_mode": selected_mode,
        "model_info":    info,
        "n_det":         len(detections),
        "avg_conf":      round(float(np.mean([d["score"] for d in detections])), 3)
                         if detections else 0,
        "max_conf":      round(max([d["score"] for d in detections]), 3)
                         if detections else 0,
    })


if __name__ == "__main__":
    print("=" * 50)
    print("İnsan Tespit Demo Uygulaması")
    print("http://localhost:5000 adresinde çalışıyor")
    print("=" * 50)

    # Modellerin var olup olmadığını kontrol et
    for name, path in MODEL_PATHS.items():
        status = "✓" if os.path.exists(path) else "✗ BULUNAMADI"
        print(f"  {name}: {status}")

    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=5000)