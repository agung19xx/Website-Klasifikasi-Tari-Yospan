import os
import io
import base64
import numpy as np

from flask import Flask, render_template, request, jsonify

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# KONFIGURASI FLASK
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "mobilenetv2_yospan_best.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "labels.txt"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("==============================================")
print("MEMUAT MODEL MOBILENETV2 YOSPAN")
print("==============================================")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model berhasil dimuat.")
print("Model:", MODEL_PATH)


# ============================================================
# LOAD LABEL
# ============================================================

if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r", encoding="utf-8") as file:
        labels = [
            line.strip()
            for line in file
            if line.strip()
        ]
else:
    labels = [
        "bukan_yospan",
        "gale_gale",
        "jef",
        "pacul_tiga",
        "pancar",
        "seka"
    ]

print("Label:", labels)
print("Jumlah kelas:", len(labels))
print("==============================================")
print("MODEL SIAP DIGUNAKAN")
print("==============================================")


# ============================================================
# NAMA GERAKAN
# ============================================================

DISPLAY_NAMES = {
    "bukan_yospan": "Bukan Yospan",
    "gale_gale": "Gale-gale",
    "gale-gale": "Gale-gale",
    "gale gale": "Gale-gale",
    "jef": "Jef",
    "pacul_tiga": "Pacul Tiga",
    "pacul-tiga": "Pacul Tiga",
    "pacul tiga": "Pacul Tiga",
    "pancar": "Pancar",
    "seka": "Seka"
}


def format_label(label):
    key = label.strip().lower()
    return DISPLAY_NAMES.get(
        key,
        label.replace("_", " ").title()
    )


# ============================================================
# PREPROCESSING GAMBAR
# ============================================================


def preprocess_image(image_bytes):
    """Memproses gambar langsung dari memory."""

    image = load_img(
        io.BytesIO(image_bytes),
        target_size=(224, 224)
    )

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    return image_array


# ============================================================
# PREDIKSI
# ============================================================


def predict_image(image_bytes):
    image_array = preprocess_image(image_bytes)

    prediction = model.predict(
        image_array,
        verbose=0
    )

    if prediction.ndim != 2:
        raise ValueError(
            "Output model tidak memiliki format yang sesuai."
        )

    if prediction.shape[0] < 1:
        raise ValueError(
            "Model tidak menghasilkan hasil prediksi."
        )

    if prediction.shape[1] != len(labels):
        raise ValueError(
            f"Jumlah output model ({prediction.shape[1]}) "
            f"tidak sesuai dengan jumlah label ({len(labels)})."
        )

    predicted_index = int(np.argmax(prediction[0]))
    predicted_label = labels[predicted_index]
    confidence = float(prediction[0][predicted_index])

    is_yospan = (
        predicted_label.strip().lower() != "bukan_yospan"
    )

    return {
        "label": format_label(predicted_label),
        "confidence": confidence,
        "is_yospan": is_yospan
    }


# ============================================================
# GAMBAR UNTUK FRONTEND
# ============================================================


def create_image_data_url(image_bytes, mimetype):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mimetype};base64,{encoded_image}"


# ============================================================
# HALAMAN WEBSITE
# ============================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tentang")
def tentang():
    return render_template("tentang.html")


@app.route("/klasifikasi")
def klasifikasi():
    return render_template("klasifikasi.html")


# ============================================================
# API /predict
# ============================================================


@app.route("/predict", methods=["POST"])
def predict():
    """
    Menerima gambar dan melakukan klasifikasi.

    PENTING:
    File upload TIDAK PERNAH disimpan menggunakan file.save().
    Gambar dibaca langsung ke memory karena filesystem Vercel
    bersifat read-only.
    """

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "status": "error",
            "message": "Gambar belum dipilih."
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({
            "success": False,
            "status": "error",
            "message": "Gambar belum dipilih."
        }), 400

    try:
        # ====================================================
        # PENTING: BACA KE MEMORY, JANGAN file.save()
        # ====================================================
        image_bytes = file.read()

        if not image_bytes:
            return jsonify({
                "success": False,
                "status": "error",
                "message": "File gambar kosong."
            }), 400

        # ====================================================
        # PREDIKSI
        # ====================================================
        result = predict_image(image_bytes)

        # ====================================================
        # DATA URL UNTUK PREVIEW FRONTEND
        # Tidak membuat file di static/uploads.
        # ====================================================
        mimetype = file.mimetype or "image/jpeg"

        image_data_url = create_image_data_url(
            image_bytes,
            mimetype
        )

        return jsonify({
            "success": True,
            "status": "success",
            "label": result["label"],
            "confidence": result["confidence"],
            "is_yospan": result["is_yospan"],
            "image": image_data_url
        }), 200

    except Exception as error:
        print("==============================================")
        print("ERROR PREDIKSI")
        print("Tipe error:", type(error).__name__)
        print("Pesan:", str(error))
        print("==============================================")

        return jsonify({
            "success": False,
            "status": "error",
            "message": f"Terjadi kesalahan saat melakukan klasifikasi: {str(error)}"
        }), 500


# ============================================================
# SERVER LOCAL
# ============================================================


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
    