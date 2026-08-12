import os
import numpy as np

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ============================================================
# KONFIGURASI FLASK
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "yospan_5class_finetuned_best.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "labels.txt"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# LOAD MODEL CNN
# ============================================================

print("==============================================")
print("MEMUAT MODEL CNN")
print("==============================================")

model = load_model(MODEL_PATH)

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
        "gale_gale",
        "jef",
        "pacul_tiga",
        "pancar",
        "seka"
    ]

print("Label:", labels)
print("==============================================")


# ============================================================
# NAMA GERAKAN UNTUK DITAMPILKAN DI WEBSITE
# ============================================================

DISPLAY_NAMES = {
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


# ============================================================
# FUNGSI MENGUBAH LABEL MENJADI NAMA YANG RAPI
# ============================================================

def format_label(label):

    key = label.strip().lower()

    return DISPLAY_NAMES.get(
        key,
        label.replace("_", " ").title()
    )


# ============================================================
# FUNGSI PREDIKSI
# ============================================================

def predict_image(filepath):

    # Ukuran input model
    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Normalisasi
    img_array = img_array / 255.0

    # Prediksi
    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(prediction[0])
    )

    predicted_label = labels[predicted_index]

    return format_label(predicted_label)


# ============================================================
# HALAMAN BERANDA
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# HALAMAN TENTANG
# ============================================================

@app.route("/tentang")
def tentang():

    return render_template(
        "tentang.html"
    )


# ============================================================
# HALAMAN KLASIFIKASI
# ============================================================

@app.route("/klasifikasi")
def klasifikasi():

    return render_template(
        "klasifikasi.html"
    )


# ============================================================
# API / PROSES KLASIFIKASI
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Periksa apakah file dikirim
    if "image" not in request.files:

        return jsonify({
            "success": False,
            "message": "Gambar belum dipilih."
        })

    file = request.files["image"]

    # Periksa nama file
    if file.filename == "":

        return jsonify({
            "success": False,
            "message": "Gambar belum dipilih."
        })

    # Amankan nama file
    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Simpan gambar
    file.save(filepath)

    try:

        # Jalankan prediksi
        predicted_label = predict_image(
            filepath
        )

        # URL gambar
        image_url = url_for(
            "static",
            filename=f"uploads/{filename}"
        )

        return jsonify({

            "success": True,

            "label": predicted_label,

            "image": image_url

        })

    except Exception as error:

        print("==============================================")
        print("ERROR PREDIKSI")
        print(error)
        print("==============================================")

        return jsonify({

            "success": False,

            "message":
                "Terjadi kesalahan saat melakukan klasifikasi."

        })


# ============================================================
# MENJALANKAN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )