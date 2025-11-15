import os
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.lib import colors
import io, base64, uuid, tempfile
from datetime import datetime
import numpy as np
import tempfile
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import firebase_admin
from firebase_admin import credentials, firestore
from urllib.parse import urljoin
from twilio.rest import Client
import requests

# ✅ TensorFlow lightweight settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["DEBUG"] = True
CORS(app)
# Initialize Firebase Admin SDK
service_account_json = os.getenv("serviceAccountKey")

if not service_account_json:
    raise ValueError("❌ Firebase serviceAccountKey not found in environment variables.")

# Write to a temporary file so Firebase can load it correctly
with open("firebase_key.json", "w") as f:
    f.write(service_account_json)

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
# ✅ Load TensorFlow Model
model = None
def load_model():
    global model
    if model is None:
        print("🧠 Loading TensorFlow model...")
        model_path = os.getenv("MODEL_PATH", "model.keras")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    return model

CLASSES = ['1. Eczema 1677', '10. Warts Molluscum and other Viral Infections - 2103', '2. Melanoma 15.75k', '3. Atopic Dermatitis - 1.25k', '4. Basal Cell Carcinoma (BCC) 3323', '5. Melanocytic Nevi (NV) - 7970', '6. Benign Keratosis-like Lesions (BKL) 2624', '7. Psoriasis pictures Lichen Planus and related diseases - 2k', '8. Seborrheic Keratoses and other Benign Tumors - 1.8k', '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k']

DISEASE_INFO = {
    "1. Eczema 1677": {
        "name": "Eczema",
        "description": [
            "A chronic condition causing itchy, inflamed, and dry skin.",
            "Often triggered by allergens, stress, or environmental factors."
        ],
        "medication": "Topical corticosteroids, antihistamines for itching, and moisturizers to reduce dryness.",
        "diet": "Omega-3-rich foods (like fish and flaxseeds), avoid processed foods and excessive dairy."
    },
    "2. Melanoma 15.75k": {
        "name": "Melanoma",
        "description": [
            "A serious form of skin cancer that develops from pigment-producing cells (melanocytes).",
            "Early detection is crucial to prevent spreading."
        ],
        "medication": "Surgical removal, immunotherapy, or targeted therapy. Urgent dermatologist consultation required.",
        "diet": "High-antioxidant diet including berries, leafy greens, and vitamin D-rich foods."
    },
    "3. Atopic Dermatitis - 1.25k": {
        "name": "Atopic Dermatitis",
        "description": [
            "A chronic inflammatory skin condition characterized by itchy, inflamed skin.",
            "Usually starts in childhood and may flare up periodically."
        ],
        "medication": "Moisturizers, steroid creams, or calcineurin inhibitors. Avoid irritants and stress.",
        "diet": "Probiotic-rich foods (yogurt, kefir) and foods high in omega-3s to reduce inflammation."
    },
    "4. Basal Cell Carcinoma (BCC) 3323": {
        "name": "Basal Cell Carcinoma (BCC)",
        "description": [
            "A slow-growing type of skin cancer appearing as pearly or waxy bumps.",
            "Usually caused by long-term UV exposure."
        ],
        "medication": "Surgical excision, topical treatments, or Mohs surgery. Regular dermatologist checkups are advised.",
        "diet": "Include vitamin E, green tea, and antioxidant-rich fruits."
    },
    "5. Melanocytic Nevi (NV) - 7970": {
        "name": "Melanocytic Nevi (NV)",
        "description": [
            "Commonly known as moles; usually harmless pigment spots on the skin.",
            "Monitor for any changes in size, color, or shape."
        ],
        "medication": "No treatment required unless suspicious. Surgical removal if necessary.",
        "diet": "Healthy balanced diet with fruits and vegetables; avoid excessive sun exposure."
    },
    "6. Benign Keratosis-like Lesions (BKL) 2624": {
        "name": "Benign Keratosis-like Lesions (BKL)",
        "description": [
            "Non-cancerous skin growths that appear as rough or crusty patches.",
            "Common in older adults and may resemble warts or sun damage."
        ],
        "medication": "Laser therapy, cryotherapy, or minor surgery for cosmetic reasons.",
        "diet": "Balanced diet; include foods rich in vitamins A and E for skin health."
    },
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": {
        "name": "Psoriasis, Lichen Planus, and Related Diseases",
        "description": [
            "Autoimmune conditions causing thick, scaly patches of skin.",
            "May flare up due to stress, infections, or certain medications."
        ],
        "medication": "Topical corticosteroids, phototherapy, or biologic drugs depending on severity.",
        "diet": "Anti-inflammatory foods such as turmeric, salmon, and leafy greens."
    },
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": {
        "name": "Seborrheic Keratoses and Other Benign Tumors",
        "description": [
            "Common, noncancerous skin growths that appear waxy or wart-like.",
            "Usually harmless but can be removed for cosmetic reasons."
        ],
        "medication": "Cryotherapy, curettage, or laser treatment if removal desired.",
        "diet": "Maintain a healthy diet; ensure proper hydration and vitamin E intake."
    },
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": {
        "name": "Tinea, Ringworm, Candidiasis, and Other Fungal Infections",
        "description": [
            "Caused by fungal organisms that thrive in moist environments.",
            "Common symptoms include itching, redness, and circular rashes."
        ],
        "medication": "Topical or oral antifungal medications (clotrimazole, fluconazole). Keep affected area dry.",
        "diet": "Low sugar diet, include garlic and probiotics to support antifungal defense."
    },
    "10. Warts Molluscum and other Viral Infections - 2103": {
        "name": "Warts, Molluscum, and Other Viral Infections",
        "description": [
            "Caused by viral infections such as HPV or Molluscum contagiosum.",
            "Appear as small bumps or lesions, sometimes contagious through contact."
        ],
        "medication": "Cryotherapy, salicylic acid treatments, or topical antivirals under medical supervision.",
        "diet": "Boost immunity with vitamin C and zinc-rich foods (citrus, spinach, pumpkin seeds)."
    }
}

from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)  # <-- critical step
    return np.expand_dims(img_array, axis=0)

from tensorflow.keras.applications.resnet import preprocess_input

def preprocess_image(image_file):
    """
    Preprocess image exactly like ResNet training preprocessing.
    """
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)  # 👈 critical: ResNet normalization
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise

@app.route("/")
def home():
    return "✅ SmartSkinAI Backend is Running!"

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        img_input = preprocess_image(image_file)
        # Load model lazily
        model_instance = load_model()
        preds = model_instance.predict(img_input)
        pred_index = int(np.argmax(preds))
        confidence = float(np.max(preds))
        pred_class = CLASSES[pred_index]
        print(f"Prediction: {pred_class}, Confidence: {confidence:.2f}", flush=True)

        print("🧩 Predicted Class:", pred_class)
        print("📚 Available Disease Info Keys:", list(DISEASE_INFO.keys()))

        
    

        disease_info = DISEASE_INFO.get(pred_class)

        if disease_info is None:
            print(f"⚠️ No match found for: {pred_class}")
            return jsonify({
                "disease": pred_class,
                "description": ["Information not found for this class."],
                "medication": "N/A",
                "diet": "N/A"
            }), 200

        return jsonify({
            "disease": disease_info["name"],
            "description": disease_info["description"],
            "medication": disease_info["medication"],
            "diet": disease_info["diet"]
        })

        
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/download/<filename>")
def download(filename):
    """Serve PDF files from temp directory"""
    try:
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(
            filepath,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/find_doctors', methods=['GET'])
def find_doctors():
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude are required'}), 400

    area_name = "Your current area"
    doctors = []

    try:
        headers = {'User-Agent': 'SmartSkinHealthApp/1.0'}
        geocode_url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
        
        geocode_response = requests.get(geocode_url, headers=headers)
        geocode_response.raise_for_status()
        geocode_data = geocode_response.json()
        
        if geocode_data and 'display_name' in geocode_data:
            area_name = geocode_data.get('display_name')

        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        (
          node["amenity"~"doctors|clinic|hospital"](around:5000,{lat},{lon});
          way["amenity"~"doctors|clinic|hospital"](around:5000,{lat},{lon});
          relation["amenity"~"doctors|clinic|hospital"](around:5000,{lat},{lon});
        );
        out center;
        """
        
        places_response = requests.post(overpass_url, data=overpass_query, headers=headers)
        places_response.raise_for_status()
        places_data = places_response.json()
        
        for place in places_data.get('elements', [])[:10]:
            tags = place.get('tags', {})
            address = f"{tags.get('addr:street', '')} {tags.get('addr:housenumber', '')}".strip()
            if not address:
                address = tags.get('addr:full', 'Address not available')

            doctors.append({
                'name': tags.get('name', 'Medical Facility'),
                'address': address
            })

        return jsonify({
            'area_name': area_name,
            'doctors': doctors
        })

    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return jsonify({'error': 'Failed to fetch data from OpenStreetMap APIs'}), 500

@app.route("/api/save-prescription", methods=["POST"])
def save_prescription():
    try:
        data = request.json
        name = data.get("name")
        age = data.get("age")
        gender = data.get("gender")
        whatsapp = data.get("whatsapp", "")
        disease = data.get("disease")
        description = data.get("description", [])
        medication = data.get("medication", "")
        diet = data.get("diet", "")

        # Validate required fields
        if not all([name, age, gender, disease]):
            return jsonify({"error": "Missing required fields"}), 400

        print(f"📄 Generating PDF for {name}...")

        # 🧾 Generate PDF in memory
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 100
        x_margin, line_height = 60, 22
        max_width = width - 2 * x_margin

        # Header
        p.setFont("Helvetica-Bold", 14)
        p.setFillColor(colors.green)
        p.drawString(x_margin, height - 60, "Derm AI")

        p.setFont("Helvetica-Bold", 16)
        p.setFillColor(colors.darkblue)
        for line in simpleSplit("AI-Based Skin Diagnosis Report", "Helvetica-Bold", 16, max_width):
            p.drawCentredString(width / 2, y, line)
            y -= line_height

        p.setFont("Helvetica", 12)
        p.setFillColor(colors.black)
        y -= 20

        def draw_label(label, value):
            nonlocal y
            p.setFont("Helvetica-Bold", 12)
            p.drawString(x_margin, y, f"{label}: {value}")
            y -= line_height

        def draw_list(label, items):
            nonlocal y
            p.setFont("Helvetica-Bold", 12)
            p.drawString(x_margin, y, f"{label}:")
            y -= line_height
            for item in (items if isinstance(items, list) else [items]):
                p.setFont("Helvetica", 12)
                for line in simpleSplit(f"• {item}", "Helvetica", 12, max_width):
                    p.drawString(x_margin + 15, y, line)
                    y -= line_height
            y -= 10

        # Content
        draw_label("Patient Name", name)
        draw_label("Age", age)
        draw_label("Gender", gender)
        draw_label("Disease", disease)
        draw_list("Description", description)
        draw_list("Medication", medication)
        draw_list("Diet", diet)

        # Footer
        p.setFont("Helvetica-Oblique", 10)
        p.setFillColor(colors.gray)
        p.drawCentredString(width / 2, 40, "Generated by Smart Skin Health | For clinical guidance only")
        p.showPage()
        p.save()

        pdf_bytes = buffer.getvalue()
        buffer.close()
        doc_id = str(uuid.uuid4())

        # ✅ Save to Firestore
        db.collection("predictions").document(doc_id).set({
            "patient": {"name": name, "age": age, "gender": gender, "whatsapp": whatsapp},
            "prediction": disease,
            "description": description,
            "medication": medication,
            "diet": diet,
            "createdAt": datetime.utcnow().isoformat()
        })

        # ✅ Save PDF to /tmp directory
        filename = f"{uuid.uuid4()}.pdf"
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)
        
        print(f"✅ PDF saved to: {filepath}")

        # ✅ Construct download URL
        base_url = os.getenv("BASE_URL", "https://Himavamsi23BCE-backendDL.hf.space")
        pdf_url = f"{base_url}/download/{filename}"

        print(f"📎 PDF URL: {pdf_url}")

        # ✅ Send via WhatsApp (Twilio)
        if whatsapp:
            try:
                account_sid = os.getenv("TWILIO_ACCOUNT_SID")
                auth_token = os.getenv("TWILIO_AUTH_TOKEN")
                whatsapp_from = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
                
                if not account_sid or not auth_token:
                    print("⚠️ Twilio credentials not found, skipping WhatsApp")
                else:
                    from twilio.rest import Client
                    client = Client(account_sid, auth_token)
                    
                    # Fix phone number format
                    if not whatsapp.startswith("whatsapp:"):
                        whatsapp = f"whatsapp:{whatsapp}"
                    
                    message = client.messages.create(
                        from_=whatsapp_from,
                        to=whatsapp,
                        body=f"Hello {name}, here's your AI-generated skin diagnosis report 🩺📄",
                        media_url=[pdf_url]
                    )
                    print(f"✅ WhatsApp message sent to {whatsapp} (SID: {message.sid})")
            except Exception as e:
                print(f"⚠️ WhatsApp sending failed: {e}")
                # Don't fail the whole request if WhatsApp fails

        return jsonify({
            "message": "Prescription saved successfully ✅",
            "id": filename.replace(".pdf", ""),
            "pdf_url": pdf_url
        })

    except Exception as e:
        print(f"❌ Save prescription error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
