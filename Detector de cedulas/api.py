from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import base64
import re
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# --- RUTAS ---
# Adjust this path if necessary to match the user's file structure
MODEL_PATH = "D:/LENGUAJE DE PROGRAMACIÓN/React/Detector de cedulas/runs/detect/train3/weights/best.pt"

# Load Model
try:
    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- OCR FUNCTIONS ---
def procesar_numero(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    adjusted = cv2.convertScaleAbs(scaled, alpha=1.5, beta=0)
    binary = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 19, 9)
    return binary

def procesar_numero_suave(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(scaled, (3,3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def procesar_texto(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    adjusted = cv2.convertScaleAbs(scaled, alpha=1.3, beta=10)
    binary = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 25, 12)
    final = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    return final

def detectar_fecha_nacimiento(img, ref_box=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    procesamientos = [
        ("Bilateral", cv2.bilateralFilter(gray, 11, 17, 17)),
        ("Original", gray),
        ("Threshold", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    ]
    date_pattern = r'\b\d{2}[/.\s-]\d{2}[/.\s-]\d{4}\b'
    found_dates = []
    custom_config = r'--oem 3 --psm 6'

    for _, img_proc in procesamientos:
        try:
            ocr_data = pytesseract.image_to_data(img_proc, output_type=Output.DICT, config=custom_config)
            for i in range(len(ocr_data['text'])):
                t = ocr_data['text'][i].strip()
                match = re.search(date_pattern, t)
                if match:
                    t_str = match.group(0)
                    t_clean = re.sub(r'[-.\s]', '/', t_str)
                    try:
                        dt = datetime.strptime(t_clean, "%d/%m/%Y")
                        if 1900 < dt.year < datetime.now().year:
                            found_dates.append((dt, t_clean))
                    except:
                        continue
            if found_dates: break
        except Exception:
            pass

    if found_dates:
        birth_date = min(found_dates, key=lambda x: x[0])
        return birth_date[1]

    if ref_box:
        nx1, ny1, nx2, ny2 = ref_box
        h, w, _ = img.shape
        roi_y1 = min(h, ny2 + 5)
        roi_y2 = min(h, ny2 + 200)
        roi_x1 = 0
        roi_x2 = w
        roi_local = img[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi_local.size > 0:
            gray_local = cv2.cvtColor(roi_local, cv2.COLOR_BGR2GRAY)
            scaled = cv2.resize(gray_local, None, fx=2, fy=2)
            _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            configs = [r'--oem 3 --psm 6', r'--oem 3 --psm 11', r'--oem 3 --psm 4']
            
            for config in configs:
                try:
                    d = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=config)
                    for i in range(len(d['text'])):
                        t = d['text'][i].strip()
                        match = re.search(date_pattern, t)
                        if match:
                            t_clean = match.group(0)
                            t_clean = re.sub(r'[-.\s]', '/', t_clean)
                            try:
                                dt = datetime.strptime(t_clean, "%d/%m/%Y")
                                if 1900 < dt.year < datetime.now().year:
                                    return t_clean
                            except:
                                pass
                except Exception:
                    pass
    return None

def limpiar_texto_basico(text):
    text = re.sub(r'\b(NOMBRES|APELLIDOS|NOMBRE|APELLIDO)\b', '', text)
    text = re.sub(' +', ' ', text).strip()
    return text

NOMBRES_COMUNES = ["ALEJANDRO", "JOSE", "MARIA", "JUAN", "CARLOS", "LUIS", "ANA", "MIGUEL", "ANGEL", "FRANCISCO", "JESUS", "PEDRO", "RAFAEL", "GABRIEL", "ANDRES", "DANIEL", "DAVID", "MANUEL", "ANTONIO", "JAVIER", "ROSA", "CARMEN", "ELIZABETH", "JENNIFER"]
APELLIDOS_COMUNES = ["VILLA", "MARTINEZ", "PEREZ", "GARCIA", "RODRIGUEZ", "GONZALEZ", "HERNANDEZ", "LOPEZ", "SANCHEZ", "RAMIREZ", "FLORES", "TORRES", "DIAZ", "VASQUEZ", "CASTRO", "ROMERO", "SUAREZ", "ALVAREZ"]

def separar_palabras_pegadas(texto, diccionario):
    texto_upper = texto.upper()
    for palabra in diccionario:
        if palabra in texto_upper:
            texto_upper = texto_upper.replace(palabra, f" {palabra} ")
    return re.sub(r'\s+', ' ', texto_upper).strip()

def limpiar_apellido(texto):
    if texto and texto.startswith("W"):
        texto = "V" + texto[1:]
    texto = separar_palabras_pegadas(texto, APELLIDOS_COMUNES)
    return texto

@app.route('/extract', methods=['POST'])
def extract_data():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    image_data = data.get('image')

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image
    try:
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    # Run YOLO detection
    results = model(img)[0]
    
    extracted = {
        "firstName": "",
        "lastName": "",
        "cedula": "",
        "dob": ""
    }
    
    detecciones_clases = {}

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_name = results.names[int(class_id)]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if class_name == "Nombre":
            detecciones_clases["Nombre"] = (x1, y1, x2, y2)
        elif class_name == "Apellido":
            detecciones_clases["Apellido"] = (x1, y1, x2, y2)

        if class_name != "Foto":
            h, w, _ = img.shape
            pad_x = 15 
            pad_y = 5  
            roi = img[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]

            roi_ocr = None
            config = r'--oem 3 --psm 7'

            if class_name == "Numero":
                roi_ocr = procesar_numero(roi)
                config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=VvEe0123456789.-'
            else:
                roi_ocr = procesar_texto(roi)
                config = r'--oem 3 --psm 7 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÑ\ '

            try:
                texto = pytesseract.image_to_string(roi_ocr, config=config).strip()
                
                if class_name == "Numero" and not texto:
                    roi_ocr = procesar_numero_suave(roi)
                    texto = pytesseract.image_to_string(roi_ocr, config=config).strip()

                texto = re.sub(r'\s+', ' ', texto) 
                texto = limpiar_texto_basico(texto)

                if class_name == "Apellido":
                    texto = limpiar_apellido(texto)
                    extracted["lastName"] = texto
                elif class_name == "Nombre":
                    texto = separar_palabras_pegadas(texto, NOMBRES_COMUNES)
                    extracted["firstName"] = texto
                elif class_name == "Numero":
                    extracted["cedula"] = texto.replace(" ", "")

            except Exception as e:
                print(f"OCR Error for {class_name}: {e}")

    # Detect Date of Birth
    ref_nombre = detecciones_clases.get("Nombre")
    dob = detectar_fecha_nacimiento(img, ref_box=ref_nombre)
    if dob:
        # Convert DD/MM/YYYY to YYYY-MM-DD for HTML date input
        try:
            dt_obj = datetime.strptime(dob, "%d/%m/%Y")
            extracted["dob"] = dt_obj.strftime("%Y-%m-%d")
        except:
            extracted["dob"] = dob # Fallback

    return jsonify(extracted)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
