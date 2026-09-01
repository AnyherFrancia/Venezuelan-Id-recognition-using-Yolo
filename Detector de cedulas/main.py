from ultralytics import YOLO
import cv2
import pytesseract
import re
import numpy as np
from pytesseract import Output
from datetime import datetime

# --- RUTAS ---
model_path = "D:/LENGUAJE DE PROGRAMACIÓN/React/Detector de cedulas/runs/detect/train3/weights/best.pt"
# image_path = "D:/LENGUAJE DE PROGRAMACIÓN/React/Detector de cedulas/CEDULA2.jpeg"
image_path = "D:/LENGUAJE DE PROGRAMACIÓN/React/Detector de cedulas/data/images/train/1.jpg"

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def procesar_numero(roi):
    """Procesamiento agresivo para el Número"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    adjusted = cv2.convertScaleAbs(scaled, alpha=1.5, beta=0)
    binary = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 19, 9)
    return binary

def procesar_numero_suave(roi):
    """Procesamiento alternativo (Otsu) si el agresivo falla"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # Un pequeño blur ayuda si la imagen tiene ruido
    blurred = cv2.GaussianBlur(scaled, (3,3), 0)
    # Otsu calcula el umbral automaticamente
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def procesar_texto(roi):
    """
    Procesamiento equilibrado para Nombres:
    Busca resaltar letras pero RESPETA los espacios en blanco.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 1. Escalar x3 (Suficiente para texto)
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # 2. Aumentar contraste ligeramente
    adjusted = cv2.convertScaleAbs(scaled, alpha=1.3, beta=10)
    
    # 3. Umbral Adaptativo "Suave"
    # BlockSize: 25 (más grande mira más área, preservando espacios)
    # C: 12 (Un valor más alto limpia más ruido blanco alrededor de las letras)
    binary = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 25, 12)
    
    # 4. Limpieza final de ruido (puntos pequeños)
    # Esto elimina basurita entre palabras sin borrar las letras
    final = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return final

def detectar_fecha_nacimiento(img, ref_box=None):
    """
    Busca patrones de fecha en toda la imagen usando varias estrategias.
    Si se da ref_box (coords de Nombre), busca especificamente debajo de esa zona.
    """
    # Estrategias de preprocesado
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    procesamientos = [
        ("Bilateral", cv2.bilateralFilter(gray, 11, 17, 17)),
        ("Original", gray),
        ("Threshold", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    ]

    # Patrón más flexible: acepta /, - o espacios como separador
    # Ej: 12/05/2000, 12-05-2000, 12 05 2000
    # NOTA: El guion va al final de la clase [] para evitar error de rango
    date_pattern = r'\b\d{2}[/.\s-]\d{2}[/.\s-]\d{4}\b'
    found_dates = []

    custom_config = r'--oem 3 --psm 6'

    for nombre_proc, img_proc in procesamientos:
        ocr_data = pytesseract.image_to_data(img_proc, output_type=Output.DICT, config=custom_config)
        
        for i in range(len(ocr_data['text'])):
            t = ocr_data['text'][i].strip()
            # DEBUG: Imprimir candidatos potenciales
            # if re.search(r'\d', t) and len(t) > 5:
            if re.search(date_pattern, t):
                try:
                    match = re.search(date_pattern, t)
                    if match:
                        t = match.group(0)
                        # Normalizar separadores a /
                        t_clean = re.sub(r'[-.\s]', '/', t)
                        dt = datetime.strptime(t_clean, "%d/%m/%Y")
                        # Filtrar fechas imposibles (ej. año 0001 o futuro lejano)
                        if 1900 < dt.year < datetime.now().year:
                             found_dates.append((dt, t_clean))
                except:
                    continue
        
        if found_dates:
             break # Si ya encontramos en algun metodo, paramos para velocidad

    if found_dates:
        # La fecha de nacimiento suele ser la menor encontrada (menor que vencimiento)
        birth_date = min(found_dates, key=lambda x: x[0])
        return birth_date[1]
    
    # --- ESTRATEGIA 2: BUSQUEDA LOCALIZADA (Si tenemos referencia) ---
    if ref_box:
         # print("DEBUG: Intentando búsqueda localizada por referencia...")
         nx1, ny1, nx2, ny2 = ref_box
         h, w, _ = img.shape
         
         # Zona estimada: Debajo del nombre, ancho completo o parcial
         # La fecha suele estar más abajo del nombre
         
         # Empezamos un poco más abajo del nombre
         roi_y1 = min(h, ny2 + 5) # Menos margen inicial
         roi_y2 = min(h, ny2 + 200) # Más altura por si acaso
         
         # Tomamos todo el ancho
         roi_x1 = 0
         roi_x2 = w
         
         roi_local = img[roi_y1:roi_y2, roi_x1:roi_x2]
         
         if roi_local.size > 0:
             # cv2.imshow("Debug ROI Fecha", roi_local)
             gray_local = cv2.cvtColor(roi_local, cv2.COLOR_BGR2GRAY)
             
             # Aplicar un poco de zoom puede ayudar
             scaled = cv2.resize(gray_local, None, fx=2, fy=2)
             
             # Probar otsu simple
             _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
             

             
             # OCR localized - Probar varios modos PSM
             # 6: Block of text, 11: Sparse text, 3: Auto
             configs = [r'--oem 3 --psm 6', r'--oem 3 --psm 11', r'--oem 3 --psm 4']
             
             for config in configs:
                 d = pytesseract.image_to_data(thresh, output_type=Output.DICT, config=config)
                 
                 for i in range(len(d['text'])):
                    t = d['text'][i].strip()
                    if re.search(date_pattern, t):
                        try:
                            # Extract the date part specifically
                            match = re.search(date_pattern, t)
                            if match:
                                t_clean = match.group(0)
                                t_clean = re.sub(r'[-.\s]', '/', t_clean)
                                dt = datetime.strptime(t_clean, "%d/%m/%Y")
                                if 1900 < dt.year < datetime.now().year:
                                     return t_clean
                        except:
                            pass
                    try:
                        t_clean = re.sub(r'[-.\s]', '/', t)
                        dt = datetime.strptime(t_clean, "%d/%m/%Y")
                        if 1900 < dt.year < datetime.now().year:
                             return t_clean
                    except:
                        pass
    
    return None

# Listas de referencia para separar palabras pegadas
NOMBRES_COMUNES = [
    "ALEJANDRO", "JOSE", "MARIA", "JUAN", "CARLOS", "LUIS", "ANA", "MIGUEL",
    "ANGEL", "FRANCISCO", "JESUS", "PEDRO", "RAFAEL", "GABRIEL", "ANDRES",
    "DANIEL", "DAVID", "MANUEL", "ANTONIO", "JAVIER", "ROSA", "CARMEN",
    "ELIZABETH", "JENNIFER"
]

APELLIDOS_COMUNES = [
    "VILLA", "MARTINEZ", "PEREZ", "GARCIA", "RODRIGUEZ", "GONZALEZ",
    "HERNANDEZ", "LOPEZ", "SANCHEZ", "RAMIREZ", "FLORES", "TORRES",
    "DIAZ", "VASQUEZ", "CASTRO", "ROMERO", "SUAREZ", "ALVAREZ"
]

def separar_palabras_pegadas(texto, diccionario):
    texto_upper = texto.upper()
    for palabra in diccionario:
        if palabra in texto_upper:
             # Reemplazar la palabra por " PALABRA " pero cuidando no duplicar si ya está separado
             # El replace simple funciona si luego limpiamos dobles espacios
             texto_upper = texto_upper.replace(palabra, f" {palabra} ")
    
    # Limpiar espacios dobles generados
    return re.sub(r'\s+', ' ', texto_upper).strip()

def limpiar_apellido(texto):
    # Corrección específica: W -> V al inicio (común en errores de OCR VILLA -> WILLA)
    if texto and texto.startswith("W"):
        texto = "V" + texto[1:]
    
    # Separar apellidos pegados
    texto = separar_palabras_pegadas(texto, APELLIDOS_COMUNES)
    return texto

def limpiar_texto_basico(text):
    # Eliminar etiquetas que el OCR haya leído por error
    text = re.sub(r'\b(NOMBRES|APELLIDOS|NOMBRE|APELLIDO)\b', '', text)
    
    text = re.sub(' +', ' ', text).strip()
    return text

# Cargar Modelo
model = YOLO(model_path)
img = cv2.imread(image_path)
resultados = model(img)[0]

print("-" * 30)

detecciones_clases = {}

for result in resultados.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    class_name = resultados.names[int(class_id)]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Guardamos detección
    if class_name == "Nombre":
        detecciones_clases["Nombre"] = (x1, y1, x2, y2)
    elif class_name == "Apellido":
        detecciones_clases["Apellido"] = (x1, y1, x2, y2)

    if class_name != "Foto":
        h, w, _ = img.shape
        # Padding lateral generoso para que no corte la primera/ultima letra
        pad_x = 15 
        pad_y = 5  
        
        roi = img[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
        
        # --- ESTRATEGIA Y CONFIGURACIÓN ---
        if class_name == "Numero":
            roi_ocr = procesar_numero(roi)
            # Agregamos guión por si acaso
            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=VvEe0123456789.-'
        else:
            # Nombres, Apellidos, Cedula
            roi_ocr = procesar_texto(roi)
            
            # --- CAMBIO CLAVE AQUÍ ---
            # 1. preserve_interword_spaces=1: Forza a mantener espacios
            # 2. whitelist: Incluye explícitamente el espacio en blanco al final de la cadena
            config = r'--oem 3 --psm 7 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÑ\ '

        # --- LECTURA ---
        try:
            texto = pytesseract.image_to_string(roi_ocr, config=config).strip()
            
            # --- INTENTO DE RECUPERACIÓN PARA NUMERO ---
            # Si fallo el primer intento y es Numero, probamos el metodo suave
            if class_name == "Numero" and not texto:
                # print("Re-intentando OCR de Numero con metodo suave...")
                roi_ocr = procesar_numero_suave(roi)
                texto = pytesseract.image_to_string(roi_ocr, config=config).strip()

            
            # Limpieza extra por si quedaron doble espacios
            texto = re.sub(r'\s+', ' ', texto) 
            texto = limpiar_texto_basico(texto)

            if class_name == "Apellido":
                texto = limpiar_apellido(texto)
            elif class_name == "Nombre":
                texto = separar_palabras_pegadas(texto, NOMBRES_COMUNES)
            
            print(f"CAMPO: {class_name}")
            print(f"DETECTADO: {texto}")
            
            # Mira esta ventana: Debe haber "caminito" negro claro entre ALEJANDRO y JOSE
            cv2.imshow(f"Debug {class_name}", roi_ocr)
            
        except Exception as e:

            print(f"Error en {class_name}")

        print("-" * 20)

# --- DETECCIÓN DE FECHA DE NACIMIENTO ---
ref_nombre = detecciones_clases.get("Nombre") # Puede ser None
fecha = detectar_fecha_nacimiento(img, ref_box=ref_nombre)
if fecha:
    print(f"CAMPO: FECHA DE NACIMIENTO")
    print(f"DETECTADO: {fecha}")
    print("-" * 20)

cv2.waitKey(0)
cv2.destroyAllWindows()