# Tutorial de Instalación y Uso de OpenCV y Face_Recognition

## Introducción
OpenCV y `face_recognition` son herramientas poderosas para la visión por computadora, utilizadas para detectar y reconocer rostros en imágenes y videos. Este tutorial explica cómo instalar estas bibliotecas y muestra ejemplos de uso.

---

## 1. Instalación de OpenCV y Face_Recognition

Para comenzar, es necesario instalar las bibliotecas requeridas en Python. Abre una terminal y ejecuta los siguientes comandos:

```bash
pip install opencv-python
pip install face-recognition
pip install numpy
```

Si estás en Windows, es posible que también necesites instalar `dlib` manualmente:

```bash
pip install cmake
pip install dlib
```

### Nota

Si tienes problemas al instalar CMake, puedes optar por instalar Dlib utilizando archivos precompilados. Esto te permitirá instalar Dlib sin necesidad de configurar CMake manualmente. Puedes descargarlos desde este enlace. [Descargar Dlib con archivos precompilados](https://github.com/peter0226/dlib-easy-install)

Para verificar la instalación, ejecuta en Python:

```python
import cv2
import face_recognition
print(cv2.__version__)
print(face_recognition.__version__)
```

---

## 2. Carga y Detección de Rostros en Imágenes

Una de las funciones básicas de `face_recognition` es la detección de rostros en una imagen.

```python
import cv2
import face_recognition

# Cargar imagen
image = face_recognition.load_image_file("persona.jpg")

# Detectar rostros
face_locations = face_recognition.face_locations(image)

print(f"Se detectaron {len(face_locations)} rostros en la imagen.")
```

---

## 3. Detección en Video

Para analizar un video en tiempo real:

```python
import cv2
import face_recognition

# Capturar video desde la cámara
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
```

---

## 4. Reconocimiento Facial Comparando Rostros

Podemos comparar una imagen con una base de datos de rostros conocidos.

```python
import face_recognition
import cv2
import numpy as np

# Cargar la imagen de referencia
known_image = face_recognition.load_image_file("persona_conocida.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Cargar una nueva imagen para comparar
unknown_image = face_recognition.load_image_file("persona_nueva.jpg")
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Comparar rostros
results = face_recognition.compare_faces([known_encoding], unknown_encoding)

if results[0]:
    print("Las personas coinciden.")
else:
    print("Las personas no coinciden.")
```

---

## 5. Aplicaciones Comunes
- Seguridad y control de acceso.
- Autenticación biométrica.
- Análisis de comportamiento en videos.
- Clasificación de rostros en redes sociales.

Este tutorial proporciona una base para explorar la visión por computadora con OpenCV y `face_recognition`. Puedes extender estos ejemplos según tus necesidades. ¡Buena suerte!

