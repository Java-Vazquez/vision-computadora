import cv2

# Función para detectar los ojos en un rostro
def detect_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen en escala de grises
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Para cada rostro detectado, detectar los ojos
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Dibujar rectángulos alrededor de los ojos detectados
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return frame

# Función principal
def main():
    # Cargar los clasificadores pre-entrenados para rostros y ojos
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Crear el objeto de captura de video
    cap = cv2.VideoCapture(0)

    # Comprobar si la webcam se abrió correctamente
    if not cap.isOpened():
        print("No se pudo abrir la webcam")
        return

    while True:
        # Leer el siguiente marco de la webcam
        ret, frame = cap.read()

        if not ret:
            print("No se pudo recibir el marco de la webcam")
            break

        # Detectar los ojos en el marco
        tracked_frame = detect_eyes(frame, face_cascade, eye_cascade)

        # Mostrar el marco resultante
        cv2.imshow("Detección de ojos", tracked_frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función principal
if __name__ == '__main__':
    main()
