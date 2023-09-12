import cv2

# Función para realizar el seguimiento de objetos
def track_objects(frame):
    # Convertir el marco a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral adaptativo
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Realizar el seguimiento de cada contorno
    for contour in contours:
        # Ignorar contornos pequeños
        if cv2.contourArea(contour) > 100:
            # Obtener el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Dibujar el rectángulo delimitador
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

# Función principal
def main():
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
        
        # Aplicar el seguimiento de objetos al marco
        tracked_frame = track_objects(frame)
        
        # Mostrar el marco resultante
        cv2.imshow("Seguimiento de objetos", tracked_frame)
        
        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función principal
if __name__ == '__main__':
    main()
