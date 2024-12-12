import cv2
import numpy as np
import time

# Kleuren en hun HSV Hue-bereiken
COLOR_RANGES = {
    "Groen": (35, 85),       # Hue bereik voor groen
    "Oranje": (10, 25),      # Hue bereik voor oranje
    "Donkerblauw": (100, 130), # Hue bereik voor donkerblauw
    "Rood": (0, 10),         # Hue bereik voor rood
    "Wit": (0, 0),           # Speciaal: gebaseerd op hoge waarde en lage saturatie
    "Geel": (25, 35),        # Hue bereik voor geel
    "Paars": (130, 160)      # Hue bereik voor paars
}

# Functie om de vorm te bepalen
def detect_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    vertices = len(approx)

    if vertices == 3:
        return "Driehoek"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "Vierkant"
        else:
            return "Parallellogram"
    else:
        return "Onbekende vorm"

# Functie om een kleurmasker te maken op basis van Hue
def get_color_mask(hsv_image, lower_hue, upper_hue):
    lower = np.array([lower_hue, 50, 50])  # Minimale Saturatie en Value om zwart te filteren
    upper = np.array([upper_hue, 255, 255])
    return cv2.inRange(hsv_image, lower, upper)

# Speciaal masker voor wit
def get_white_mask(hsv_image):
    lower = np.array([0, 0, 200])  # Lage saturatie, hoge value
    upper = np.array([180, 20, 255])
    return cv2.inRange(hsv_image, lower, upper)

# Normaliseer de rotatiehoek naar het bereik van 0 tot 360 graden
def normalize_angle(angle):
    # Zorg ervoor dat de hoek altijd tussen 0 en 360 ligt
    if angle < 0:
        angle += 360
    elif angle >= 360:
        angle -= 360
    return angle

# Verwerk het frame
def process_frame(frame, min_area=1000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output_frame = frame.copy()

    detected_pieces = []

    for color_name, hue_range in COLOR_RANGES.items():
        # Speciaal masker voor wit
        if color_name == "Wit":
            mask = get_white_mask(hsv)
        else:
            lower_hue, upper_hue = hue_range
            mask = get_color_mask(hsv, lower_hue, upper_hue)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Sorteer contouren op grootte (van groot naar klein)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > min_area:
                    shape = detect_shape(contour)
                    x, y, w, h = cv2.boundingRect(contour)

                    # Bepaal de rotatie van de contour (ongeacht de vorm)
                    # Gebruik de grootste zijde om de rotatiehoek te berekenen
                    if shape == "Driehoek" or shape == "Parallellogram":
                        # Bereken de hoeken van de lijnen in de contour
                        min_rect = cv2.minAreaRect(contour)
                        angle = min_rect[2]

                        # Normaliseer de hoek naar het bereik 0-360 graden
                        angle = normalize_angle(angle)
                    else:
                        # Vierkant of rechthoek
                        rect = cv2.minAreaRect(contour)
                        angle = rect[2]
                        # Normaliseer de hoek naar het bereik 0-360 graden
                        angle = normalize_angle(angle)

                    # Teken de contouren en label met positie en rotatie
                    cv2.drawContours(output_frame, [contour], -1, (0, 255, 0), 2)
                    cv2.putText(output_frame, f"{color_name} {shape}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(output_frame, f"Pos: ({x},{y}) Rot: {angle:.2f}", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Sla de gedetecteerde kleur, vorm, positie en rotatie op
                    detected_pieces.append((color_name, shape, (x, y), angle))

    return output_frame, detected_pieces

# Main-functie
def main():
    # Parameters: time delay en minimale grootte
    time_delay = 0.5  # Time delay in seconden
    min_area = 1500   # Minimale grootte van een contour in pixels

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Kan de externe webcam niet openen!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Druk op 'q' om te stoppen.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame niet gelezen!")
            break

        processed_frame, detected_pieces = process_frame(frame, min_area)

        # Toon de uitvoer
        cv2.imshow("Tangram Detectie", processed_frame)

        # Print de gedetecteerde blokken met positie en rotatie
        if detected_pieces:
            for color, shape, position, angle in detected_pieces:
                print(f"Gedetecteerd: Kleur = {color}, Vorm = {shape}, Positie = {position}, Rotatie = {angle:.2f} graden")

        # Wacht op een time delay
        time.sleep(time_delay)

        # Stop als 'q' wordt ingedrukt
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

