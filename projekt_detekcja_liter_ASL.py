import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import sys
import time
from collections import Counter
import mediapipe as mp
from collections import deque

# Wczytanie i kompilacja modelu
try:
    model = tf.keras.models.load_model('C:/Users/szyns/Desktop/magister semestr I/sztuczna-inteligencja-lab/model_large_asl_mobilenet-v1.h5', compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model wczytany i skompilowany pomyślnie.")
except Exception as e:
    print(f"Błąd podczas wczytywania modelu: {e}")
    # Fallback: Zdefiniowanie architektury modelu i zaladowanie wag
    try:
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(29, activation='softmax')(x)  # 29 klas (A-Z, del,space,nothing)
        model = Model(inputs=base_model.input, outputs=x)
        model.load_weights('C:/Users/szyns/Desktop/magister semestr I/sztuczna-inteligencja-lab/model_large_asl_mobilenet-v1.h5')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model zrekonstruowany i wagi wczytane pomyślnie.")
    except Exception as e2:
        print(f"Błąd podczas wczytywania wag modelu: {e2}")
        sys.exit(1)

# Mapowanie klas (etykiet)
labels = {i: chr(65 + i) for i in range(26)}  # A-Z
labels[26] = 'del'
labels[27] = 'nothing'
labels[28] = 'space'
print(labels)
# Parametry
IMG_SIZE = (224, 224)
MAX_WORD_LENGTH = 10  # Maksymalna długość słowa
ROI_SIZE = (224, 224)  # Rozmiar ROI dla ręki
PREDICTION_INTERVAL = 0.1  # Sekundy między predykcjami
DISPLAY_INTERVAL = 0.1  # Sekundy między aktualizacjami wyświetlacza
PREDICTION_WINDOW = 100  # Maksymalna liczba predykcji do analizy
MIN_PREDICTIONS = 60  # Minimalna liczba predykcji do analizy
CONSISTENCY_THRESHOLD = 0.8  # 80% zgodności
BASE_CONFIDENCE_THRESHOLD = 0.75  # Bazowy próg pewności
HIGH_CONFIDENCE_THRESHOLD = 0.95  # Próg dla natychmiastowego dodania
STABILITY_THRESHOLD = 10.0  # Próg wariancji punktów charakterystycznych (piksele)
VELOCITY_THRESHOLD = 20.0  # Próg prędkości ruchu ręki (piksele/sekundę)
FLIP_CAMERA = True  # Ustaw na True, aby odwrócić lustrzane odbicie kamery


# Inicjalizacja MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Inicjalizacja tła (fallback)
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Funkcja do wyszukiwania dostępnej kamery
def find_camera(max_index=4):
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Użyj DirectShow dla Windows
        if cap.isOpened():
            print(f"Kamera znaleziona na indeksie {index}")
            return cap, index
        cap.release()
    return None, None

# Funkcja do obliczania wariancji punktów charakterystycznych
def calculate_landmark_variance(landmarks, width, height):
    if not landmarks:
        return float('inf')
    coords = [[lm.x * width, lm.y * height] for lm in landmarks]
    coords = np.array(coords)
    variance = np.var(coords, axis=0).sum()
    return variance

# Funkcja do obliczania prędkości punktów charakterystycznych
def calculate_landmark_velocity(prev_landmarks, current_landmarks, width, height, time_diff):
    if not prev_landmarks or not current_landmarks or time_diff <= 0:
        return float('inf')
    prev_coords = np.array([[lm.x * width, lm.y * height] for lm in prev_landmarks])
    current_coords = np.array([[lm.x * width, lm.y * height] for lm in current_landmarks])
    displacement = np.linalg.norm(current_coords - prev_coords, axis=1).mean()
    velocity = displacement / time_diff
    return velocity

# Inicjalizacja kamery
cap, camera_index = find_camera()
if cap is None:
    print("Błąd: Nie znaleziono żadnej kamery. Upewnij się, że kamera jest podłączona i działa.")
    print("Sprawdź uprawnienia lub podłącz inną kamerę.")
    sys.exit(1)

# Zmienne do przechowywania wyrazu i stanu
current_word = ""
last_appended_letter = None
space_detected = False
last_prediction_time = 0
last_display_time = 0
prediction_history = deque(maxlen=PREDICTION_WINDOW)
current_predicted_label = 'nothing'
current_confidence = 0.0
prev_landmarks = None
last_landmark_time = 0
gesture_status = "No Hand Detected"
word_color = (0, 255, 0)  # Zielony domyślnie

try:
    while True:
        # Odczyt klatki z kamery
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie można odczytać klatki z kamery.")
            break

        # Odwrócenie lustrzanego odbicia kamery, jeśli FLIP_CAMERA=True
        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        # Definiowanie statycznego ROI (fallback)
        height, width = frame.shape[:2]
        roi_x = (width - ROI_SIZE[0]) // 2
        roi_y = (height - ROI_SIZE[1]) // 2
        roi = frame[roi_y:roi_y + ROI_SIZE[1], roi_x:roi_x + ROI_SIZE[0]]

        # Sprawdzenie, czy ROI ma poprawny rozmiar
        if roi.shape[0] != ROI_SIZE[0] or roi.shape[1] != ROI_SIZE[1]:
            print("Błąd: ROI ma nieprawidłowy rozmiar. Dostosuj rozdzielczość kamery.")
            continue

        # MediaPipe Hand Detection
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = hands.process(roi_rgb)
        hand_detected = False
        masked_roi = roi.copy()
        current_landmarks = None

        if results.multi_hand_landmarks:
            hand_detected = True
            hand = results.multi_hand_landmarks[0]
            current_landmarks = hand.landmark
            # Obliczanie dynamicznego ROI na podstawie punktów charakterystycznych
            h, w, _ = roi.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            # Rozszerzenie bounding boxa dla lepszego pokrycia
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            # Wycięcie dynamicznego ROI
            hand_roi = roi[y_min:y_max, x_min:x_max]
            if hand_roi.size > 0:
                masked_roi = np.zeros_like(roi)
                masked_roi[y_min:y_max, x_min:x_max] = hand_roi
            # Rysowanie punktów charakterystycznych (dla debugowania)
            mp_drawing.draw_landmarks(roi, hand, mp_hands.HAND_CONNECTIONS)
        else:
            # Fallback: Background subtraction (nie używane do predykcji)
            fg_mask = back_sub.apply(roi)
            masked_roi = cv2.bitwise_and(roi, roi, mask=fg_mask)
            gesture_status = "No Hand Detected"

        # Obliczanie stabilności gestu i prędkości (tylko jeśli ręka wykryta)
        current_time = time.time()
        if hand_detected:
            variance = calculate_landmark_variance(current_landmarks, ROI_SIZE[0], ROI_SIZE[1])
            time_diff = current_time - last_landmark_time if last_landmark_time > 0 else 1.0
            velocity = calculate_landmark_velocity(prev_landmarks, current_landmarks, ROI_SIZE[0], ROI_SIZE[1], time_diff)
            prev_landmarks = current_landmarks
            last_landmark_time = current_time

            # Adaptacyjny próg pewności
            confidence_threshold = BASE_CONFIDENCE_THRESHOLD
            if variance > STABILITY_THRESHOLD or velocity > VELOCITY_THRESHOLD:
                confidence_threshold = 0.9  # Wyższy próg dla niestabilnych gestów
                gesture_status = "Gesture Unclear"
            else:
                gesture_status = "Gesture Clear"
        else:
            confidence_threshold = BASE_CONFIDENCE_THRESHOLD
            prev_landmarks = None

        # Rysowanie prostokąta ROI (zielony jeśli ręka wykryta, czerwony jeśli nie)
        roi_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + ROI_SIZE[0], roi_y + ROI_SIZE[1]), roi_color, 2)

        # Aktualizacja wyświetlacza co 0.1 sekundy (tylko jeśli ręka wykryta)
        if current_time - last_display_time >= DISPLAY_INTERVAL and hand_detected:
            # Przygotowanie obrazu do predykcji
            img = cv2.resize(masked_roi, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # Predykcja
            pred = model.predict(img, verbose=0)
            pred_class = np.argmax(pred, axis=1)[0]
            current_confidence = np.max(pred)
            current_predicted_label = labels[pred_class]
            last_display_time = current_time

        # Zapis predykcji i analiza co dany czas (tylko jeśli ręka wykryta)
        if current_time - last_prediction_time >= PREDICTION_INTERVAL and hand_detected:
            # Dodaj predykcję do historii, jeśli pewność jest wystarczająca
            if current_confidence >= confidence_threshold:
                prediction_history.append(current_predicted_label)
                # Ogranicz historię do maksymalnie ... predykcji
                if len(prediction_history) > PREDICTION_WINDOW:
                    prediction_history.popleft()

            last_prediction_time = current_time

            # Natychmiastowe dodanie litery, jeśli pewność > ... i ... ostatnie predykcje są zgodne
            if current_confidence > HIGH_CONFIDENCE_THRESHOLD and len(prediction_history) >= 3 and \
                    prediction_history[-1] == prediction_history[-2] and  prediction_history[-2] == \
                    prediction_history[-3] and  prediction_history[-3] :
                dominant_label = current_predicted_label
                # Logika składania wyrazu
                if dominant_label == 'del':
                    current_word = current_word[:-1]
                    last_appended_letter = None
                    space_detected = False
                    prediction_history.clear()
                    word_color = (0, 255, 255)  # Żółty dla potwierdzenia
                elif dominant_label == 'space' and len(current_word) < MAX_WORD_LENGTH:
                    current_word += ' '
                    last_appended_letter = None
                    space_detected = True
                    prediction_history.clear()
                    word_color = (0, 255, 255)
                elif dominant_label != 'nothing' and len(current_word) < MAX_WORD_LENGTH:
                    # Dodaj literę tylko jeśli różni się od ostatniej dodanej lub była spacja
                    if dominant_label != last_appended_letter or space_detected:
                        current_word += dominant_label
                        last_appended_letter = dominant_label
                        space_detected = False
                        prediction_history.clear()
                        word_color = (0, 209, 255)
                elif len(current_word) >= MAX_WORD_LENGTH:
                    current_word = ''
                    prediction_history.clear()
                    word_color = (0, 255, 255)
                # Reset koloru po 0.5 sekundy
                if word_color != (0, 255, 0):
                    cv2.setMouseCallback('ASL Recognition', lambda *args: None)  # Dummy callback
                    cv2.waitKey(500)
                    word_color = (0, 255, 0)

            # Analiza historii predykcji, jeśli jest wystarczająco dużo predykcji
            elif len(prediction_history) >= MIN_PREDICTIONS:
                # Liczenie wystąpień każdej litery
                counter = Counter(prediction_history)
                most_common = counter.most_common(1)
                if most_common:
                    dominant_label, count = most_common[0]
                    # Sprawdzenie, czy dominująca litera spełnia próg
                    if count / len(prediction_history) >= CONSISTENCY_THRESHOLD and (current_confidence >= confidence_threshold or len(prediction_history) >= 3):
                        # Logika składania wyrazu
                        if dominant_label == 'del':
                            current_word = current_word[:-1]
                            last_appended_letter = None
                            space_detected = False
                            prediction_history.clear()
                            word_color = (0, 255, 255)
                        elif dominant_label == 'space' and len(current_word) < MAX_WORD_LENGTH:
                            current_word += ' '
                            last_appended_letter = None
                            space_detected = True
                            prediction_history.clear()
                            word_color = (0, 255, 255)
                        elif dominant_label != 'nothing' and len(current_word) < MAX_WORD_LENGTH:
                            # Dodaj literę tylko jeśli różni się od ostatniej dodanej lub była spacja
                            if dominant_label != last_appended_letter or space_detected:
                                current_word += dominant_label
                                last_appended_letter = dominant_label
                                space_detected = False
                                prediction_history.clear()
                                word_color = (0, 255, 255)
                        elif len(current_word) >= MAX_WORD_LENGTH:
                            current_word = ''
                            prediction_history.clear()
                            word_color = (0, 255, 255)
                        # Reset koloru po 0.5 sekundy
                        if word_color != (0, 255, 0):
                            cv2.setMouseCallback('ASL Recognition', lambda *args: None)
                            cv2.waitKey(500)
                            word_color = (0, 255, 0)

        # Wyświetlenie rozpoznanej litery, wyrazu, pewności i statusu gestu
        cv2.putText(frame, f"Letter: {current_predicted_label} ({current_confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Word: {current_word}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, word_color, 2)
        cv2.putText(frame, f"Status: {gesture_status}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Wyświetlenie obrazu z kamery
        cv2.imshow('ASL Recognition', frame)

        # Opcjonalne: Wyświetlenie maskowanego ROI dla debugowania
        #cv2.imshow('Masked ROI', masked_roi)

        # Zakończenie programu po naciśnięciu klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Zamykanie aplikacji po naciśnięciu 'q'.")
            break

except KeyboardInterrupt:
    print("Program zakończony przez użytkownika.")
except Exception as e:
    print(f"Wystąpił błąd: {e}")
finally:
    # Zwolnienie zasobów
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Zasoby kamery zwolnione.")