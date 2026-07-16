import cv2

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(1)

# Optional: Higher resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # -----------------------------
    # Convert to grayscale
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Slight blur
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # -----------------------------
    # Edge Detection
    # -----------------------------
    edges = cv2.Canny(blur, 10, 40)

    # 'edges' already has:
    # Black background (0)
    # White edges (255)
    sketch = edges

    # -----------------------------
    # Display
    # -----------------------------
    cv2.imshow("Original", frame)
    cv2.imshow("White Sketch", sketch)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
