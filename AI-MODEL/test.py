from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO(r"C:\Users\USER\OneDrive\Desktop\AI-MODEL\runs\detect\sitophilus_oryzae-4\weights\best.pt")
    
    model.to('cuda')  # Use GPU for inference

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower resolution
    cap.set(cv2.CAP_PROP_FPS, 30)            # Set FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=0.5,
            device=0,        # GPU
            verbose=False,   # No console spam
            imgsz=320        # Smaller = faster
        )

        annotated_frame = results[0].plot()
        cv2.imshow('Rice Weevil Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()