from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"C:\Users\USER\OneDrive\Desktop\AI-MODEL\runs\detect\sitophilus_oryzae-4\weights\last.pt")

    model.train(
        data=r"C:\Users\USER\OneDrive\Desktop\AI-MODEL\Sitophilus Oryzae YOLO Detection.v2-sitophilus_oryzae_augmented_dataset_v1.0_yolov11n.yolov11\data.yaml",
        epochs=157,
        imgsz=640,
        batch=8,
        device=0,
        patience=0,
        name="sitophilus_oryzae_v2"
    )