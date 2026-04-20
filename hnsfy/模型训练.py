from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo11s-cls.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="F:\hnsfy20_yolo", epochs=50, imgsz=640)

if __name__ == '__main__':
    main()