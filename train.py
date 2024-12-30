from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    model.train(data='dataset/split/data.yaml', epochs=5)


if __name__ == '__main__':
    main()