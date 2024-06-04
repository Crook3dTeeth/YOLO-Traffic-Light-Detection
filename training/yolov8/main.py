from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('best.pt')

#model.export(format='ncnn')

# Train the model
if __name__ == '__main__':

    # 
    results = model.train(data='data.yaml', epochs=2000, imgsz=640)
    #results = model.train(data='data.yaml', epochs=2000, imgsz=640, patience = 500)
    #results = model.val()