from ultralytics import YOLO

model = YOLO('yolo11n.yaml') # you can change the size (n,s,m,l,x) or change to .pt for pretrained model
results = model.train(data='dataset/data.yaml', epochs=500, imgsz=640) # there is also patience involved when overfitting is detected, and also you can increase epochs if it is not enough