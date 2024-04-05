from ultralytics import YOLO

# set file directories
source_file = "dataset/2.mp4"
yolo_model = "latest.pt"

# load model
model = YOLO(yolo_model)

# run and save video
results = model.predict(source = source_file, conf = .1, show = True, save = True)