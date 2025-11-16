from ultralytics import YOLO

model = YOLO('yolov13x.yaml')

# Train the model
results = model.train(
  data='/kaggle/input/US-needle/needle/data.yaml',
  epochs=600, 
  batch=32, 
  imgsz=640,
  scale=0.9,  # S:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.2,  # S:0.05; L:0.15; X:0.2
  copy_paste=0.6,  # S:0.15; L:0.5; X:0.6
  device="0,1",
)

# Evaluate model performance on the validation set
metrics = model.val('/kaggle/input/US-needle/needle/data.yaml')

