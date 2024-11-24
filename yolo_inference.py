from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x")
# model = YOLO("new/models/yolo8_last.pt")


# result = model.predict('new/input_videos/input_video.mp4',conf=0.2,save=True)
result = model.track('new/input_videos/input_video.mp4',conf=0.2,save=True)

print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)

