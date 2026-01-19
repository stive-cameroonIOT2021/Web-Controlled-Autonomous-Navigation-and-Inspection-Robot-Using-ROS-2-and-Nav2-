from ultralytics import YOLO

# Load model (handles architecture, custom classes, safe loading, device, etc.)
model = YOLO(r"C:\Users\Administrator\Documents\Python\weights\best.pt")

# Run inference - very convenient
results = model(r"C:\Users\Administrator\Documents\Python\weights\4.jpg")[0]

# Visualize / save
results.show()                    # opens window with boxes
results.save(filename="prediction.jpg")

# Or access data programmatically
print(results.boxes)              # xyxy format, confidences, class ids...
print(results.names)              # dict of class index → name