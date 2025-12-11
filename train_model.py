# import cv2
# import os
# import numpy as np
#
# data_path = 'known_faces'
# faces = []
# labels = []
# label_map = {}
#
# current_label = 0
#
# for person_name in os.listdir(data_path):
#     person_path = os.path.join(data_path, person_name)
#     if not os.path.isdir(person_path):
#         continue
#
#     label_map[current_label] = person_name
#     for filename in os.listdir(person_path):
#         img_path = os.path.join(person_path, filename)
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         faces.append(img)
#         labels.append(current_label)
#
#     current_label += 1
#
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.train(faces, np.array(labels))
# recognizer.save("trained_model.yml")
#
# # Save label map
# import pickle
# with open("labels.pickle", "wb") as f:
#     pickle.dump(label_map, f)
#
# print("Training complete.")
import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces, ids = [], []

image_dir = "dataset"
for image_name in os.listdir(image_dir):
    path = os.path.join(image_dir, image_name)
    gray_img = Image.open(path).convert('L')
    img_np = np.array(gray_img, 'uint8')
    id_ = int(os.path.split(image_name)[-1].split('.')[1])
    faces.append(img_np)
    ids.append(id_)

recognizer.train(faces, np.array(ids))
recognizer.write('trainer.yml')
print("✅ Model trained and trainer.yml saved.")
