import os
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import cv2
import numpy as np


def prepare_training_data(training_dir):
    labels = []
    embeddings = []
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            faces = app.get(img)
            for face in faces:
                aligned_face = face_align.norm_crop(img, landmark=face.kps)
                embedding = face.embedding.flatten()
                labels.append(person_name)
                embeddings.append(embedding)
    
    return np.array(embeddings), np.array(labels)

def train_model(embeddings, labels):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels_encoded)
    
    return model, label_encoder

def evaluate_model(model, label_encoder, evaluation_dir, confidence_threshold=0.5):
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    for img_name in os.listdir(evaluation_dir):
        img_path = os.path.join(evaluation_dir, img_name)
        img = cv2.imread(img_path)
        faces = app.get(img)
        for face in faces:
            aligned_face = face_align.norm_crop(img, landmark=face.kps)
            embedding = face.embedding.flatten().reshape(1, -1)
            probabilities = model.predict_proba(embedding)
            label_index = np.argmax(probabilities)
            confidence = probabilities[0][label_index]
            
            if confidence >= confidence_threshold:
                predicted_label = label_encoder.inverse_transform([label_index])[0]
            else:
                predicted_label = "Unknown"
            
            # Draw the label and confidence on the image
            text = f"{predicted_label} ({confidence:.2f})"
            if predicted_label == "Unknown":
                text = f"{predicted_label}"
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Background rectangle for text
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            y1_text = y1 - 10 if y1 - h - 10 > 0 else y1 + h + 10 
            img = cv2.rectangle(img, (x1, y1_text - h), (x1 + w, y1_text), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, text, (x1, y1_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            print(f"Predicted: {predicted_label}, Confidence: {confidence:.2f}")
        
        # Display the image
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()