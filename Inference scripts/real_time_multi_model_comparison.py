import cv2
import torch
import torch.nn as nn
import numpy as np
import json
import time
import mediapipe as mp
import joblib
from collections import Counter, defaultdict
from torchvision import transforms, models
from cvzone.HandTrackingModule import HandDetector
from ultralytics import YOLO

# ================= PATHS =================

RESNET_MODEL_PATH = r"D:/Projects/Capstone Project/Vision based hand sign detection & classification usig DL models/runs/resnet50/best_resnet50.pth"
EFF_MODEL_PATH = r"D:/Projects/Capstone Project/Vision based hand sign detection & classification usig DL models/runs/efficientnet_b0/best_efficientnet_b0.pth"
YOLO_MODEL_PATH = r"D:/Projects/Capstone Project/Vision based hand sign detection & classification usig DL models/runs/yolov8n/asl_yolov8/weights/best.pt"

MLP_MODEL_PATH = r"D:/Projects/Capstone Project/Vision based hand sign detection & classification usig DL models/runs/hand_landmark_MLP_pipeline_Normalized-0/asl_landmark_mlp_best.pth"
ENCODER_PATH = r"D:/Projects/Capstone Project/Vision based hand sign detection & classification usig DL models/runs/hand_landmark_MLP_pipeline_Normalized-0/label_encoder.pkl"

CLASS_PATH = r"D:/Projects/Capstone Project/Vision based hand sign detection & classification usig DL models/runs/resnet50/class_names.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD LABELS =================

with open(CLASS_PATH) as f:
    class_names = json.load(f)

# ================= LOAD RESNET =================

resnet = models.resnet50(weights=None)

resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features,512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512,len(class_names))
)

resnet.load_state_dict(torch.load(RESNET_MODEL_PATH,map_location=device,weights_only=True))
resnet = resnet.to(device)
resnet.eval()

print("ResNet Loaded")

# ================= LOAD EFFICIENTNET =================

effnet = models.efficientnet_b0(weights=None)

effnet.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(effnet.classifier[1].in_features,512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512,len(class_names))
)

effnet.load_state_dict(torch.load(EFF_MODEL_PATH,map_location=device,weights_only=True))
effnet = effnet.to(device)
effnet.eval()

print("EfficientNet Loaded")

# ================= LOAD YOLO =================

yolo_model = YOLO(YOLO_MODEL_PATH)
print("YOLO Loaded")

# ================= LOAD MLP =================

le = joblib.load(ENCODER_PATH)

class ASL_MLP(nn.Module):

    def __init__(self,num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(63,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,num_classes)
        )

    def forward(self,x):
        return self.net(x)

mlp = ASL_MLP(len(le.classes_)).to(device)
mlp.load_state_dict(torch.load(MLP_MODEL_PATH,map_location=device))
mlp.eval()

print("MLP Loaded")

# ================= MEDIAPIPE =================

mp_hands = mp.solutions.hands

hands_mp = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

# ================= TRANSFORM =================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ================= HAND DETECTOR =================

detector = HandDetector(maxHands=2,detectionCon=0.8)

cap = cv2.VideoCapture(0)

# ================= ACCURACY TRACKER (Separate for Left and Right) =================

stats = {
    "YOLO": {"left": [0,0], "right": [0,0]},
    "ResNet": {"left": [0,0], "right": [0,0]},
    "EffNet": {"left": [0,0], "right": [0,0]},
    "MLP": {"left": [0,0], "right": [0,0]}
}

ground_truth_left = None
ground_truth_right = None

# Track which hand is currently being processed
current_hand_type = None

# Store predictions for display
display_predictions = {}  # Will store predictions for each hand

# ================= FPS =================

prev_time = 0

# ================= HELPER FUNCTIONS =================

def predict_cnn(model,img):
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs,dim=1)
        conf,pred = torch.max(probs,1)

    return class_names[pred.item()],conf.item()

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    landmarks = landmarks - landmarks[0]
    max_val = np.max(np.abs(landmarks))
    if max_val!=0:
        landmarks = landmarks/max_val
    return landmarks.flatten()

def calc_acc(model, hand_type):
    correct,total = stats[model][hand_type]
    if total==0:
        return 0
    return (correct/total)*100

def determine_hand_type(hand_landmarks, frame_shape):
    """Determine if hand is left or right based on wrist position"""
    h, w = frame_shape[:2]
    wrist_x = hand_landmarks.landmark[0].x * w
    
    # If wrist is in left half of frame, it's likely left hand
    if wrist_x < w/2:
        return "left"
    else:
        return "right"

# ================= MAIN LOOP =================

while True:
    success,frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    # Get MediaPipe results for hand type detection
    mp_results = hands_mp.process(rgb_frame)
    
    # Store hand types mapping
    hand_types = {}
    if mp_results.multi_handedness:
        for idx, hand_handedness in enumerate(mp_results.multi_handedness):
            hand_type = hand_handedness.classification[0].label.lower()  # 'left' or 'right'
            hand_types[idx] = hand_type

    hands = detector.findHands(frame,draw=False)

    # Clear display predictions for this frame
    display_predictions = {}

    if hands is not None and len(hands)>0:
        for hand_idx, hand in enumerate(hands):
            x,y,w,h = hand["bbox"]
            
            # Determine hand type
            if hand_idx in hand_types:
                hand_type = hand_types[hand_idx]
            else:
                # Fallback to position-based detection
                hand_type = "left" if x < frame.shape[1]/2 else "right"
            
            current_hand_type = hand_type

            x1=max(0,x-30)
            y1=max(0,y-30)
            x2=min(frame.shape[1], x+w+30)
            y2=min(frame.shape[0], y+h+30)

            crop = frame[y1:y2,x1:x2]

            if crop.size==0:
                continue

            crop = cv2.resize(crop,(224,224))

            # ===== CNN MODELS =====
            res_class, res_conf = predict_cnn(resnet, crop)
            eff_class, eff_conf = predict_cnn(effnet, crop)

            # ===== YOLO =====
            yolo_results = yolo_model(crop)
            yolo_class = "None"
            yolo_conf = 0
            
            if len(yolo_results[0].boxes)>0:
                box = yolo_results[0].boxes[0]
                yolo_class = yolo_model.names[int(box.cls[0])]
                yolo_conf = float(box.conf[0])

            # ===== MLP =====
            mlp_class = "None"
            mlp_conf = 0
            
            if mp_results.multi_hand_landmarks and hand_idx < len(mp_results.multi_hand_landmarks):
                lm=[]
                for p in mp_results.multi_hand_landmarks[hand_idx].landmark:
                    lm.append([p.x,p.y,p.z])

                lm = normalize_landmarks(lm)
                tensor = torch.tensor(lm,dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = mlp(tensor)
                    probs = torch.softmax(outputs,dim=1)
                    mlp_conf, pred = torch.max(probs,1)
                    mlp_class = le.inverse_transform([pred.item()])[0]
                    mlp_conf = mlp_conf.item()

            # ===== ENSEMBLE =====
            predictions = [yolo_class, res_class, eff_class, mlp_class]
            
            # Filter out "None" predictions for ensemble
            valid_predictions = [p for p in predictions if p != "None"]
            if valid_predictions:
                final = Counter(valid_predictions).most_common(1)[0][0]
            else:
                final = "None"

            # Calculate ensemble confidence
            if final != "None":
                confidences = []
                if yolo_conf > 0 and yolo_class == final:
                    confidences.append(yolo_conf)
                if res_conf > 0 and res_class == final:
                    confidences.append(res_conf)
                if eff_conf > 0 and eff_class == final:
                    confidences.append(eff_conf)
                if mlp_conf > 0 and mlp_class == final:
                    confidences.append(mlp_conf)
                
                ensemble_conf = np.mean(confidences) if confidences else 0
            else:
                ensemble_conf = 0

            # Store predictions for display
            display_predictions[hand_type] = {
                'bbox': (x1, y1, x2, y2),
                'yolo': (yolo_class, yolo_conf),
                'resnet': (res_class, res_conf),
                'effnet': (eff_class, eff_conf),
                'mlp': (mlp_class, mlp_conf),
                'final': (final, ensemble_conf)
            }

            # ===== ACCURACY UPDATE (Separate for Left/Right) =====
            ground_truth = ground_truth_left if hand_type == "left" else ground_truth_right
            
            if ground_truth is not None:
                # Update stats for this hand type
                stats["YOLO"][hand_type][1] += 1
                stats["ResNet"][hand_type][1] += 1
                stats["EffNet"][hand_type][1] += 1
                stats["MLP"][hand_type][1] += 1

                if yolo_class == ground_truth:
                    stats["YOLO"][hand_type][0] += 1
                if res_class == ground_truth:
                    stats["ResNet"][hand_type][0] += 1
                if eff_class == ground_truth:
                    stats["EffNet"][hand_type][0] += 1
                if mlp_class == ground_truth:
                    stats["MLP"][hand_type][0] += 1

    # Draw all predictions after processing all hands
    for hand_type, preds in display_predictions.items():
        x1, y1, x2, y2 = preds['bbox']
        yolo_class, yolo_conf = preds['yolo']
        res_class, res_conf = preds['resnet']
        eff_class, eff_conf = preds['effnet']
        mlp_class, mlp_conf = preds['mlp']
        final, ensemble_conf = preds['final']
        
        # ===== DRAW BOUNDING BOX WITH HAND TYPE =====
        box_color = (255,0,0) if hand_type == "left" else (0,0,255)  # Blue for left, Red for right
        cv2.rectangle(frame,(x1,y1),(x2,y2),box_color,2)
        
 
        cv2.putText(frame,f"{hand_type.upper()} HAND",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,box_color,2)

        # ===== DRAW PREDICTIONS AT BOTTOM OF BOUNDING BOX =====
        base_y = y2 + 20 
        

        yolo_text = f"YOLO: {yolo_class}" + (f" ({yolo_conf:.2f})" if yolo_conf > 0 else "")
        cv2.putText(frame, yolo_text, (x1, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        
  
        res_text = f"ResNet: {res_class} ({res_conf:.2f})"
        cv2.putText(frame, res_text, (x1, base_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        
      
        eff_text = f"EffNet: {eff_class} ({eff_conf:.2f})"
        cv2.putText(frame, eff_text, (x1, base_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
      
        mlp_text = f"MLP: {mlp_class}" + (f" ({mlp_conf:.2f})" if mlp_conf > 0 else "")
        cv2.putText(frame, mlp_text, (x1, base_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        
        final_text = f"FINAL: {final} ({ensemble_conf:.2f})"
        cv2.putText(frame, final_text, (x1, base_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 3)

    # ===== FPS =====
    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ===== ACCURACY DASHBOARD (Separate for Left and Right) =====
    y0 = 60
    x_offset = 10

    # ===== GROUND TRUTH DISPLAY =====
    if ground_truth_left is not None or ground_truth_right is not None:
        gt_text = "GT: "
        if ground_truth_left is not None:
            gt_text += f"L:{ground_truth_left} "
        if ground_truth_right is not None:
            gt_text += f"R:{ground_truth_right}"
        cv2.putText(frame, gt_text, (frame.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    # Add instructionq text
    cv2.putText(frame, "Press A-Z to set Ground Truth for current hand | R: Reset | Q: Quit", 
                (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("SignWise+ Ensemble AI System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key != 255:
        char = chr(key).upper()
        
        # Set ground truth for left or right hand
        if char in class_names:
            # Check if we need to set for left or right based on current hand
            if current_hand_type == "left":
                ground_truth_left = char
                print(f"Ground Truth for LEFT hand set to: {char}")
            elif current_hand_type == "right":
                ground_truth_right = char
                print(f"Ground Truth for RIGHT hand set to: {char}")
            else:
                # If no hand detected, set both? Or prompt user?
                ground_truth_left = char
                ground_truth_right = char
                print(f"Ground Truth set to: {char} for both hands")
        
        # Reset ground truths
        if char == "R":  # Press 'R' to reset both ground truths
            ground_truth_left = None
            ground_truth_right = None
            print("Ground truths reset")
        
        if char == "Q":
            break

cap.release()
cv2.destroyAllWindows()

# ===== FINAL STATISTICS =====
print("\n" + "="*50)
print("FINAL ACCURACY STATISTICS")
print("="*50)

for model in stats:
    print(f"\n{model}:")
    for hand_type in ["left", "right"]:
        correct, total = stats[model][hand_type]
        acc = (correct/total)*100 if total > 0 else 0
        print(f"  {hand_type.capitalize()} Hand: {correct}/{total} = {acc:.2f}%")