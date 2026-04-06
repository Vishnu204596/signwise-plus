import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import joblib
from collections import deque, Counter
import time
import pyttsx3
import threading
import queue

# ================= PATHS =================
MODEL_PATH = r"D:\Projects\Capstone Project\Vision based hand sign detection & classification usig DL models\runs\hand_landmark_MLP_original_structure-1\asl_landmark_mlp_best.pth"
ENCODER_PATH = r"D:\Projects\Capstone Project\Vision based hand sign detection & classification usig DL models\runs\hand_landmark_MLP_original_structure-1\label_encoder.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD ENCODER =================
le = joblib.load(ENCODER_PATH)
NUM_CLASSES = len(le.classes_)
print(f"Loaded classes: {le.classes_}")

# ================= MODEL =================
class ASL_MLP(nn.Module):
    def __init__(self, num_classes):
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

model = ASL_MLP(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
model.eval()

print("✅ Model Loaded Successfully!")

# ================= SPEECH ENGINE =================
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 0.9)

# Get available voices
voices = engine.getProperty('voices')
if voices:
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)

# Create a queue for speech requests
speech_queue = queue.Queue()
speech_thread_running = True

def speech_worker():
    global engine, speech_thread_running
    while speech_thread_running:
        try:
            text = speech_queue.get(timeout=0.5)
            if text:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"Speech error: {e}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Speech worker error: {e}")

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_async(text):
    if text and text.strip():
        speech_queue.put(text.strip())

# ================= NORMALIZATION =================
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    landmarks = landmarks - landmarks[0]
    max_value = np.max(np.abs(landmarks))
    if max_value != 0:
        landmarks = landmarks / max_value
    return landmarks.flatten()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= GUI SETUP =================
root = tk.Tk()
root.title("🚀 SignWise+ Advanced Multi-Mode")
root.geometry("1400x900+50+50")
root.configure(bg="#1a1a1a")

# ================= MAIN CONTAINER =================
main_container = tk.Frame(root, bg="#1a1a1a")
main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# ================= LEFT PANEL - MODE SELECTION =================
left_panel = tk.Frame(main_container, bg="#2a2a2a", width=250, relief=tk.RAISED, bd=2)
left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
left_panel.pack_propagate(False)

# Mode Selection Title
tk.Label(
    left_panel,
    text="🎮 MODE SELECTION",
    font=("Arial", 16, "bold"),
    fg="cyan",
    bg="#2a2a2a"
).pack(pady=15)

mode_var = tk.StringVar(value="right")

def set_mode(mode):
    mode_var.set(mode)
    # Update button colors
    colors = {
        "right": "#00aa00",
        "left": "#aaaa00",
        "dual": "#00aaaa",
        "free": "#aa00aa"
    }
    for btn_mode, btn in mode_buttons.items():
        if btn_mode == mode:
            btn.config(bg=colors[btn_mode], fg="white")
        else:
            btn.config(bg="#333333", fg="#cccccc")

mode_buttons = {}

# Mode buttons with descriptions
modes = [
    ("👉 RIGHT MODE", "right", "Only right hand accepted", "#00aa00"),
    ("👈 LEFT MODE", "left", "Only left hand accepted", "#aaaa00"),
    ("🖐️ DUAL MODE", "dual", "Both hands tracked", "#00aaaa"),
    ("✨ FREE MODE", "free", "First hand input", "#aa00aa")
]

for text, mode, desc, color in modes:
    # Mode button
    btn = tk.Button(
        left_panel,
        text=text,
        command=lambda m=mode: set_mode(m),
        bg=color if mode == "right" else "#333333",
        fg="white",
        font=("Arial", 12, "bold"),
        padx=15,
        pady=10,
        cursor="hand2",
        width=18,
        relief=tk.RAISED,
        bd=3
    )
    btn.pack(pady=5)
    mode_buttons[mode] = btn
    
    # Mode description
    tk.Label(
        left_panel,
        text=desc,
        font=("Arial", 9),
        fg="#cccccc",
        bg="#2a2a2a"
    ).pack(pady=(0, 10))

# Current mode indicator
current_mode_frame = tk.Frame(left_panel, bg="#2a2a2a")
current_mode_frame.pack(pady=20)

tk.Label(
    current_mode_frame,
    text="Current Mode:",
    font=("Arial", 10),
    fg="white",
    bg="#2a2a2a"
).pack()

mode_indicator = tk.Label(
    current_mode_frame,
    text="RIGHT MODE",
    font=("Arial", 14, "bold"),
    fg="#00aa00",
    bg="#2a2a2a"
)
mode_indicator.pack()

# ================= CENTER PANEL - VIDEO =================
center_panel = tk.Frame(main_container, bg="#1a1a1a")
center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Video frame
video_frame = tk.Frame(center_panel, bg="#2a2a2a", bd=3, relief=tk.SUNKEN)
video_frame.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

video_label = tk.Label(video_frame, bg="#1a1a1a")
video_label.pack(padx=10, pady=10, expand=True)

# Prediction and Confidence
pred_frame = tk.Frame(center_panel, bg="#1a1a1a")
pred_frame.pack(pady=10, fill=tk.X)

prediction_label = tk.Label(
    pred_frame,
    text="Prediction: ---",
    font=("Arial", 24, "bold"),
    fg="cyan",
    bg="#1a1a1a"
)
prediction_label.pack()

confidence_label = tk.Label(
    pred_frame,
    text="Confidence: ---",
    font=("Arial", 14),
    fg="white",
    bg="#1a1a1a"
)
confidence_label.pack()

# UPDATED: Detection Status Label (NEW)
detection_label = tk.Label(
    center_panel,
    text="Detection: ---",
    font=("Arial", 14, "bold"),
    fg="yellow",
    bg="#1a1a1a"
)
detection_label.pack()

# Warning Label
warning_label = tk.Label(
    center_panel,
    text="",
    font=("Arial", 14, "bold"),
    fg="red",
    bg="#1a1a1a",
    height=1
)
warning_label.pack()

# ================= RIGHT PANEL - FPS & STABILITY =================
right_panel = tk.Frame(main_container, bg="#2a2a2a", width=280, relief=tk.RAISED, bd=2)
right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
right_panel.pack_propagate(False)

# FPS Controller Section
tk.Label(
    right_panel,
    text="⚡ FPS CONTROL",
    font=("Arial", 16, "bold"),
    fg="cyan",
    bg="#2a2a2a"
).pack(pady=15)

fps_frame = tk.Frame(right_panel, bg="#2a2a2a")
fps_frame.pack(pady=10, padx=15, fill=tk.X)

tk.Label(fps_frame, text="Target FPS:", fg="white", bg="#2a2a2a", font=("Arial", 11)).pack()

fps_scale = tk.Scale(
    fps_frame,
    from_=5,
    to=60,
    orient="horizontal",
    length=220,
    bg="#2a2a2a",
    fg="cyan",
    highlightbackground="#2a2a2a",
    troughcolor="#4a4a4a",
    resolution=1
)
fps_scale.set(30)
fps_scale.pack(pady=5)

# FPS Display
fps_display_frame = tk.Frame(right_panel, bg="#2a2a2a")
fps_display_frame.pack(pady=10)

tk.Label(
    fps_display_frame,
    text="Current FPS:",
    font=("Arial", 11),
    fg="white",
    bg="#2a2a2a"
).pack()

fps_value_label = tk.Label(
    fps_display_frame,
    text="30",
    font=("Arial", 24, "bold"),
    fg="lime",
    bg="#2a2a2a"
)
fps_value_label.pack()

# Separator
tk.Frame(right_panel, bg="#4a4a4a", height=2).pack(fill=tk.X, pady=15, padx=20)

# Stability Controller Section
tk.Label(
    right_panel,
    text="🎯 STABILITY",
    font=("Arial", 16, "bold"),
    fg="yellow",
    bg="#2a2a2a"
).pack(pady=15)

stab_frame = tk.Frame(right_panel, bg="#2a2a2a")
stab_frame.pack(pady=10, padx=15, fill=tk.X)

tk.Label(stab_frame, text="Stability Frames:", fg="white", bg="#2a2a2a", font=("Arial", 11)).pack()

stability_scale = tk.Scale(
    stab_frame,
    from_=3,
    to=20,
    orient="horizontal",
    length=220,
    bg="#2a2a2a",
    fg="yellow",
    highlightbackground="#2a2a2a",
    troughcolor="#4a4a4a"
)
stability_scale.set(8)
stability_scale.pack(pady=5)

# Stability Display
stab_display_frame = tk.Frame(right_panel, bg="#2a2a2a")
stab_display_frame.pack(pady=10)

tk.Label(
    stab_display_frame,
    text="Current Setting:",
    font=("Arial", 11),
    fg="white",
    bg="#2a2a2a"
).pack()

stab_value_label = tk.Label(
    stab_display_frame,
    text="8 frames",
    font=("Arial", 20, "bold"),
    fg="yellow",
    bg="#2a2a2a"
)
stab_value_label.pack()

# Hand Status in Right Panel
tk.Frame(right_panel, bg="#4a4a4a", height=2).pack(fill=tk.X, pady=15, padx=20)

hand_status_frame = tk.Frame(right_panel, bg="#2a2a2a")
hand_status_frame.pack(pady=10, fill=tk.X)

# Left Hand Status
left_status = tk.Frame(hand_status_frame, bg="#2a2a2a")
left_status.pack(pady=5)

tk.Label(left_status, text="🤚 LEFT HAND", font=("Arial", 10, "bold"), fg="yellow", bg="#2a2a2a").pack()
left_hand_label = tk.Label(left_status, text="---", font=("Arial", 11), fg="white", bg="#2a2a2a")
left_hand_label.pack()

# Right Hand Status
right_status = tk.Frame(hand_status_frame, bg="#2a2a2a")
right_status.pack(pady=5)

tk.Label(right_status, text="✋ RIGHT HAND", font=("Arial", 10, "bold"), fg="lime", bg="#2a2a2a").pack()
right_hand_label = tk.Label(right_status, text="---", font=("Arial", 11), fg="white", bg="#2a2a2a")
right_hand_label.pack()

# ================= BOTTOM PANEL - SENTENCE & BUTTONS =================
bottom_panel = tk.Frame(root, bg="#1a1a1a")
bottom_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

# Sentence Display
sentence_text = tk.StringVar()
sentence_text.set("")

sentence_label = tk.Label(
    bottom_panel,
    textvariable=sentence_text,
    font=("Arial", 24, "bold"),
    fg="lime",
    bg="#1a1a1a",
    wraplength=1300,
    height=2
)
sentence_label.pack(pady=5)

# Buttons Frame
button_frame = tk.Frame(bottom_panel, bg="#333333", relief=tk.RAISED, bd=3)
button_frame.pack(pady=10, fill=tk.X)

button_style = {
    "font": ("Arial", 14, "bold"),
    "bg": "#4a4a4a",
    "fg": "white",
    "activebackground": "#6a6a6a",
    "activeforeground": "white",
    "relief": tk.RAISED,
    "bd": 3,
    "padx": 40,
    "pady": 12,
    "cursor": "hand2",
    "width": 12
}

def speak_sentence():
    global current_sentence
    if current_sentence.strip():
        speak_async(current_sentence)
        status_label.config(text="🔊 Speaking...", fg="green")
        speak_btn.config(bg="#00aa00")
        root.after(200, lambda: speak_btn.config(bg="#4a4a4a"))

def clear_sentence():
    global current_sentence
    if current_sentence:
        current_sentence = ""
        status_label.config(text="🗑 Sentence cleared", fg="yellow")
        sentence_text.set("")
        clear_btn.config(bg="#aa5500")
        root.after(200, lambda: clear_btn.config(bg="#4a4a4a"))
    else:
        status_label.config(text="⚠️ Sentence empty", fg="orange")

def backspace():
    global current_sentence
    if current_sentence:
        current_sentence = current_sentence[:-1]
        status_label.config(text="⌫ Deleted", fg="white")
        sentence_text.set(current_sentence)
        backspace_btn.config(bg="#aa0000")
        root.after(200, lambda: backspace_btn.config(bg="#4a4a4a"))
    else:
        status_label.config(text="⚠️ Nothing to delete", fg="orange")

speak_btn = tk.Button(button_frame, text="🔊 SPEAK", command=speak_sentence, **button_style)
speak_btn.pack(side=tk.LEFT, padx=15, pady=10, expand=True)

clear_btn = tk.Button(button_frame, text="🗑 CLEAR", command=clear_sentence, **button_style)
clear_btn.pack(side=tk.LEFT, padx=15, pady=10, expand=True)

backspace_btn = tk.Button(button_frame, text="⌫ DELETE", command=backspace, **button_style)
backspace_btn.pack(side=tk.LEFT, padx=15, pady=10, expand=True)

# Status Label
status_label = tk.Label(
    bottom_panel,
    text="Status: Ready",
    font=("Arial", 11, "bold"),
    fg="yellow",
    bg="#1a1a1a"
)
status_label.pack(pady=5)

# Instruction Label
tk.Label(
    bottom_panel,
    text="✋ Stop = Speak | del = Backspace | Space = Space | Left hand auto-flipped",
    font=("Arial", 11),
    fg="orange",
    bg="#1a1a1a"
).pack(pady=5)

# ================= CAMERA =================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ================= LOGIC VARIABLES =================
CONFIDENCE_THRESHOLD = 0.85
prediction_history = deque(maxlen=7)
stable_prediction = ""
stable_count = 0
stop_hold_start = None
STOP_HOLD_DURATION = 1.5
last_speak_time = 0
SPEAK_COOLDOWN = 2
prev_time = time.time()
frame_times = deque(maxlen=30)
last_del_time = 0
DEL_COOLDOWN = 0.8
del_performed = False
current_sentence = ""
active_hand_in_free_mode = None

# ================= MAIN LOOP =================
def update_frame():
    global stable_prediction, stable_count, current_sentence
    global stop_hold_start, prev_time, last_speak_time
    global last_del_time, del_performed, frame_times
    global active_hand_in_free_mode

    # Update mode indicator
    current_mode = mode_var.get()
    mode_colors = {"right": "#00aa00", "left": "#aaaa00", "dual": "#00aaaa", "free": "#aa00aa"}
    mode_indicator.config(text=f"{current_mode.upper()} MODE", fg=mode_colors[current_mode])

    # FPS Control
    target_fps = fps_scale.get()
    target_frame_time = 1.0 / target_fps if target_fps > 0 else 0.033
    
    current_time = time.time()
    frame_time = current_time - prev_time
    
    frame_times.append(frame_time)
    avg_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
    fps_value_label.config(text=f"{int(avg_fps)}")
    stab_value_label.config(text=f"{stability_scale.get()} frames")
    
    if frame_time < target_frame_time:
        time.sleep(target_frame_time - frame_time)
    
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # UPDATED: Detection status variables
    display_prediction = "---"
    display_conf = "---"
    hand_type_display = "No Hand"
    hand_color = (0, 255, 0)
    warning_text = ""
    detection_status = "🔴 NO HAND DETECTED"
    detection_color = "red"
    
    left_hand_pred = "---"
    left_hand_conf = "---"
    right_hand_pred = "---"
    right_hand_conf = "---"
    
    del_performed = False

    if results.multi_hand_landmarks:
        # UPDATED: MediaPipe detected a hand
        detection_status = "🟢 HAND DETECTED"
        detection_color = "lime"
        
        hand_predictions = {}
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if results.multi_handedness:
                handedness = results.multi_handedness[idx]
                hand_label_value = handedness.classification[0].label
                
                if hand_label_value == 'Left':
                    hand_type = 'Left'
                    hand_color_current = (0, 255, 255)
                    landmarks = [[1.0 - lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                else:
                    hand_type = 'Right'
                    hand_color_current = (0, 255, 0)
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=hand_color_current, thickness=2),
                    mp_draw.DrawingSpec(color=(255,255,255), thickness=1)
                )
                
                if len(landmarks) == 21:
                    lm_array = normalize_landmarks(landmarks)
                    input_tensor = torch.tensor(lm_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        confidence, pred = torch.max(probs, 1)

                    if confidence.item() > CONFIDENCE_THRESHOLD:
                        predicted_label = le.inverse_transform([pred.item()])[0]
                        hand_predictions[hand_type] = {
                            'label': predicted_label,
                            'confidence': confidence.item()
                        }

        # Update hand labels
        if 'Left' in hand_predictions:
            left_hand_pred = hand_predictions['Left']['label']
            left_hand_conf = f"{hand_predictions['Left']['confidence']:.2f}"
        if 'Right' in hand_predictions:
            right_hand_pred = hand_predictions['Right']['label']
            right_hand_conf = f"{hand_predictions['Right']['confidence']:.2f}"
        
        left_hand_label.config(text=f"{left_hand_pred} ({left_hand_conf})")
        right_hand_label.config(text=f"{right_hand_pred} ({right_hand_conf})")

        # UPDATED: Check if any gesture was recognized
        if hand_predictions:
            # Mode handling
            if current_mode == "right":
                if 'Right' in hand_predictions:
                    process_hand_input(hand_predictions['Right']['label'], hand_predictions['Right']['confidence'], frame)
                    display_prediction = hand_predictions['Right']['label']
                    display_conf = f"{hand_predictions['Right']['confidence']:.2f}"
                    hand_type_display = "✅ RIGHT MODE Active"
                    hand_color = (0, 255, 0)
                elif 'Left' in hand_predictions:
                    warning_text = "⚠️ SHOW RIGHT HAND ONLY!"
                    hand_type_display = "❌ Wrong hand - Left detected"
                    hand_color = (0, 0, 255)

            elif current_mode == "left":
                if 'Left' in hand_predictions:
                    process_hand_input(hand_predictions['Left']['label'], hand_predictions['Left']['confidence'], frame)
                    display_prediction = hand_predictions['Left']['label']
                    display_conf = f"{hand_predictions['Left']['confidence']:.2f}"
                    hand_type_display = "✅ LEFT MODE Active"
                    hand_color = (0, 255, 255)
                elif 'Right' in hand_predictions:
                    warning_text = "⚠️ SHOW LEFT HAND ONLY!"
                    hand_type_display = "❌ Wrong hand - Right detected"
                    hand_color = (0, 0, 255)

            elif current_mode == "dual":
                display_prediction = f"L:{left_hand_pred} R:{right_hand_pred}"
                display_conf = f"L:{left_hand_conf} R:{right_hand_conf}"
                hand_type_display = "🌐 DUAL MODE"

            elif current_mode == "free":
                if hand_predictions:
                    if active_hand_in_free_mode is None:
                        active_hand_in_free_mode = 'Right' if 'Right' in hand_predictions else 'Left'
                    
                    if active_hand_in_free_mode in hand_predictions:
                        process_hand_input(hand_predictions[active_hand_in_free_mode]['label'], 
                                         hand_predictions[active_hand_in_free_mode]['confidence'], frame)
                        display_prediction = hand_predictions[active_hand_in_free_mode]['label']
                        display_conf = f"{hand_predictions[active_hand_in_free_mode]['confidence']:.2f}"
                        hand_type_display = f"✨ FREE MODE (Using {active_hand_in_free_mode})"
                    else:
                        display_prediction = "Wait for hand"
                        hand_type_display = f"✨ Waiting for {active_hand_in_free_mode}"
        else:
            # UPDATED: Hand detected but no gesture recognized
            display_prediction = "No Gesture"
            display_conf = "Low Confidence"
            hand_type_display = "Hand detected - No gesture"
            hand_color = (255, 165, 0)  # Orange

    else:
        # No hand detected by MediaPipe
        active_hand_in_free_mode = None
        left_hand_label.config(text="---")
        right_hand_label.config(text="---")
        display_prediction = "No Hand"
        display_conf = "---"
        hand_type_display = "No hand in frame"
        hand_color = (255, 0, 0)  # Red

    # Update detection label
    detection_label.config(text=f"Detection: {detection_status}", fg=detection_color)
    warning_label.config(text=warning_text)

    # Draw on frame
    cv2.putText(frame, f"FPS: {int(avg_fps)}/{target_fps}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"MODE: {current_mode.upper()}", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, hand_type_display, (20, 115), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

    # Update GUI
    prediction_label.config(text=f"Prediction: {display_prediction}")
    confidence_label.config(text=f"Confidence: {display_conf}")
    sentence_text.set(current_sentence)

    prev_time = current_time

    # Show frame
    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    video_label.imgtk = img
    video_label.configure(image=img)

    root.after(10, update_frame)

def process_hand_input(predicted_label, confidence_value, frame):
    global stable_prediction, stable_count, current_sentence
    global stop_hold_start, last_speak_time, last_del_time
    
    prediction_history.append(predicted_label)
    
    if predicted_label == "DEL":
        current_time = time.time()
        if current_time - last_del_time > DEL_COOLDOWN:
            if current_sentence:
                current_sentence = current_sentence[:-1]
                status_label.config(text="⌫ Deleted", fg="white")
                last_del_time = current_time
                sentence_text.set(current_sentence)
            else:
                status_label.config(text="⚠️ Nothing to delete", fg="orange")
        cv2.putText(frame, "COMMAND: DELETE", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
        stable_prediction = ""
        stable_count = 0
    
    elif predicted_label == "Stop":
        if stop_hold_start is None:
            stop_hold_start = time.time()
            status_label.config(text=f"⏳ Hold to speak...", fg="orange")
        elif time.time() - stop_hold_start > STOP_HOLD_DURATION:
            if current_sentence.strip() and time.time() - last_speak_time > SPEAK_COOLDOWN:
                speak_async(current_sentence)
                last_speak_time = time.time()
                status_label.config(text="🔊 Speaking...", fg="green")
                stop_hold_start = None
        cv2.putText(frame, "COMMAND: STOP", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
        stable_prediction = ""
        stable_count = 0
    
    elif predicted_label == "Space":
        if predicted_label == stable_prediction:
            stable_count += 1
        else:
            stable_prediction = predicted_label
            stable_count = 1

        if stable_count >= stability_scale.get():
            current_sentence += " "
            status_label.config(text="␣ Added space", fg="white")
            stable_prediction = ""
            stable_count = 0
            sentence_text.set(current_sentence)
        
        cv2.putText(frame, f"Stable: {stable_count}/{stability_scale.get()}", 
                   (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    else:
        if predicted_label == stable_prediction:
            stable_count += 1
        else:
            stable_prediction = predicted_label
            stable_count = 1

        if stable_count >= stability_scale.get():
            if len(current_sentence) < 60:
                current_sentence += predicted_label
                status_label.config(text=f"✅ Added: {predicted_label}", fg="white")
                sentence_text.set(current_sentence)
            stable_prediction = ""
            stable_count = 0

        cv2.putText(frame, f"Stable: {stable_count}/{stability_scale.get()}", 
                   (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

def on_closing():
    global speech_thread_running
    speech_thread_running = False
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

print("\n" + "="*60)
print("🚀 SignWise+ Started Successfully!")
print(f"📊 Loaded {len(le.classes_)} classes")
print("="*60)
print("✅ New Detection System:")
print("   🔴 NO HAND DETECTED - MediaPipe can't see any hand")
print("   🟢 HAND DETECTED - MediaPipe sees hand(s)")
print("   📊 No Gesture - Hand detected but no sign recognized")
print("="*60)

# Start the application
update_frame()
root.mainloop()