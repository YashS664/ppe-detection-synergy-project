from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import os
import uuid
import base64
import time
from app.inference import predict, is_overlapping
from app.email_utils import send_email_alert
from app.database import db
from app.config import (
    PERSON_CLASS_ID, 
    STATIC_DIR, 
    UPLOAD_DIR, 
    INFERENCE_INTERVAL, 
    VIOLATION_WAIT_TIME, 
    BASE_URL,
    PIPELINE_MODE
)
from dotenv import load_dotenv

load_dotenv()
from app.model import init_models

app = FastAPI(title="PPE Detection API")

from app.rag.rag_router import router as rag_router
app.include_router(rag_router)

@app.on_event("startup")
async def startup_event():
    init_models()

# Mount static directory for processed results (preserved for images)
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=STATIC_DIR), name="results")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


# Helper: Draw bounding boxes on frame
def draw_detections(img, detections):
    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        is_violation = det["violation"]

        color = (0, 0, 255) if is_violation else (0, 255, 0)
        track_id = det.get("track_id")
        global_id = det.get("global_id")
        
        id_str = ""
        if track_id is not None:
            id_str += f" T:{track_id}"
        if global_id is not None:
            id_str += f" REID:{global_id}"
            
        label = f"{class_name}{id_str} ({confidence})"

        # Draw box
        p1 = (int(bbox["x1"]), int(bbox["y1"]))
        p2 = (int(bbox["x2"]), int(bbox["y2"]))
        cv2.rectangle(img, p1, p2, color, 2)

        # Draw background for text
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_w, text_h = text_size
        cv2.rectangle(img, p1, (p1[0] + text_w, p1[1] - text_h - 10), color, -1)

        # Draw text
        cv2.putText(img, label, (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img


# Helper: Check file type
def is_image(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))


def is_video(filename):
    return filename.lower().endswith((".mp4", ".avi", ".mov"))


# Image Processing
async def process_image(file: UploadFile):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = predict(img)
    
    # RE-ID Logic for singular image
    for det in result["detections"]:
        if det["class_id"] == PERSON_CLASS_ID:
            emb = det.get("embedding")
            if emb:
                gid, sim = db.find_match(emb)
                if gid:
                    print(f"RE-ID (Image): Matched to Global ID {gid} (sim: {sim:.3f})")
                    det["global_id"] = gid
                    db.update_last_seen(gid)
                else:
                    print(f"RE-ID (Image): No match found (best sim: {sim:.3f}). Saving new person...")
                    gid = db.save_person(emb, metadata={"source": "image_upload"})
                    if gid:
                        print(f"RE-ID (Image): New person, assigned Global ID {gid}")
                        det["global_id"] = gid
                    else:
                        print("RE-ID (Image): FAILED to save person to database.")
            else:
                print("RE-ID (Image): No embedding found for detected person.")
    
    #  Draw detections on the image
    img = draw_detections(img, result["detections"])

    #  Send email if violation detected in image
    if result["violations_detected"]:
        violation_summary = []
        for det in result["detections"]:
            if det.get("violation") and det.get("class_id") == PERSON_CLASS_ID:
                gid = det.get("global_id", "New")
                missing = det.get("missing_gear", ["General Violation"])
                violation_summary.append(f"- Person REID:{gid} is missing: {', '.join(missing)}")
                if isinstance(gid, int):
                    db.update_email_alert_status(gid)
        
        email_msg = "PPE Violation(s) Detected in Uploaded Image:\n\n" + "\n".join(violation_summary)
        print(f"Sending email alert for image violation: {email_msg}")
        send_email_alert(img, email_msg)

    #  Encode into base64 for direct return
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        **result,
        "processed_image": f"data:image/jpeg;base64,{img_base64}"
    }


#  Video Stream Generator
def gen_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    current_detections = []
    current_status = "SAFE"
    status_color = (0, 255, 0)

    violation_start_time = None
    violation_threshold = VIOLATION_WAIT_TIME
    alerted_ids = set() # Track IDs for which email has already been sent
    id_violation_starts = {} # track_id -> first_time_seen_violating
    
    # RE-ID Person Mapping (track_id -> global_id)
    track_to_global = {} 


    if not cap.isOpened():
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        
        #  Inference every Nth frame
        if frame_count % INFERENCE_INTERVAL == 0:
            result = predict(frame, persist=True)
            current_detections = result["detections"]
            
            # Persistent RE-ID Logic
            for det in current_detections:
                tid = det.get("track_id")
                if tid is not None and det["class_id"] == PERSON_CLASS_ID:
                    if tid not in track_to_global:
                        emb = det.get("embedding")
                        if emb:
                            gid, sim = db.find_match(emb)
                            if gid:
                                print(f"RE-ID: Matched track {tid} to Global ID {gid} (sim: {sim:.3f})")
                                track_to_global[tid] = gid
                                db.update_last_seen(gid)
                            else:
                                gid = db.save_person(emb, metadata={"source": video_path})
                                print(f"RE-ID: New person detected, assigned Global ID {gid}")
                                track_to_global[tid] = gid
                    
                    det["global_id"] = track_to_global.get(tid)

                    if det["class_id"] == PERSON_CLASS_ID:
                        vlm_scores = det.get("vlm_scores", {})
                        global_id = det.get("global_id")

                        if PIPELINE_MODE == "VLM" and global_id and vlm_scores:
                            current_time = time.time()
                            email_status = db.get_email_status(global_id)

                            last_sent = email_status.get("last_sent") if email_status else None
                            should_log   = (last_sent is None) or (
                            current_time - time.mktime(
                                time.strptime(last_sent, "%Y-%m-%d %H:%M:%S")
                            ) > VIOLATION_WAIT_TIME)

                            if should_log:
                                if not vlm_scores.get("hardhat"):
                                    db.save_violation(
                                        person_id=global_id, 
                                        violation_type="hardhat",
                                        confidence=abs(vlm_scores.get("hardhat_confidence", 0))
                                    )
                                if not vlm_scores.get("vest"):
                                    db.save_violation(
                                        person_id=global_id, 
                                        violation_type="vest",
                                        confidence=abs(vlm_scores.get("vest_confidence", 0))
                                    )
                    
                        elif PIPELINE_MODE != "VLM":
                            violation_class_names = ["NO-Hardhat", "NO-Safety Vest"]
                            for vdet in current_detections:
                                if vdet.get("class_name") in violation_class_names:
                                    if is_overlapping(vdet["bbox"], det["bbox"]):
                                        violation_type = "hardhat" if "Hardhat" in vdet["class_name"] else "vest"
                                        db.save_violation(
                                            person_id=global_id, 
                                            violation_type=violation_type,
                                            confidence=vdet.get("confidence", 0)
                                        )


                    # Print confidence details only in VLM mode
                    if PIPELINE_MODE == "VLM":
                        vlm = det.get("vlm_scores", {})
                        h_score = vlm.get("hardhat_confidence", 0)
                        v_score = vlm.get("vest_confidence", 0)
                        print(f"REID:{det['global_id']} [H:{h_score:+.2f}, V:{v_score:+.2f}]")
            
            if result["violations_detected"]:
                current_time = time.time()
                
                # Identify currently violating persons
                current_violating_ids = {det["track_id"] for det in current_detections 
                                        if det["violation"] and det.get("track_id") is not None}
                
                # Remove timers for IDs no longer violating
                ids_to_forget = set(id_violation_starts.keys()) - current_violating_ids
                for rid in ids_to_forget:
                    del id_violation_starts[rid]

                # Update status
                current_status = "MONITORING VIOLATION..."
                status_color = (0, 165, 255) # orange

                new_alerts_to_send = []
                for tid in current_violating_ids:
                    if tid in alerted_ids:
                        continue
                        
                    # Start timer if new
                    if tid not in id_violation_starts:
                        id_violation_starts[tid] = current_time
                    
                    # Check duration
                    elif current_time - id_violation_starts[tid] >= violation_threshold:
                        new_alerts_to_send.append(tid)

                if new_alerts_to_send:
                    # Collect details for the email alert
                    violation_summary = []
                    for tid in new_alerts_to_send:
                        # Find the detection for this track_id to get its missing gear and global_id
                        for det in current_detections:
                            if det.get("track_id") == tid and det.get("violation") and det.get("class_id") == PERSON_CLASS_ID:
                                gid = det.get("global_id", "Unknown")
                                missing = det.get("missing_gear", ["General Violation"])
                                
                                if PIPELINE_MODE == "VLM":
                                    vlm = det.get("vlm_scores", {})
                                    h_score = vlm.get("hardhat_confidence", 0)
                                    v_score = vlm.get("vest_confidence", 0)
                                    violation_summary.append(f"- Person REID:{gid} (Track:{tid}) is missing: {', '.join(missing)} [Scores: H={h_score:+.2f}, V={v_score:+.2f}]")
                                else:
                                    violation_summary.append(f"- Person REID:{gid} (Track:{tid}) is missing: {', '.join(missing)}")
                                # Update DB if we have a valid global_id
                                if isinstance(gid, int):
                                    db.update_email_alert_status(gid)
                                break # Found the matching detection
                    
                    email_msg = "PPE Violation(s) Detected in Video Stream:\n\n" + "\n".join(violation_summary)
                    print(f"Sending email alert for video: {email_msg}")
                    
                    current_status = "VIOLATION DETECTED"
                    status_color = (0, 0, 255)
                    
                    drawn_frame = draw_detections(frame.copy(), current_detections)
                    send_email_alert(drawn_frame, email_msg)
                    
                    # Mark as alerted
                    alerted_ids.update(new_alerts_to_send)
            else:
                # Reset all timers if total safety
                id_violation_starts = {}
                current_status = "SAFE"
                status_color = (0, 255, 0)


            
            # Log for the user in console
            print(f"Frame {frame_count}: {current_status} | Detections: {len(current_detections)}")

        # Draw detections
        frame = draw_detections(frame, current_detections)
        
        #  Draw Status Banner on top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(frame, f"STATUS: {current_status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"FRAME: {frame_count}", (frame.shape[1] - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        frame_count += 1

        #  Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    cap.release()


#  Video Processing (Fast Return for Streaming)
async def process_video_init(file: UploadFile):
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}_in.mp4"
    input_path = os.path.join(UPLOAD_DIR, input_filename)

    # Save original video temporarily
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    return {
        "type": "video",
        "file_id": file_id,
        "stream_url": f"{BASE_URL}/stream/{file_id}"
    }


#  Streaming Endpoint
@app.get("/stream/{file_id}")
async def stream_video(file_id: str):
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}_in.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return StreamingResponse(gen_frames(video_path), 
                             media_type="multipart/x-mixed-replace; boundary=frame")


#  Main API Endpoint
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    filename = file.filename

    if is_image(filename):
        result = await process_image(file)
        return {
            "type": "image",
            **result
        }

    elif is_video(filename):
        #  Return stream URL immediately instead of processing whole video
        result = await process_video_init(file)
        return result

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload image or video."
        )


# Health Check
@app.get("/")
def health():
    return {"status": "API is running"}

# Dashboard Data Endpoint
@app.get("/dashboard")
async def get_dashboard():
    data = db.get_dashboard_data()
    return data