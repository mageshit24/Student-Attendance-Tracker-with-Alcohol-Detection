import os, cv2, time, datetime, random
import numpy as np, pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox as mess
from PIL import Image
from pymongo import MongoClient
import serial
import threading

# ----- CONFIG -----
SERIAL_PORT = "COM3"  # Replace with your ESP32 Serial port
BAUD_RATE = 115200
ALCOHOL_THRESHOLD = 0.90
SMOKE_THRESHOLD = 0.90
STAGGER_HEAD_THRESHOLD = 15  # pixels for head movement detection

# ----- SERIAL INIT -----
latest_sensors = {"device":"esp32_01","timestamp_ms":None,"alcohol_norm":0.0,"smoke_norm":0.0}

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow ESP32 to initialize
    # Discard boot messages
    for _ in range(10):
        ser.readline()
    print("Serial connected to ESP32")
except Exception as e:
    ser = None
    print("Serial connection failed:", e)

# ----- SERIAL THREAD -----
def serial_read_thread():
    global latest_sensors
    if ser is None: 
        return
    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                parts = line.split(",")
                if len(parts) == 2:
                    try:
                        alcohol = float(parts[0])
                        smoke = float(parts[1])
                        latest_sensors = {
                            "device": "esp32_01",
                            "timestamp_ms": int(time.time() * 1000),
                            "alcohol_norm": alcohol,
                            "smoke_norm": smoke
                        }
                        print(f"ESP32 sensors -> Alcohol: {alcohol:.2f}, Smoke: {smoke:.2f}")
                    except ValueError:
                        pass
        except Exception as e:
            print("Serial read error:", e)
        time.sleep(0.05)

threading.Thread(target=serial_read_thread, daemon=True).start()

# ----- MONGODB INIT -----
MONGO_URI = "mongodb+srv://Attendance_Taker:Magesh%402004@cluster0.ydizft5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
try:
    mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo.server_info()
    mongo_coll = mongo["attendance_db"]["attendance_records"]
    print("MongoDB connected")
except Exception as e:
    mongo_coll = None
    print("MongoDB connection failed:", e)

# ----- UTILS -----
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def safe_float(val, default=0.0):
    try: return float(val)
    except (TypeError, ValueError): return default

def fetch_sensors_from_esp32():
    return latest_sensors.copy()

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess._show(title='Missing', message='haarcascade_frontalface_default.xml not found')
        window.destroy()

# ----- IMAGE REGISTRATION -----
def TakeImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImage/")
    assure_path_exists("StudentDetails/")
    Id, name = txt.get().strip(), txt2.get().strip()
    if not Id.isdigit() or not name.replace(" ","").isalpha():
        mess._show(title='Error', message='Invalid ID/Name'); return

    student_file = "StudentDetails/StudentDetails.csv"
    if os.path.isfile(student_file):
        df = pd.read_csv(student_file); serial_num = df.shape[0]
    else:
        df = pd.DataFrame(columns=['SERIAL NO.','ID','NAME']); serial_num = 0

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sampleNum = 0

    while True:
        ret, img = cam.read()
        if not ret: break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            sampleNum += 1
            cv2.imwrite(f"TrainingImage/{name}.{serial_num}.{Id}.{sampleNum}.jpg", gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imshow("Taking Images", img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 50: break

    cam.release(); cv2.destroyAllWindows()
    df = pd.concat([df, pd.DataFrame([[serial_num,int(Id),name]], columns=df.columns)])
    df.to_csv(student_file, index=False)
    message1.config(text=f"Images Taken for ID: {Id}")
    mess._show("Success", f"Images for {name} saved successfully!")

def getImagesAndLabels(path):
    faces, Ids = [], []
    for imagePath in [os.path.join(path, f) for f in os.listdir(path)]:
        try:
            pil = Image.open(imagePath).convert('L')
            faces.append(np.array(pil, 'uint8'))
            Ids.append(int(os.path.split(imagePath)[-1].split(".")[1]))
        except: pass
    return faces, Ids

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Ids = getImagesAndLabels("TrainingImage")
    if not faces: mess._show(title='Error', message='No images to train'); return
    recognizer.train(faces, np.array(Ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    message1.config(text="Training complete")
    mess._show("Success", "Face recognizer trained successfully!")

# ----- REACTION TEST -----
def run_reaction_test(timeout_ms=3000, min_delay_ms=500, max_delay_ms=2000):
    win_name = "Reaction Test - Press 'r' when PROMPT appears"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    blank = np.zeros((300,600,3),dtype=np.uint8)
    cv2.putText(blank, "Get ready...", (40,150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.imshow(win_name, blank); cv2.waitKey(1)
    delay = random.randint(min_delay_ms, max_delay_ms)
    start_wait = time.time()
    while (time.time()-start_wait)*1000.0 < delay:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow(win_name); return None
    prompt = np.zeros((300,600,3),dtype=np.uint8)
    cv2.putText(prompt, "PRESS 'r' NOW!", (40,150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    cv2.imshow(win_name, prompt); cv2.waitKey(1)
    t0 = time.time(); reacted = False
    while (time.time()-t0)*1000.0 < timeout_ms:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'): rt = int((time.time()-t0)*1000.0); reacted = True; break
        if k == ord('q'): cv2.destroyWindow(win_name); return None
    cv2.destroyWindow(win_name)
    return rt if reacted else None

# ----- ATTENDANCE -----
def TrackImages():
    check_haarcascadefile()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_path = "TrainingImageLabel/Trainner.yml"
    if not os.path.isfile(model_path): mess._show(title='Error', message='Trainer model not found. Train first.'); return
    recognizer.read(model_path)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    student_file = "StudentDetails/StudentDetails.csv"
    if not os.path.isfile(student_file): mess._show(title='Error', message='No student details found. Register students first.'); return
    df = pd.read_csv(student_file)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened(): mess._show(title='Error', message='Camera not accessible.'); return

    expected_ids = set(df['ID'].astype(int).tolist()) if not df.empty else set()
    tested_ids = set(); attendance = []; prev_head_pos = {}

    for k in tv.get_children(): tv.delete(k)

    while True:
        ret, im = cam.read()
        if not ret: break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.2,5)
        for (x,y,w,h) in faces:
            serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
            name, Id = "Unknown", 0
            if conf < 50:
                try: row = df.loc[df['SERIAL NO.']==serial].iloc[0]; name=row['NAME']; Id=int(row['ID'])
                except: name, Id = "Unknown", 0
            cv2.putText(im, f"{name}", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            if Id != 0 and Id not in tested_ids:
                date = datetime.datetime.now().strftime("%d-%m-%Y")
                timeStamp = datetime.datetime.now().strftime("%H:%M:%S")
                sensor_data = fetch_sensors_from_esp32()
                alcohol = safe_float(sensor_data.get("alcohol_norm",0.0))
                smoke   = safe_float(sensor_data.get("smoke_norm",0.0))
                status = []
                if alcohol > ALCOHOL_THRESHOLD: status.append("Alcohol")
                if smoke > SMOKE_THRESHOLD: status.append("Smoke")
                if Id in prev_head_pos:
                    prev_x, prev_y = prev_head_pos[Id]
                    dx, dy = abs(prev_x - x), abs(prev_y - y)
                    if dx>STAGGER_HEAD_THRESHOLD or dy>STAGGER_HEAD_THRESHOLD and "Staggering" not in status: status.append("Staggering")
                prev_head_pos[Id]=(x,y)
                roi=im[y:y+h,x:x+w]
                if roi.size!=0:
                    avg_color = np.mean(roi,axis=(0,1))
                    if avg_color[2]>120 and "Flushed Face" not in status: status.append("Flushed Face")
                    if avg_color[2]>avg_color[1]+40 and "Red Eyes" not in status: status.append("Red Eyes")
                if not status: status=["OK"]
                rec_status="; ".join(status)
                reaction_ms=run_reaction_test(timeout_ms=3000)
                reaction_display=reaction_ms if reaction_ms is not None else "Timeout"
                attendance.append([Id,name,date,timeStamp,rec_status,reaction_display])
                record={
                    "id":Id,"name":name,"date":date,"time":timeStamp,
                    "status":rec_status,"sensors":sensor_data,"reaction_ms":reaction_ms,
                    "ts":datetime.datetime.utcnow()
                }
                if mongo_coll is not None:
                    try: mongo_coll.insert_one(record)
                    except Exception as e: print("Mongo insert error:",e)
                tv.insert('',0,text=Id,values=(name,date,timeStamp,f"{rec_status} | RT: {reaction_display} ms"))
                tested_ids.add(Id)
                if expected_ids and tested_ids >= expected_ids: break
        cv2.imshow("Attendance + Health",im)
        if cv2.waitKey(1)==ord('q') or (expected_ids and tested_ids >= expected_ids): break

    cam.release(); cv2.destroyAllWindows()
    assure_path_exists("Attendance/")
    file=f"Attendance/Attendance_{datetime.datetime.now().strftime('%d-%m-%Y')}.csv"
    pd.DataFrame(attendance,columns=['ID','NAME','DATE','TIME','STATUS','REACTION_MS']).to_csv(
        file,mode='a',header=not os.path.isfile(file),index=False
    )
    message1.config(text=f"Attendance saved: {len(attendance)} entries")

# ----- GUI -----
window=tk.Tk(); window.geometry("1280x720"); window.title("Attendance + IoT Sensors"); window.configure(bg="#262523")
frame1=tk.Frame(window,bg="#00aeff"); frame1.place(relx=0.11,rely=0.17,relwidth=0.39,relheight=0.80)
frame2=tk.Frame(window,bg="#00aeff"); frame2.place(relx=0.51,rely=0.17,relwidth=0.38,relheight=0.80)
message3=tk.Label(window,text="Face Recognition Attendance + IoT Sensors",fg="white",bg="#262523",width=55,height=1,font=('times',24,'bold'))
message3.place(x=10,y=10)
clock=tk.Label(window,fg="orange",bg="#262523",font=('times',22,'bold')); clock.place(x=1000,y=20)
def tick(): clock.config(text=time.strftime('%H:%M:%S')); clock.after(200,tick)
tick()
tk.Label(frame2,text="Enter ID").place(x=30,y=60); txt=tk.Entry(frame2); txt.place(x=30,y=90)
tk.Label(frame2,text="Enter Name").place(x=30,y=130); txt2=tk.Entry(frame2); txt2.place(x=30,y=160)
message1=tk.Label(frame2,text="1) Take Images >>> 2) Save Profile",bg="#00aeff"); message1.place(x=7,y=230)
tk.Button(frame2,text="Take Images",command=TakeImages).place(x=30,y=300)
tk.Button(frame2,text="Save Profile",command=TrainImages).place(x=30,y=380)
tv=ttk.Treeview(frame1,height=13,columns=('name','date','time','status'))
tv.column('#0',width=80,anchor='center'); tv.column('name',width=130,anchor='w')
tv.column('date',width=120,anchor='center'); tv.column('time',width=100,anchor='center'); tv.column('status',width=320,anchor='w')
tv.heading('#0',text='ID'); tv.heading('name',text='NAME'); tv.heading('date',text='DATE')
tv.heading('time',text='TIME'); tv.heading('status',text='STATUS | RT'); tv.grid(row=0,column=0,padx=10,pady=10,sticky='nsew')
scroll_x=tk.Scrollbar(frame1,orient='horizontal',command=tv.xview); tv.configure(xscrollcommand=scroll_x.set); scroll_x.grid(row=1,column=0,sticky='ew')
tk.Button(frame1,text="Take Attendance",command=TrackImages).grid(row=2,column=0,pady=5)
window.mainloop()
