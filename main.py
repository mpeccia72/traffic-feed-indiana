import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

# configuring project paths
yolo_model = "latest.pt" 
video_path = "dataset/2.mp4"

# set model
model=YOLO(yolo_model)

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('Traffic Cam')
cv2.setMouseCallback('Traffic Cam', RGB)
cap=cv2.VideoCapture(video_path)

# open and read from file
my_file = open("coco.txt", "r") 
data = my_file.read()
class_list = data.split("\n") # class_list is an array of strings 
#print(class_list)

count = 0

# counters
counter_in = []
counter_out = []
car_counter_in = []
car_counter_out = []
truck_counter_in = []
truck_counter_out = []

# tracker objects for both car and truck
car_tracker = Tracker()
truck_tracker = Tracker()

# dictionary
vh_down = {}
vh_up = {}

# frame dimensions
xDim = 1152
yDim = 648

# top line information
cy1left=538
cy1right=70
cy1Rate = -(cy1left - cy1right)/xDim # for every one unit x, y decreases by cy1Rate (x(0) = cy1left)

# bottom line information
cy2left=648
cy2right=180
cy2Rate = -(cy2left - cy2right)/xDim # for every one unit x, y decreases by cy1Rate (x(0) = cy1left)

# +/- pixels from line
offset = 6

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1152,648))
   

    results=model.predict(frame, conf = .05)

    a=results[0].cpu().boxes.data # needs to be converted to cpu if running cuda toolkit
    px=pd.DataFrame(a).astype("float")

    car_list = []
    truck_list = []
             
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5]) # object
        c=class_list[d]
        if 'Truck' in c:
            truck_list.append([x1,y1,x2,y2])
        if 'Car' in c:
            car_list.append([x1,y1,x2,y2])

    car_bbox_id = car_tracker.update(car_list)
    truck_bbox_id = truck_tracker.update(truck_list)

    # Car Tracking
    for bbox in car_bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        # Cars Going Down
        if (cy1left + cy1Rate * cx) < (cy + offset) and (cy1left + cy1Rate * cx) > (cy - offset):
            vh_down[id] = cy
        if id in vh_down:
            if (cy2left + cy2Rate * cx) < (cy + offset) and (cy2left + cy2Rate * cx) > (cy - offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,255),1)
                if car_counter_in.count(id) == 0:
                    car_counter_in.append(id)

        # Cars Going Up
        if (cy2left + cy2Rate * cx) < (cy + offset) and (cy2left + cy2Rate * cx) > (cy - offset):
            vh_up[id] = cy
        if id in vh_up:
            if (cy1left + cy1Rate * cx) < (cy + offset) and (cy1left + cy1Rate * cx) > (cy - offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,255),1)
                if car_counter_out.count(id) == 0:
                    car_counter_out.append(id)
    
    # Truck Tracking
    for bbox in truck_bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        # Trucks Going Down
        if (cy1left + cy1Rate * cx) < (cy + offset) and (cy1left + cy1Rate * cx) > (cy - offset):
            vh_down[id] = cy
        if id in vh_down:
            if (cy2left + cy2Rate * cx) < (cy + offset) and (cy2left + cy2Rate * cx) > (cy - offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,255),1)
                if truck_counter_in.count(id) == 0:
                    truck_counter_in.append(id)

        # Trucks Going Up
        if (cy2left + cy2Rate * cx) < (cy + offset) and (cy2left + cy2Rate * cx) > (cy - offset):
            vh_up[id] = cy
        if id in vh_up:
            if (cy1left + cy1Rate * cx) < (cy + offset) and (cy1left + cy1Rate * cx) > (cy - offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,255),1)
                if truck_counter_out.count(id) == 0:
                    truck_counter_out.append(id)
                

    # top line       
    cv2.line(frame,(0,cy1left),(1152,cy1right),(81,26,0),4)

    # bottom line
    cv2.line(frame,(0,cy2left),(1152,cy2right),(81,26,255),4) 

    in1 = len(car_counter_in) # car counter in
    cv2.putText(frame,"Cars Going In: " + str(in1),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),4)
    cv2.putText(frame,"Cars Going In: " + str(in1),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)

    out1 = len(car_counter_out) # car counter out
    cv2.putText(frame,"Cars Going Out: " + str(out1),(60,100),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),4)
    cv2.putText(frame,"Cars Going Out: " + str(out1),(60,100),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)

    in2 = len(truck_counter_in) # truck counter in
    cv2.putText(frame,"Trucks Going In: " + str(in2),(60,160),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),4)
    cv2.putText(frame,"Trucks Going In: " + str(in2),(60,160),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    out2 = len(truck_counter_out) # truck counter out
    cv2.putText(frame,"Trucks Going Out: " + str(out2),(60,220),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),4)
    cv2.putText(frame,"Trucks Going Out: " + str(out2),(60,220),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    cv2.imshow("Traffic Cam", frame)

    if cv2.waitKey(0)&0xFF==27:
        break
    
cap.release()
cv2.destroyAllWindows()