import cv2, time, pandas
from datetime import datetime

first_frame = None
status_list = [None, None]
times = []
df = pandas.DataFrame(columns = ["Start", "End"])
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read() #Capturing the first frame
    status = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converts the frame to gray 
    gray = cv2.GaussianBlur(gray,(21,21),0) #Blurs the image and applies a GuassianBlur 

    if first_frame is None: #Store the first frame, in the variable provided (first_frame)
        first_frame = gray
        continue 

    delta_frame = cv2.absdiff(first_frame, gray) #Calculating the difference between the first frame and the current frame
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000: #If the contour finds movement less than 10000 it changes the status to 1
            continue 
        status = 1 #The status updates to show whether there is an image on screen

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3) #Creates a green box around the image 
    status_list.append(status) #If object enters the frame it will be 1, if there is nothing it will be 0

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())
    
    #On the execution these lines will output four different variations of the video 
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Colour Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'): #A break key to end the application
        if status == 1:
            times.append(datetime.now())
        break

    print(status_list)
    print(times)

    for i in range(0, len(times), 2):
        df = df.append({"Start": times[i], "End":times[i+1]}, ignore_index = True)

df.to_csv("Times.csv")
    
video.release()
cv2.destroyAllWindows