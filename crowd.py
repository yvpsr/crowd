import cv2
import winsound

pedestrian_cascade = cv2.CascadeClassifier('fullbody.xml')
fullbody_cascade = cv2.CascadeClassifier('upperbody.xml')
video_source = "people.mp4"  
cap = cv2.VideoCapture(video_source)

people_count = 0
group_count = 0
group_threshold = 20  
message = ""
beep_played = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    fullbodies = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    all_people = list(pedestrians) + list(fullbodies)

    frame_group_count = 0

    for (x, y, w, h) in all_people:
        if w * h > 1000:  
            if frame_group_count == 0:
                group_count += 1
            frame_group_count += 1
            color = (0, 0, 255) if frame_group_count > group_threshold else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    people_count = len(all_people)
    if frame_group_count > group_threshold:
        message = "Crowd Formed"
        if not beep_played and cap.get(cv2.CAP_PROP_POS_MSEC) >= 1000:
            winsound.Beep(1000, 500)  
            beep_played = True
    elif frame_group_count < group_threshold:
        message = "The Area is in normal state"
    else:
        message = ""

    cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()