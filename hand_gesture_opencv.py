import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import load_model
import cv2

newmod=load_model('hand_gestures_model.h5')
background = None
gesture = ['Fist', 'Five', 'None', 'Okay', 'Peace', 'Rad', 'Straight', 'Thumbs']

accumulated_weight = 0.5
#region of interest (ROI) coordinates
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def calc_accum_avg(frame, accumulated_weight):
    global background
    
    #backgroung in first frame is None so we assign the first frame to background
    if background is None:
        background = frame.copy().astype("float")
        return None

    #background = accumulated_weight*background + (1-accumulated_weight)*frame
    cv2.accumulateWeighted(frame, background, accumulated_weight)

#segment function to segment the hand region in the image
def segment(frame, threshold=20):
    global background
    
    #soustract the background from the current frame to get the hand region(pixels that are different from the background)
    diff = cv2.absdiff(background.astype("uint8"), frame)

    #binary threshold the diff image so that we get the foreground(hand region) as white and the rest as black (20 is the threshold value)
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    #find the contours in the thresholded image to segment the hand region(list of white pixels)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #if length is 0 we didnt find any countour(hand region)
    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)
 
#prediction function to predict the hand gesture index using the trained model
def thres_display(img):
    width=64
    height=64
    dim=(width,height)
    #resize the image to 64x64 pixels as the model was trained on 64x64 pixel images
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #convert the image to array and expand the dimensions to match the input shape of the model (1(batch_size), 64, 64, 1)
    test_img=image.img_to_array(resized)
    test_img=np.expand_dims(test_img,axis=0)
    #predict the index of the hand gesture using the model and return it as a list
    result= newmod.predict(test_img)
    val=[index for index,value in enumerate(result[0]) if value ==1]
    return val
    
cam = cv2.VideoCapture(0)

num_frames = 0

# loop until esc key is pressed
while True:
    #get the current frame 
    ret, frame = cam.read()
    #mirror the frame
    frame = cv2.flip(frame, 1)
    frame_copy = frame

    #get the ROI
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    #grayscale and blur to roi (delete the noise and smooth the image)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    #for the first 60 frames
    if num_frames < 60:
        #calculate the accumulated weight (background(without hand) = accumulated_weight*background + (1-accumulated_weight)*frame)
        calc_accum_avg(gray, accumulated_weight) #to have a stable background
        if num_frames <= 59:
            cv2.putText(frame_copy, "HI WAIT! WERE GETTING THE BACKGROUND AVERAGE", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
    else:
        # we display the instructions to the user on the screen
        cv2.putText(frame_copy, "Place your hand in side the box", (330, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "index 0 : Fist", (330, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "index 1 : Five", (330, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "index 2 : None", (330, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "index 3 : Okay", (330, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "index 4 : Peace", (330, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "index 5 : Rad", (330, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "index 6 : Straight", (330, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        cv2.putText(frame_copy, "index 7 : Thumbs", (330, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)

        #segment the hand region(contour of the hand) and display the thresholded image(binary image)
        hand = segment(gray)

        #check whether hand region is segmented or not
        if hand is not None:
            thresholded, hand_segment = hand
            
            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

            # display the thresholded image
            cv2.imshow("Thresholded Image", thresholded)
            #get the predicted index of the hand gesture and display it on the screen
            res=thres_display(thresholded)
            
            if len(res)==0:
                cv2.putText(frame_copy, str('None'), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                #get the index of the predicted hand gesture
                x='index'+str(res[0])+':'+gesture[res[0]]
                cv2.putText(frame_copy, str(x), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
    #draw the roi rectangle on the screen
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 2)

    #increment the number of frames
    num_frames += 1

    #display the frame with segmented hand and predicted index
    cv2.imshow("Hand Gestures", frame_copy)

    #close the camera when esc key is pressed
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()