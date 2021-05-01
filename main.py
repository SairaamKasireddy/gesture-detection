import math
import time
import pyautogui
import numpy as np
import cv2
import sys
# from demo import *
import webbrowser
webbrowser.open("C:/Users/kvnsa/PycharmProjects/detector/hello.html")
print('it did come here')

def gesture():
    # creating an object to capture video from device camera
    videoCapture = cv2.VideoCapture(0)
    screen_width = pyautogui.size().width
    screen_height = pyautogui.size().height
    ret, frame = videoCapture.read()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    half_frame_width = (int)(frame_width/2) +150
    half_frame_height = (int)(frame_height/2) +100
    prev_centroid = (-1,-1)
    print(screen_height,screen_width,frame_height,frame_width,half_frame_height,half_frame_width)

    # running a loop to continuously capture video and detect until the user wishes to quit by pressing 'q' key
    while True:
        # using a try catch block to prevent from exiting when no gesture is found
        try:
            # reading a frame from video input to work on
            ret, frame = videoCapture.read()
            # flipping the image to show mirrored view to the user to make it easier to follow through
            frame = cv2.flip(frame, 1)

            # defining the range for color to be considered as skin in HSV format
            # lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            # upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            lower_skin = np.array([0, 10, 60], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)

            # defining the area being concentrated and expected area for gesture to appear in
            # will be extended to whole screen in future development
            concentrated_region = frame[200:frame_height, 200:frame_width]
            concentrated_region = cv2.resize(concentrated_region, (half_frame_width, half_frame_height))
            # drawing a rectangle around this region to make it easier for the user to visualize the boundary within
            # which he should place the gesture
            cv2.rectangle(frame, (200, 200), (frame_width, frame_height), (0, 255, 0), 0)
            # converting the concentrated region from RGB to HSV format
            region_in_hsv = cv2.cvtColor(concentrated_region, cv2.COLOR_BGR2HSV)

            # extract the portion of concentrated input in defined skin range
            masked_input = cv2.inRange(region_in_hsv, lower_skin, upper_skin)
            # defining a kernel to blur the input image
            kernel = np.ones((3, 3), np.uint8)
            # going through the masked_input and extrapolating the image and blurring the image
            # to cover any missing spots in the process this help us to get a better complete portion of hand
            masked_input = cv2.dilate(masked_input, kernel, iterations=4)
            masked_input = cv2.GaussianBlur(masked_input, (5, 5), 100)

            # finding contours or boundaries to the shape of masked gesture
            contours, hierarchy = cv2.findContours(masked_input, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # We assume the contour with maximum area is the gesture
            contour = max(contours, key=lambda x: cv2.contourArea(x))

            # calculate moments for each contour
            M = cv2.moments(contour)
            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(concentrated_region, (cX, cY), 5, (255, 0, 255), -1)
            cv2.putText(frame, "centroid", (250 + cX - 25, 100 + cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # approximating the contour
            epsilon = 0.0005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # make convex hull or outline around gesture area
            hull = cv2.convexHull(contour)
            # define area of hull and area of hand
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)

            # find the ratio of area not covered by hand in convex hull
            arearatio = ((hull_area - contour_area) / contour_area) * 100

            # finding defects in convex hull with respect to hand
            # here defects are like the spaces between fingers
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)

    #################################################
            # extLeft1 = tuple(contour[contour[:, :, 0].argmin()][0])
            # extRight1 = tuple(contour[contour[:, :, 0].argmax()][0])
            # extTop1 = tuple(contour[contour[:, :, 1].argmin()][0])
            # extBot1 = tuple(contour[contour[:, :, 1].argmax()][0])

            # extLeft = (extLeft1[0] + 250, extLeft1[1] + 100)
            # extRight = (extRight1[0] + 250, extRight1[1] + 100)
            # extTop = (extTop1[0] + 250, extTop1[1] + 100)
            # extBot = (extBot1[0] + 250, extBot1[1] + 100)

            # cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
            # cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
            # cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
            # cv2.circle(frame, extBot, 8, (255, 255, 0), -1)


            # initializing a variable to count the number of defects
            number_of_defects = 0

            # finding number of defects (counting fingers)
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100, 180)

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                # distance between point and convex hull
                d = (2 * ar) / a

                # apply cosine rule here
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # ignore angles > 90 and points close to convex hull because they are generally caused by noise from
                # external entities or other body parts
                if angle <= 90 and d > 30:
                    number_of_defects += 1
                    cv2.circle(concentrated_region, far, 3, [255, 0, 0], -1)

                # drawing lines around the hand region
                cv2.line(concentrated_region, start, end, [0, 255, 0], 2)

            # adding 1 as generally number of fingers = defects + 1
            number_of_defects += 1
            print('no of defects', number_of_defects)
            # identifying the corresponding gesture from number of defects and area ratio
            font = cv2.FONT_HERSHEY_SIMPLEX
            if number_of_defects == 1:
                if contour_area < 2000:
                    cv2.putText(concentrated_region, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    if arearatio < 12:
                        # cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(concentrated_region, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        print('gesture 0')
                    # elif arearatio < 17.5:
                    #     cv2.putText(concentrated_region, 'Good Job', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    #     print('gesture goodjob')
                    else:
                        # cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(concentrated_region, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                        extTop1 = tuple(contour[contour[:, :, 1].argmin()][0])
                        # extTop = (extTop1[0] + screen_width - half_frame_width, extTop1[1] + screen_height - half_frame_height)
                        pyautogui.moveTo(extTop1[0] * screen_width / half_frame_width, extTop1[1] * screen_height / half_frame_height)
                        # pyautogui.click()
                        print('gesture 1')
            elif number_of_defects == 2:
                cv2.putText(concentrated_region, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                pyautogui.click()
                # pyautogui.press('volumedown')
                # pyautogui.press('tab')
                print('gesture 2')
            elif number_of_defects == 3:
                print('gesture 3')
                if arearatio < 27:
                    cv2.putText(concentrated_region, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(concentrated_region, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            elif number_of_defects == 4:
                print('gesture 4')
                cv2.putText(concentrated_region, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                # pyautogui.press('volumeup')
                pyautogui.keyDown('shift')
                pyautogui.press('tab')
                pyautogui.keyUp('shift')

            elif number_of_defects == 5:
                print('gesture 5')
                pyautogui.press('tab')
                cv2.putText(concentrated_region, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                if cX - prev_centroid[0] > 250:
                    cv2.putText(concentrated_region, 'right', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    print('right')
                elif cY - prev_centroid[1] > 250:
                    cv2.putText(concentrated_region, 'up', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    print('up')
                elif cX - prev_centroid[0] < -250:
                    cv2.putText(concentrated_region, 'left', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    print('left')
                elif cY - prev_centroid[1] < -250:
                    cv2.putText(concentrated_region, 'down', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    print('down')
                else:
                    cv2.putText(concentrated_region, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                # pyautogui.moveTo(cX * screen_width / half_frame_width, cY * screen_height / half_frame_height)
                prev_centroid = (cX, cY)
                # cal_total()

            elif number_of_defects == 6:
                print('gesture 6')
                cv2.putText(concentrated_region, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            else:
                print('gesture nothing')
                cv2.putText(concentrated_region, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            # showing windows with camera input and masked concentrated region
            cv2.imshow('masked_input', masked_input)
            # cv2.imshow('frame', frame)
            cv2.imshow('Area', concentrated_region)
            cv2.setWindowProperty('Area', cv2.WND_PROP_TOPMOST, 1)
            cv2.moveWindow('Area', screen_width-half_frame_width-100, screen_height-half_frame_height-125)
            cv2.namedWindow('Area', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Area', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # window()
        except :
            pass
            # print('error')

        # checking if input key 'q' is pressed by user to exit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5)

    # closing all windows and releasing camera resources
    cv2.destroyAllWindows()
    videoCapture.release()

gesture()
