# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import pygame
# import os

cool_off_override = True

# Initialize the pygame mixer for sounds
pygame.mixer.init()

# Load sound files
perfect_sound = pygame.mixer.Sound('sounds/perfect.wav')
nice_sound = pygame.mixer.Sound('sounds/nice.wav')
useless_sound = pygame.mixer.Sound('sounds/useless.wav')

score = 0
frame_count = 0

# Measure euclid dist
def get_dist(bullseye, captured):
	return ( (bullseye[0] - captured[0])**2 + (bullseye[1] - captured[1])**2 )**0.5

# The lower the score the better
def audio_feedback(score):
	if score < 50:
		perfect_sound.play()
	elif score < 150:
		nice_sound.play()
	else:
		useless_sound.play()

# The lower the score the better
def get_text_score(score):
	if score < 50:
		return "Perfect!"
	elif score < 150:
		return "Nice"
	else:
		return "Useless"

def get_text_clr(score):
	if score < 50:
		return (0, 255, 0)
	elif score < 150:
		return (255, 255, 255)
	else:
		return (0, 0, 255)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
ap.add_argument("-r", "--record", type=int, default=False,
    help="record or not")
args = vars(ap.parse_args())


# if(args.get# Save the captured images
# dirname = int(time.time())
# os.mkdir(dirname)

orangeLower = (0, 88, 91)

# Hardcoded bullseye val
bullseye = (300, 180)

# HSV color space, bound what we consider "orange"
orangeUpper = (56, 255, 255)
pts = deque(maxlen=args["buffer"])
rds = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)

uniques = 0

# Ball should initially be going away
was_returning = False


# Constant
max_diminish = 10

diminishing = max_diminish

# keep looping
while True:
	# grab the current frame
	frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "orange", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, orangeLower, orangeUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# Draw bullseye
	cv2.circle(frame, bullseye, 10, (100, 0, 100), 2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	center = None
	radius = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), rad) = cv2.minEnclosingCircle(c)
		radius = rad
		# print("At ({},{}), radius: {}".format(x,y,radius))
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		# if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points

		# Draw circles. Center, radius and bullseye
		cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
		cv2.circle(frame, center, 5, (0, 0, 255), -1)
		cv2.circle(frame, center, 10, (0, 0, 255), -1)

		# Update the radii queue only if unique
		if len(rds) == 0 or radius != rds[0]:

			if (len(rds) > 2):
				print("Prev pos: ", pts[1])

			uniques += 1
			diminishing -= 1

			rds.appendleft(radius)
			pts.appendleft(center)

			# Cool off period for capturing of startup points
			if (cool_off_override and len(rds) > 2) or len(rds) > 5:
				now_returning = rds[0] > rds[1]
				direction_change = was_returning != now_returning
				# print("Check diminishing ", diminishing)

				# Wall hit
				if direction_change and diminishing < 0:
					# Take currnet point as the wall hit loc
					score = get_dist(pts[1], bullseye)
					audio_feedback(score)
					print("Wall hit @ ", pts[1])
					print("Score: ", get_dist(pts[1], bullseye))
					was_returning = True
					diminishing = max_diminish

	# Else nothing detected
	else:
		diminishing = max_diminish
		was_returning = False
		# Revert deques
		pts = deque(maxlen=args["buffer"])
		rds = deque(maxlen=args["buffer"])

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), thickness)

		text = get_text_score(score)
		font = cv2.FONT_HERSHEY_SIMPLEX
		color = get_text_clr(score)
		text_pos = (200, 250)

		# Put text score on frame (size 2, thickness 5)
		cv2.putText(frame, text, text_pos, font, 2, color, 5)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()
