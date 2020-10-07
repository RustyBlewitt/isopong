import argparse
import cv2
import imutils
import numpy as np
import pygame
import time
from collections import deque
from imutils.video import VideoStream

# Measure euclid dist
def get_dist(bullseye, captured):
	return ( (bullseye[0] - captured[0])**2 + (bullseye[1] - captured[1])**2 )**0.5

# The lower the accuracy the better
def audio_feedback(accuracy):
	if accuracy < 50:
		perfect_sound.play()
	elif accuracy < 150:
		nice_sound.play()
	else:
		useless_sound.play()

# The lower the accuracy the better
def get_text_accuracy(accuracy):
	if accuracy < 50:
		return "Perfect!"
	elif accuracy < 150:
		return "Nice"
	else:
		return "Useless"

# Determine text color based off accuracy
def get_text_clr(accuracy):
	if accuracy < 50:
		return (0, 255, 0)		# Perfect -> Green
	elif accuracy < 150:
		return (255, 255, 255)	# Nice -> White
	else:
		return (0, 0, 255)		# Useless -> Red

# Initially had a 5 frame period I would let pass per shot before considering
#  if the ball has hit the wall, overridden due to shots in which the ball
#  enters the frame high and hits the wall very soon after entering frame
cool_off_override = True

# Initialize the pygame mixer for sounds
pygame.mixer.init()

# Load sound files for audio feedback
perfect_sound = pygame.mixer.Sound('sounds/perfect.wav')
nice_sound = pygame.mixer.Sound('sounds/nice.wav')
useless_sound = pygame.mixer.Sound('sounds/useless.wav')

# How many unique points we could store per "shot" 
#  (usually need less, but handy if ball left rolling around in view)
cache_len = 50

# The user's accuracy for the recent shot
accuracy = 0

# HSV color range, what I consider "table tennis ball orange" (can change through the day)
#  Future improvements could entail ball color recognition on startup
orangeLower = (0, 88, 91)
orangeUpper = (56, 255, 255)

# Hardcoded bullseye val
bullseye = (300, 180)

# Max points recorded per shot (they will be wiped when ball exits screen)
pts = deque(maxlen=50)
rds = deque(maxlen=50)

# Init video stream
stream = VideoStream(src=0).start()

# Count of unique measurements of (location, size)
uniques = 0

# Ball should initially be going away
was_returning = False

# Constant
max_diminish = 10

# Initialise as the max
diminishing = max_diminish

# Short delay before begin to avoid camera init issues
time.sleep(2.0)

# Main program loop
while True:
	# Grab the current frame
	frame = stream.read()

	# Bail early if reading of frame failed
	if frame is None:
		raise Exception('Could not read frame from video stream')

	# Resize, blur and convert to HSV
	frame = imutils.resize(frame, width = 600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# Construct a mask for ball then erode and dilate to remove small blobs
	mask = cv2.inRange(hsv, orangeLower, orangeUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# Draw bullseye
	cv2.circle(frame, bullseye, 10, (100, 0, 100), 2)

	# Find contours in the mask and initialize the current (x, y) center of the ball
	# RETR_EXTERNAL - gives "outer" contours - when multiple contours, only return outer
	# CHAIN_APPROX_SIMPLE - store only the endpoints of the lines that form the contours
	contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)

	# Init shot metrics to None
	center = None
	radius = None

	# If at least one contour was found
	if len(contours) > 0:

		# Retrieve largest contour found earlier
		max_cont = max(contours, key=cv2.contourArea)

		# Compute minimum enclosing circle of that contour (the ball)
		((x, y), radius) = cv2.minEnclosingCircle(max_cont)

		# Int the x, y floats that were returned from minEnc
		center = (int(x), int(y))

		# Draw circles. Ball outline and center
		cv2.circle(frame, center ), int(radius),(0, 255, 255), 2)
		cv2.circle(frame, center, 5, (0, 0, 255), -1)

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

				# Is the ball currently returning...
				now_returning = rds[0] > rds[1]
				
				# ... and is that the same as the previous point
				direction_change = was_returning != now_returning

				# Wall hit if change detected and enough frames have been captured since last change
				if direction_change and diminishing < 0:
					# Take the current point as the wall hit loc
					accuracy = get_dist(pts[1], bullseye)
					audio_feedback(accuracy)
					# Log the point and its accuracy score
					print("Wall hit @ ", pts[1])
					print("accuracy: ", get_dist(pts[1], bullseye))
					was_returning = True
					diminishing = max_diminish

	# Else - no contours found
	else:
		# Reset diminishing and was_returning
		diminishing = max_diminish
		was_returning = False
		# Revert deques
		pts = deque(maxlen=50)
		rds = deque(maxlen=50)

	# loop over the set of tracked points
	for i in range(1, len(pts)):

		# Ignore if either of the tracked points are None
		if pts[i - 1] is None or pts[i] is None:
			continue

		# Compute thickness and draw connecting lines
		thickness = int(np.sqrt(50 / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (255, 255, 0), thickness)

	# Generate text [perfect, nice, useless], use generic font, 
	#  color text according to accuracy and put in central pos
	text = get_text_accuracy(accuracy)
	font = cv2.FONT_HERSHEY_SIMPLEX
	color = get_text_clr(accuracy)
	text_pos = (200, 250)

	# Put text (described above) on frame with size 2 and thickness 5
	cv2.putText(frame, text, text_pos, font, 2, color, 5)
	# show the frame to our screen
	cv2.imshow("Frame", frame)

	# Quit program via 'q' key
	# 0xff is 11111111 in binary, bitwising '&' lets us capture the last byte of the waitKey
	if ord("q") == cv2.waitKey(1) & 0xFF:
		break

# Stop the video stream
stream.stop()

# close all windows
cv2.destroyAllWindows()
