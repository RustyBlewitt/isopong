import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load('perfect.mp3')
pygame.mixer.music.play()
# while pygame.mixer.music.get_busy() == True:
# 	continue

time.sleep(4)
