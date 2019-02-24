import math
import sys
import cv2
import numpy as np
import scipy.signal
# from consts import DELTA_MAX

DELTA_MAX = 160 # max px movement per frame

if __name__ == '__main__':
	def r():
		def fom(h, v):
			# figure of merit (unapoligetically heuristic) for pixel to be our target LED
			return (h.astype(np.uint32) * (v.astype(np.uint32)) ** 2 / 255)
		def peak_h(h, h0):
			return abs(90 - (h + (180 - h0)) % 180)
		def find(f):
			flat = f.flatten()
			flat_argmax = flat.argpartition(-PX_SAMPLES)[-PX_SAMPLES:] # default flatten is row-major
			return [
				np.mean(flat[flat_argmax]),
				np.average([flat_argmax // f.shape[1], flat_argmax % f.shape[1]], axis=1, weights=flat[flat_argmax]).astype(np.uint16)
			] # y, x
			
		for dname in sys.argv[1:]:
			with open('%s/out.csv' % dname, 'w') as out:
				ref_im = cv2.imread('%s/ref.png' % dname, cv2.IMREAD_COLOR)
				out.write('\n')
				V = cv2.VideoCapture('%s/vid.mp4' % dname)
				codec = cv2.VideoWriter_fourcc(*'mp4v')
				Vout = cv2.VideoWriter('%s/out.mp4' % dname, codec, V.get(cv2.CAP_PROP_FPS), (int(V.get(cv2.CAP_PROP_FRAME_WIDTH)), int(V.get(cv2.CAP_PROP_FRAME_HEIGHT))))
				
				
				i = 0
				argmin = None
				# we should be using a formal block matching algorithm
				# we'll just run exhaustive search over small windows near the max
				while True:
					valid, im = V.read()
					if not valid:
						break
					
					im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
					ref_im_hsv = cv2.cvtColor(ref_im, cv2.COLOR_BGR2HSV)
					
					im_hsv_h = peak_h(im_hsv[:,:,0], 0)
					ref_im_hsv_h = peak_h(ref_im_hsv[:,:,0], 0)
					
					if argmin == None:
						# DOWNSAMPLE = 4
						left = 0
						right = im_hsv.shape[1] - ref_im_hsv.shape[1]
						top = 0
						bottom = im_hsv.shape[0] - ref_im_hsv.shape[0]
					else:
						# DOWNSAMPLE = 1
						left = max(argmin[1] - DELTA_MAX, 0)
						right = min(argmin[1] + DELTA_MAX, im_hsv.shape[1])
						top = max(argmin[0] - DELTA_MAX, 0)
						bottom = min(argmin[0] + DELTA_MAX, im_hsv.shape[0])
						
					dists = cv2.matchTemplate(im_hsv_h[top:bottom,left:right], ref_im_hsv_h, cv2.TM_SQDIFF)
					
					# im_f = fom(im_hsv[:,:,0], im_hsv[:,:,2])
					# im_f = (im_f / np.max(im_f)).astype(np.uint8)
					
					argmin_flat = np.argmin(dists)
					argmin_ = ((argmin_flat // dists.shape[1]) + ref_im.shape[1] / 2, (argmin_flat % dists.shape[1]) + ref_im.shape[1] / 2) # relative coordinates
					if argmin == None:
						argmin = argmin_
					else:
						argmin = (argmin_[0] + top, argmin_[1] + left)
					sys.stdout.write('%s|%d\r' % (dname, i))
					# print(argmin)
					out.write('%.1f,%d,%d\n' % ((np.min(dists),)+argmin))
					
					i += 1
					
					im_dists = (dists * 255 / np.max(dists)).astype(np.uint8)
					cv2.imshow('1', im_dists)
					cv2.waitKey(1)

					
					# frame = np.concatenate((im, np.concatenate((imag_f_im, real_f_im), axis=0)), axis=1)
					# print(np.transpose(frame, (1, 0, 2)).shape)
					# Vout.write(im)
					# cv2.imshow('1', frame)
					# cv2.imshow('1', im) # red filter
					# cv2.imshow('1', im_hsv_h[top:bottom,left:right])
					# cv2.waitKey(100)
					
				Vout.release()
				sys.stdout.write('\n')
				sys.stdout.flush()
	r()