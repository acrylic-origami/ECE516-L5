import sys
import scipy.sparse
import scipy.signal
import warnings
import cv2
import re
import numpy as np
import math

NUM_AVG_FRAMES = 10

if __name__ == '__main__':
	def run():
		D = []
		print sys.argv
		for f in sys.argv[1:]:
			S = float(re.search('(?=.*\\/|^)(\\d+)\\.[^\\.]*$', f).group(1))
			sys.stdout.write('Averaging video 1/%d\r' % S)
			V = cv2.VideoCapture(f)
			
			im_avg = None
			first = True
			dtype = np.dtype('u1')
			frame_idx = 0
			while True:
				frame_idx += 1
				valid, im = V.read()
				if not valid:
					break
				
				if NUM_AVG_FRAMES != None and frame_idx > NUM_AVG_FRAMES:
					break
				
				if first:
					im_avg = np.zeros(im.shape, dtype=np.float64)
					
				dtype = im.dtype
				im_avg += im.astype(np.float64)
				first = False
				
			num_frames = int(V.get(cv2.CAP_PROP_FRAME_COUNT))
			im_avg /= (num_frames if NUM_AVG_FRAMES == None else min(num_frames, NUM_AVG_FRAMES))
			D.append((S, dtype, im_avg))
		
		sys.stdout.write('\n')
		
		# comp = np.array((side * len(D), side * len(D)))
		def make_comparagram():
			for i, (shutter_a, dtype_a, im_a) in enumerate(D):
				for j, (shutter_b, dtype_a, im_b) in enumerate(D[i:]):
					im_a = im_a.astype(dtype_a)
					im_b = im_b.astype(dtype_a)
					try:
						# comp = np.zeros((2 ** (im_b.dtype.itemsize * 8), 2 ** (im_a.dtype.itemsize * 8), im_a.shape[2]), dtype=np.uint64)
						assert im_a.shape == im_b.shape
						sys.stdout.write('1/%d - 1/%d\r' % (shutter_a, shutter_b))
						comps = []
						for color_dim in range(im_a.shape[-1]):
							a = im_a[...,color_dim].astype(np.uint64)
							b = im_b[...,color_dim].astype(np.uint64)
							p = ((a + b + 1) * (a + b)) / 2 + b # apply cantor's pairing function
							[U, counts] = np.unique(p, return_counts=True)
							w = np.floor((np.sqrt(8*U+1)-1)/2).astype(np.uint64)
							bc = (U - (w**2 + w) / 2)
							ac = (w - bc) # unapply cantor's pairing function
							comps.append(scipy.sparse.coo_matrix((np.log(counts), (ac.tolist(), bc.tolist()))).toarray())
							
						comps = np.transpose(np.array(comps), (1, 2, 0))
						
						# unique is much faster than:
						# for y in range(im_a.shape[0]):
						# 	for x in range(im_a.shape[1]):
						# 		for color_dim in range(im_a.shape[2]):
						# 			comp[im_b[y, x, color_dim], im_a[y, x, color_dim], color_dim] += 1
						
						cv2.imwrite('%d_%d.png' % (shutter_a, shutter_b), ((comps * 255) / np.max(comps)).astype(np.uint8)) # x_y
					except AssertionError as e:
						warnings.warn('Size mismatch: Cannot make comparagram between images at shutters 1/%d and 1/%d; skipping' % (shutter_a, shutter_b))
						
		def hdr():
			# find the gamma fits
			trusts = []
			log_k = []
			gamma_log_k = []
			for i, (shutter_a, dtype_a, im_a) in enumerate(D):
				for j, (shutter_b, dtype_a, im_b) in enumerate(D[i+1:]):
					sys.stdout.write('Cross 1/%d - 1/%d\r' % (shutter_a, shutter_b))
					flat_a = im_a.flatten()
					flat_b = im_b.flatten()
					nonclipping = np.logical_and(flat_a <= 254.5, flat_b <= 254.5)
					trusts.append(np.std((flat_a if shutter_a > shutter_b else flat_b)[nonclipping])) # take dynamic range of non-clipping region of darker image
						
					log_k.append(math.log(shutter_a / shutter_b))
					gamma_log_k.append(math.log(np.asscalar(np.linalg.lstsq(flat_a[nonclipping,np.newaxis], flat_b[nonclipping])[0])))
			
			gamma = np.asscalar(np.polyfit(log_k, gamma_log_k, 1, w=trusts)[0]) 
			print('Gamma: %.3f' % gamma)
			
			# compute HDR
			weights = np.array([
				(np.minimum(
					3 * np.ones(im.shape),
					255 - np.array([
						scipy.signal.convolve2d(gray_im, np.ones((15, 15)) / 225, mode='same')\
							for gray_im in np.transpose(im, (2, 0, 1)) # blur image slightly to soften the derating on clipped values
					]).transpose((1, 2, 0))
				) / 3 * 0.99 + 0.01) * (\
					(1 / shutter) *\
					(im + 0.5) ** ((gamma - 1) / gamma)
				) for (shutter, dtype, im) in D
			])
			norm_factor = np.sum(weights, axis=0)
			norm_factor[norm_factor == 0] = 1
			return (np.sum(weights * [d[2] ** (1 / gamma) / (1 / shutter) for d in D], axis=0) / norm_factor) ** gamma
		
		I = hdr()
		cv2.imwrite('hdr.png', (I * 255 / np.max(I)).astype(np.uint8))
		cv2.imshow('1', (I * 255 / np.max(I)).astype(np.uint8))
		cv2.waitKey(0)
	run()