# https://stackoverflow.com/a/33195411

import os
import cv2
import math
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import multiprocessing as mp
from torchvision import utils
from PIL import Image
from tqdm import tqdm

num_processes = 8
output = mp.Queue()

totensor = transforms.Compose([transforms.ToTensor()])
resz_16 = transforms.Compose([transforms.Resize((16, 16))])
preprocess = transforms.Compose([
			transforms.CenterCrop((128, 128)),
			transforms.Resize((64, 64))
			])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
accept_ext = ('.jpg', '.jpeg', '.png')

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_frontal_path = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
haar_profile_path = os.path.join(cv2_base_dir, 'data/haarcascade_profileface.xml')

frontal_cascade = cv2.CascadeClassifier(haar_frontal_path)
profile_cascade = cv2.CascadeClassifier(haar_profile_path)

def read(x, output):

	# ./dataset/img_align_celeba
	img_dir = os.path.abspath(os.path.expanduser(args.img_dir))
	
	data_dir = os.path.join(os.getcwd(), 'data')
	lr_dir = os.path.join(data_dir, 'LR')
	hr_dir = os.path.join(data_dir, 'HR')

	if not os.path.exists(lr_dir): os.makedirs(lr_dir)
	if not os.path.exists(hr_dir): os.makedirs(hr_dir)

	df_identity = pd.read_csv(os.path.join(os.path.dirname(img_dir), 'identity_CelebA.csv'))

	# if args.pid != 0:
	# 	filt = (df_identity['identity'] == args.pid)
	# 	df = df_identity.loc[filt]
	# else:
	# 	df = df_identity

	# # replace path
	# df_full = pd.DataFrame({
	# 				'person': pd.Series([], dtype='str'),
	# 				'identity': pd.Series([], dtype='int') 
	# 				})

	# for index, row in df.iterrows():
	# 	row['person'] = os.path.join(img_dir, row['person'])
	# 	df_full = df_full.append(row)

	# path_list = df_full['person'].to_list()
	path_list = df_identity['person'].to_list()
	output.put(x, rewrite(path_list, hr_dir, lr_dir))


def rewrite(path_list, hr_dir, lr_dir):

	# for every image
	for img_name in tqdm(path_list):
		if img_name.endswith(accept_ext):
			img_name = os.path.join(args.img_dir, img_name)

			end_name, ext = os.path.splitext(os.path.basename(img_name))

			img = cv2.imread(img_name)
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			frontal = frontal_cascade.detectMultiScale(img_gray, 1.3, 5)
			profile = profile_cascade.detectMultiScale(img_gray)

			if frontal is not None:
				for (x,y,w,h) in frontal:
					l = w if w>h else h
					#roi_img = img[y:y+h, x:x+w]
					roi_img = img[y:y+l, x:x+l]
					new_shape = (64, 64)
					#img_new = center_crop(img, new_shape)
					img_new = cv2.resize(roi_img, new_shape)

					# HR image
					cv2.imwrite(os.path.join(hr_dir, f'{end_name}.png'), img_new)

					# LR image
					img_lr = cv2.resize(img_new, (16, 16))
					cv2.imwrite(os.path.join(lr_dir, f'{end_name}.png'), img_lr)
		

			# # open image and get name and extension
			# img = Image.open(img_name).convert('RGB')
			# img = preprocess(img)
			# end_name, ext = os.path.splitext(os.path.basename(img_name))

			# # save HR img
			# img_hr = totensor(img)
			# img_hr = img_hr.unsqueeze(0).to(device)

			# utils.save_image(img_hr, os.path.join(hr_dir, f'{end_name}{ext}'))

			# # save LR img
			# img_lr = resz_16(img)
			# img_lr = totensor(img_lr)
			# img_lr = img_lr.unsqueeze(0).to(device)

			# utils.save_image(img_lr, os.path.join(lr_dir, f'{end_name}{ext}'))


def center_crop(img, new_shape):
	assert type(new_shape)==tuple, 'shape must be of type tuple'

	rows, cols, _ = map(int, img.shape)
	old_shape = (rows, cols)

	p1 = (0, 0)
	p2 = (0, old_shape[1])
	p3 = (old_shape[0], old_shape[1])
	p4 = (old_shape[0], 0)

	vertexes = (p1, p2, p3, p4)

	x_list = [vertex [0] for vertex in vertexes]
	y_list = [vertex [1] for vertex in vertexes]
	n_vert = len(vertexes)
	
	x = sum(x_list) / n_vert
	y = sum(y_list) / n_vert

	cent = (x, y)

	# upper left point cx = w + dw/2, cy = h + dh/2
	ulp_h = math.floor(cent[0] - (new_shape[0]/2))
	ulp_w = math.floor(cent[1] - (new_shape[1]/2))

	img = img[ulp_h:ulp_h+new_shape[0], ulp_w:ulp_w+new_shape[1]]
	
	return img



if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_dir', type=str, help='dir containing images')
	parser.add_argument('--pid', type=int, default=0, help='person id')
	args = parser.parse_args()

	processes = [mp.Process(target=read, args=(x, output)) for x in range(num_processes)]
	
	for p in processes:
		p.start()

	result = []
	for ii in range(num_processes):
		result.append(output.get(True))

	for p in processes:
		p.join()
