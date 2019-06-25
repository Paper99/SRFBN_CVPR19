# -*- coding: utf-8 -*-
# @Time    : 2019-05-21 19:55
# @Author  : LeeHW
# @File    : Prepare_data.py
# @Software: PyCharm
from glob import glob
from flags import *
import os
from scipy import misc
import numpy as np
import datetime
from multiprocessing.dummy import Pool as ThreadPool

starttime = datetime.datetime.now()

save_HR_path = os.path.join(save_dir, 'HR_x4')
save_LR_path = os.path.join(save_dir, 'LR_x4')
os.mkdir(save_HR_path)
os.mkdir(save_LR_path)
file_list = sorted(glob(os.path.join(train_HR_dir, '*.png')))
HR_size = [100, 0.8, 0.7, 0.6, 0.5]


def save_HR_LR(img, size, path, idx):
	HR_img = misc.imresize(img, size, interp='bicubic')
	HR_img = modcrop(HR_img, 4)
	rot180_img = misc.imrotate(HR_img, 180)
	x4_img = misc.imresize(HR_img, 1 / 4, interp='bicubic')
	x4_rot180_img = misc.imresize(rot180_img, 1 / 4, interp='bicubic')

	img_path = path.split('/')[-1].split('.')[0] + '_rot0_' + 'ds' + str(idx) + '.png'
	rot180img_path = path.split('/')[-1].split('.')[0] + '_rot180_' + 'ds' + str(idx) + '.png'
	x4_img_path = path.split('/')[-1].split('.')[0] + '_rot0_' + 'ds' + str(idx) + '.png'
	x4_rot180img_path = path.split('/')[-1].split('.')[0] + '_rot180_' + 'ds' + str(idx) + '.png'

	misc.imsave(save_HR_path + '/' + img_path, HR_img)
	misc.imsave(save_HR_path + '/' + rot180img_path, rot180_img)
	misc.imsave(save_LR_path + '/' + x4_img_path, x4_img)
	misc.imsave(save_LR_path + '/' + x4_rot180img_path, x4_rot180_img)


def modcrop(image, scale=4):
	if len(image.shape) == 3:
		h, w, _ = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w, :]
	else:
		h, w = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w]
	return image


def main(path):
	print('Processing-----{}/0800'.format(path.split('/')[-1].split('.')[0]))
	img = misc.imread(path)
	idx = 0
	for size in HR_size:
		save_HR_LR(img, size, path, idx)
		idx += 1

items = file_list
pool = ThreadPool()
pool.map(main, items)
pool.close()
pool.join()
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
