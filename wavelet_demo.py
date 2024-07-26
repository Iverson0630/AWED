from pytorch_wavelets import DWTForward, DWTInverse
import cv2
import torch
from torchvision import  transforms
dct = DWTForward(J=1, mode='zero', wave='haar')
idct = DWTInverse(mode='zero', wave='haar')
img = cv2.imread('../../data/data1k/train/full_224/20.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png',gray_img)

vimg = transforms.ToTensor()(gray_img).unsqueeze(0)

l1, yh = dct(vimg)

yh = yh[0]
lh = yh[0][0][0]
hl = yh[0][0][1]
hh = yh[0][0][2]

l1_img = transforms.ToPILImage()(l1[0])
lh_img = transforms.ToPILImage()(lh)
hl_img = transforms.ToPILImage()(hl)
hh_img = transforms.ToPILImage()(hh)

l1_img.save('ll.png')
hl_img.save('hl.png')
lh_img.save('lh.png')
hh_img.save('hh.png')