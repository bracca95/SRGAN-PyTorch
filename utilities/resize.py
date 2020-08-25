import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms 
from torchvision import utils
from PIL import Image

preprocess = transforms.Compose([transforms.Resize((64, 64))])
totensor = transforms.Compose([transforms.ToTensor()])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def rewrite(img_dir, img_names):

	for img_name in img_names:

		img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
		img = preprocess(img)
		img = totensor(img)
		img = img.unsqueeze(0).to(device)

		utils.save_image(img, os.path.join(img_dir, img_name))
