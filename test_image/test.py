import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import os

model_path = sys.argv[1] 
test_img_folder_name = sys.argv[2]
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu

model_name = model_path.split('/')[-3]
print(model_name)

test_img_folder = '/kuacc/users/ckorkmaz14/datasets/benchmarks/' + test_img_folder_name + '/LR_bicubic/*'
result_path = './results/'+ model_name + '/' + test_img_folder_name + '/'
os.makedirs(result_path, exist_ok=True)

model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()

for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0][:-3]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(result_path + '{:s}_SR.png'.format(base), output)
