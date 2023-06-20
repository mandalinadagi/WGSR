import glob, imageio, torch, logging, pyiqa
import numpy as np
import torch.nn as nn
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

folder_GT  = sys.argv[1]
folder_Gen = sys.argv[2]
device = torch.device("cuda:0")
data_name = folder_Gen.split('/')[-1]
dataset_name = folder_Gen.split('/')[-2]
print(dataset_name)
print(data_name)
logging.basicConfig(filename= data_name + '_' + dataset_name + '_evaluation_benchmarks.log', level=logging.INFO)

pyiqa_lpips     = pyiqa.create_metric('lpips', device=torch.device('cpu'))
pyiqa_lpips_vgg = pyiqa.create_metric('lpips-vgg', device=torch.device('cpu'))

pyiqa_niqe  = pyiqa.create_metric('niqe', device=torch.device('cpu'))
pyiqa_dists = pyiqa.create_metric('dists', device=torch.device('cpu'))
pyiqa_nrqm  = pyiqa.create_metric('nrqm', device=torch.device('cpu'))
pyiqa_pi    = pyiqa.create_metric('pi', device=torch.device('cpu'))
pyiqa_pieapp = pyiqa.create_metric('pieapp', device=torch.device('cpu'))
pyiqa_brisque = pyiqa.create_metric('brisque', device=torch.device('cpu'))

def get_benchmark_results(sr_data, hr_data):
    lpips     = 0
    lpips_vgg = 0
    niqe      = 0
    nrqm      = 0
    pi        = 0
    pieapp    = 0
    dists     = 0
    brisque   = 0
    for i, _ in enumerate(tqdm(sr_data)):
        logging.info(str(sr_data[i]))
        score_lpips = pyiqa_lpips(sr_data[i], hr_data[i])
        lpips       += score_lpips.item()
        logging.info('Lpips: ' + str(score_lpips.item()))
      
        score_lpips_vgg = pyiqa_lpips_vgg(sr_data[i], hr_data[i])
        lpips_vgg       += score_lpips_vgg.item()
        logging.info('Lpips-VGG: ' + str(score_lpips_vgg.item()))

        score_pieapp = pyiqa_pieapp(sr_data[i], hr_data[i])
        pieapp    += score_pieapp.item()
        logging.info('Pieapp: ' + str(score_pieapp.item()))

        score_dists = pyiqa_dists(sr_data[i], hr_data[i])
        dists       += score_dists.item()
        logging.info('DISTS: ' + str(score_dists.item()))

        score_niqe = pyiqa_niqe(sr_data[i])
        niqe       += score_niqe.item()
        logging.info('NIQE: ' + str(score_niqe.item()))

        score_nrqm = pyiqa_nrqm(sr_data[i])
        nrqm      += score_nrqm.item()
        logging.info('NRQM: ' + str(score_nrqm.item()))

        score_pi = pyiqa_pi(sr_data[i])
        pi       += score_pi.item()
        logging.info('PI: ' + str(score_pi.item()))

        score_brisque = pyiqa_brisque(sr_data[i])
        brisque     += score_brisque.item()
        logging.info('BRISQUE: ' + str(score_brisque.item()))


    lpips     /= len(sr_data)
    lpips_vgg /= len(sr_data)
    pieapp    /= len(sr_data)
    niqe      /= len(sr_data)
    nrqm      /= len(sr_data)
    pi        /= len(sr_data)
    dists     /= len(sr_data)
    brisque   /= len(sr_data)

    logging.info("****************** OVERALL RESULTS ******************")
    logging.info('Lpips: ' + str(lpips))
    logging.info('Lpips-VGG: ' + str(lpips_vgg))
    logging.info('Pieapp: ' + str(pieapp))
    logging.info('DISTS: '+ str(dists))
    logging.info('NIQE: ' + str(niqe))
    logging.info('NRQM: ' + str(nrqm))
    logging.info('PI: ' + str(pi))
    logging.info('BRISQUE: ' + str(brisque))

    return lpips, lpips_vgg, pieapp, dists, niqe, nrqm, pi, brisque


sr_data        = natsorted(sorted(glob.glob(folder_Gen + "/*.png"), key=len))
hr_data        = natsorted(sorted(glob.glob(folder_GT + "/*.png"), key=len))
logging.info("********************************************************")
logging.info("********************************************************")
lpips, lpips_vgg, pieapp, dists, niqe, nrqm, pi, brisque = get_benchmark_results(sr_data, hr_data)
