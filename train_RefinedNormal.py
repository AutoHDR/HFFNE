
import argparse
from torch.utils.data import DataLoader
import os, glob
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import *
from vgg_model import vgg19
from data_loader import *
from tools import CopyFiles

def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def train(
    args,
    train_dl,
    ph_val_dl,
    NormalEncoder,
    NormalDecoder,
    vggnet,
    g_optim,
    device,
):
    train_loader = sample_data(train_dl)
    pbar = range(args.iter + 1)

    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    g_loss_val = 0
    loss_dict = {}
    CosLoss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    PreTrainModel = UNet(channel=1).to(device)
    # ackpt = '/home/xteam/PaperCode/MM_IJCV/MM/P1_result/MM22_stage_1/exp/200000_DDA2PH.pkl'  
    ackpt = '/home/xteam/PaperCode/MM_IJCV/MM/P1_result/MM22_stage_cm_1_D0.0001/exp/200000_DDA2PH.pkl'  
    ckptModel = torch.load(ackpt, map_location=lambda storage, loc: storage)
    PreTrainModel.load_state_dict(ckptModel)
    PreTrainModel.eval()
    NormalEncoder_module = NormalEncoder
    NormalDecoder_module = NormalDecoder

    lossflag = args.lossflag
    savepath = './results/MM_P2/' + args.exp_name
    weight_path = savepath + '/'
    logPath = weight_path + 'log' + '/'
    mkdirss(logPath)
    logPathcodes = weight_path + 'codes' + '/'
    mkdirss(logPathcodes)
    writer = SummaryWriter(logPath)
    imgsPath = weight_path + 'imgs' + '/'
    mkdirss(imgsPath)
    expPath = weight_path + 'exp' + '/'
    mkdirss(expPath)

    #backup codes
    src_dir = './'
    src_file_list = glob.glob(src_dir + '*')                    # glob获得路径下所有文件，可根据需要修改
    for srcfile in src_file_list:
        CopyFiles(srcfile, logPathcodes)                       # 复制文件
    print('copy codes have done!!!')
    
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        face, face_l = next(train_loader)

        b,c,w,h = face.shape
        face = face.to(device)
        face_l = face_l.to(device) 
        
        with torch.no_grad():
            coNorm = F.normalize(PreTrainModel(face_l))
 
        NormalEncoder.train()
        NormalDecoder.train()

        requires_grad(NormalEncoder, True)
        requires_grad(NormalDecoder, True)
        
        normal_feat = NormalEncoder(coNorm)
        fine_normal = NormalDecoder((face_l, normal_feat)) # [-1, 1]
        fine_normal = F.normalize(fine_normal)
        

        recon_F = (F.smooth_l1_loss(fine_normal, coNorm)) * args.refNorm

        ## feature loss
        features_A = vggnet(face, layer_name='all')
        features_B = vggnet((fine_normal/2+0.5), layer_name='all')

        fea_loss1 = F.l1_loss(features_A[0], features_B[0]) / (256*256)

        fea_loss = (fea_loss1) * args.refVGG

        writer.add_scalar('recon_F', recon_F.item(), i)
        writer.add_scalar('fea_loss', fea_loss.item(), i)

        # print('******************************************************')
        # print('******************************************************')
        # print('******************************************************')
        LossTotal = recon_F + fea_loss
        # print('******************************************************')
        # print('******************************************************')
        # print('******************************************************')

        g_optim.zero_grad()
        LossTotal.backward()
        g_optim.step()

        # torch.cuda.empty_cache()
        
        pbar.set_description(
            (
                f"i:{i:6d}; reconF:{recon_F.item():.4f}; fea:{fea_loss.item():.4f};  "
            )
        )

        
        if i % 5000 == 0 and idx!=0:
            torch.save(
                {
                    "NormalEncoder": NormalEncoder_module.state_dict(),
                    "NormalDecoder": NormalDecoder_module.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "args": args,
                },
                f"%s/{str(i).zfill(6)}.pt"%(expPath),
            )
            
        if i>150010:
            break 


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=74143 * 20)#74143*epoch
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ckpt", type=str, default=None)
    # parser.add_argument("--ckpt", type=str, default="/home/xteam/PaperCode/MM_IJCV/MM/results/MM_P2/cm_D0.001/exp/120000.pt")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--exp_name", type=str, default="cm_D0.0001_256256")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--lossflag", type=int, default=0)
    parser.add_argument("--refNorm", type=float, default=1)
    parser.add_argument("--refVGG", type=int, default=1)
    parser.add_argument("--refDN", type=float, default=1)
    parser.add_argument("--gpuID", type=int, default=1)
    parser.add_argument("--refST", type=float, default=0.01)


    args = parser.parse_args()
    device = "cuda:" + str(args.gpuID)

    args.start_iter = 0

    vggnet = vgg19(pretrained_path = '/home/xteam/PaperCode/data_zoo/vgg19-dcbb9e9d.pth', require_grad = False)
    vggnet = vggnet.to(device)
    vggnet.eval()

    NormalEncoder = NormalEncoder(color_dim=512).to(device)
    NormalDecoder = NormalDecoder(bilinear=False).to(device)
    
    # NormalEncoder = NormalEncoderWVGG256(InputCh=512, OutputCh=256).to(device)
    # NormalDecoder = NormalDecoder1_HF(bilinear=False).to(device)

    g_optim = optim.Adam(
        list(NormalEncoder.parameters()) + list(NormalDecoder.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass
        
        NormalEncoder.load_state_dict(ckpt["NormalEncoder"])
        NormalDecoder.load_state_dict(ckpt["NormalDecoder"])
        g_optim.load_state_dict(ckpt["g_optim"])


    datasets = []

    pathd = '/home/xteam/PaperCode/data_zoo/NormalPredict/23_ph_300w_ffhq.csv'
    train_dataset, _ = get_300W_phdb_patch(csvPath=pathd, validation_split=0)
    train_dl  = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=8)

    pathd = '/home/xteam/PaperCode/data_zoo/NormalPredict/23Phdb_test_pre.csv'
    val_dataset, _ = getPhotoDB_23data(csvPath=pathd, validation_split=0)
    ph_val_dl  = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=8)

    train(
        args,
        train_dl,
        ph_val_dl,
        NormalEncoder,
        NormalDecoder,
        vggnet,
        g_optim,
        device,
    )

