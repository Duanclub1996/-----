from fit import *
from vit import *
from argparse import ArgumentParser
from mlp_mixer import MLPMixer

parser = ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--save_dir',default='./output/')
parser.add_argument('--image_size',default=32,type=int)
parser.add_argument('--patch_size',type=int, default=8)
parser.add_argument('--device',default='cuda',type=str)
parser.add_argument('--num_classes',type=int, default=10)
parser.add_argument('--d_model',type=int, default=192)
parser.add_argument('--dff',type=int, default=256)
parser.add_argument('--num_heads',type=int, default=8)


args = parser.parse_args()


vit_mode = VIT(num_classes=args.num_classes, d_model=args.d_model, num_heads=args.num_heads, dff=args.dff, 
               num_layers=args.num_layers, dropout=args.dropout, image_size=args.image_size,
               patch_size=args.patch_size)

mlp=MLPMixer(image_size=args.image_size,num_classes=10,patch_size=args.patch_size,depth=args.num_layers,dim=args.d_model)

act = fit(args=args, net=mlp)
act.train()
act.test()