from fit import *
from vit import *
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--save_dir',default='./output/')
parser.add_argument('--image_size',default=224,type=int)
parser.add_argument('--patch_size',type=int, default=16)
parser.add_argument('--device',default='cpu',type=str)
parser.add_argument('--num_classes',type=int, default=10)
parser.add_argument('--d_model',type=int, default=512)
parser.add_argument('--dff',type=int, default=256)
parser.add_argument('--num_heads',type=int, default=256)


args = parser.parse_args()


vit_mode = VIT(num_classes=args.num_classes, d_model=args.d_model, num_heads=args.num_heads, dff=args.dff, 
               num_layers=args.num_layers, dropout=args.dropout, image_size=args.image_size,
               patch_size=args.patch_size)


act = fit(args=args, net=vit_mode)
act.train()
act.test()