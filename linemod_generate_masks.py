import os
from argparse import ArgumentParser

from DPOD import datasets
from DPOD.datasets.linemod.mask_generation import generate_masks

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--linemod_dir_path', default=datasets.PATHS['linemod'],
                           help='path to directory containing linemod dataset')
    argparser.add_argument('--models_dir_path', default='models_small', help='path to directory containing 3D models')
    argparser.add_argument('--target_dir_path', default=os.path.join(datasets.PATHS['linemod'], "masks"),
                           help='path to directory to save generated masks to')
    argparser.add_argument('--show', action='store_true', help='show generated images on the go')
    argparser.add_argument('--debug', '-d', action='store_true', help='process only 20 images')
    argparser.add_argument('--save', action='store_true', help='save humanreadable masks')
    argparser.add_argument('--force', action='store_true', help='Reproduce masks anyway')

    args = argparser.parse_args()
    print(args)

    generate_masks(args.linemod_dir_path, args.models_dir_path, args.target_dir_path, debug=args.debug, show=args.show,
                   save=args.save, force=args.force)
