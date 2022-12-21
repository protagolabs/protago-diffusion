#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import cv2
import shutil



from face_align import align_face

# we use brisque score to evaluate the img quality 
# the range from brisque is 
# Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. 
# No-reference image quality assessment in the spa- tial domain. 
# IEEE Transactions on Image Processing, 21(12):4695–4708, 2012. 5

def main(args):
    files = os.listdir(args.src)

    # brisq = BRISQUE(url=False)

    

    for i, x in enumerate(files):

        print(x)

        img = align_face(os.path.join(args.src, x))

        img_np = np.array(img)

        # img_np = np.transpose(img_np, (2,0,1))
        # print(img_np.shape)

        score = cv2.quality.QualityBRISQUE_compute(img_np, "brisque_model_live.yml", "brisque_range_live.yml")[0]


        if score > args.th:
            src = os.path.join(args.src, x)
            dst = os.path.join(args.dst, x)
            shutil.copyfile(src, dst)

        ## 
        print(score)
        


    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')
    parser.add_argument('--src', type=str, default="output")
    parser.add_argument('--dst', type=str, default="filtered_output")
    parser.add_argument('--th', type=float, default=80.0)

    args = parser.parse_args()

    main(args)

