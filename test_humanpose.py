import sys
from pyvino.model.model import Model
from pyvino.model.human_pose_estimation.human_pose_estimator import HumanPoseDetector
from pyvino.util.image import imshow, cv2pil
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-t', '--task',
                      help='task for test.',
                      default='detect_face', type=str)
    return parser


def main():
    args = build_argparser().parse_args()
    test_image = './data/test/person1.jpg'
    im = mpimg.imread(test_image)
    plt.imshow(im)
    frame = cv2.imread(test_image)
    
    model = Model(args.task)
    model = HumanPoseDetector()
    results = model.compute(frame)

    new_frame = results['frame']
    new_frame = np.asarray(cv2pil(new_frame))
    print(results['preds'])

    cv2.imwrite('test_humanpose.jpg', new_frame)
    print('Save to test_humanpose.jpg')

    #new_frame = new_frame['frame']

    #new_frame = np.asarray(cv2pil(new_frame))
    #imshow(new_frame)

if __name__ == '__main__':
    sys.exit(main() or 0)