import sys
from pyvino.model.model import Model
from pyvino.model.object_recognition.emotion_recognizer import EmotionRecognizer
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
    # new_frame = model.compute(frame)
    
    
    model = EmotionRecognizer()
    new_frame = model.compute(frame)

    new_frame = new_frame['frame']
    new_frame = np.asarray(cv2pil(new_frame))
    new_frame

    cv2.imwrite('test_emotion.jpg', new_frame)
    print('Save to test_emotion.jpg')

    #new_frame = new_frame['frame']

    #new_frame = np.asarray(cv2pil(new_frame))
    #imshow(new_frame)

if __name__ == '__main__':
    sys.exit(main() or 0)