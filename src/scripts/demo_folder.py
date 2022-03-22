"""
File: demo_folder.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: tracking from folder
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from loguru import logger
from train import GoturnTrain

try:
    from goturn.helper import image_io
    from goturn.helper.vis_utils import Visualizer
    from goturn.helper.image_io import resize
    from goturn.helper.BoundingBox import BoundingBox
    from goturn.helper.image_proc import cropPadImage
    from goturn.helper.draw_util import draw
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


refPt = []
image = []
cv2.namedWindow('image')


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        global image
        img_dbg = np.copy(image)
        img_dbg = cv2.rectangle(img_dbg, refPt[0], refPt[1], (0, 255, 0), 2)
        # img_dbg = cv2.cvtColor(img_dbg, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", img_dbg)
        cv2.waitKey(0)


cv2.setMouseCallback("image", click_and_crop)


class loadfromfolder:

    """Helper function to load any video frames without gt"""

    def __init__(self, video_dir):
        """Init folder"""

        self._video_dir = video_dir
        self._videos = {}

    def get_video_frames(self):
        """Get video frames from folder"""

        vid_dir = self._video_dir
        vid_frames = [str(img_path) for img_path in
                      Path(vid_dir).glob('*.jpg')]
        if len(vid_frames) == 0:
            vid_frames = [str(img_path) for img_path in
                          Path(vid_dir).glob('*.png')]
        list_of_frames = sorted(vid_frames)

        self._vid_frames = [list_of_frames]

        return self._vid_frames


class GoturnTracker:

    """Docstring for . """

    def __init__(self, args, dbg=False):
        """load model """
        loader = loadfromfolder(args.input)
        self._vid_frames = loader.get_video_frames()

        # added ground_truth_masks
        loader_mask = loadfromfolder(args.ground_truth)
        self._mask_frames = loader_mask.get_video_frames()

        model_dir = Path(args.model_dir)
        # Checkpoint path
        ckpt_dir = model_dir.joinpath('checkpoints')
        ckpt_path = next(ckpt_dir.glob('*.ckpt'))

        model = GoturnTrain.load_from_checkpoint(ckpt_path)
        model.eval()
        model.freeze()

        self._model = model
        if dbg:
            self._viz = Visualizer()

        self._dbg = dbg

    def vis_images(self, prev, curr, gt_bb, pred_bb, prefix='train'):

        def unnormalize(image, mean, std):
            image = np.transpose(image, (1, 2, 0)) * std + mean
            image = image.astype(np.float32)

            return image

        for i in range(0, prev.shape[0]):
            _mean = np.array([104, 117, 123])
            _std = np.ones_like(_mean)

            prev_img = prev[i].cpu().detach().numpy()
            curr_img = curr[i].cpu().detach().numpy()

            prev_img = unnormalize(prev_img, _mean, _std)
            curr_img = unnormalize(curr_img, _mean, _std)

            gt_bb_i = BoundingBox(*gt_bb[i].cpu().detach().numpy().tolist())
            gt_bb_i.unscale(curr_img)
            curr_img = draw.bbox(curr_img, gt_bb_i, color=(255, 255, 255))

            pred_bb_i = BoundingBox(*pred_bb[i].cpu().detach().numpy().tolist())
            pred_bb_i.unscale(curr_img)
            curr_img = draw.bbox(curr_img, pred_bb_i)

            out = np.concatenate((prev_img[np.newaxis, ...], curr_img[np.newaxis, ...]), axis=0)
            out = np.transpose(out, [0, 3, 1, 2])

            self._viz.plot_images_np(out, title='sample_{}'.format(i),
                                     env='goturn_{}'.format(prefix))

    def _track(self, curr_frame, prev_frame, rect):
        """track current frame
        @curr_frame: current frame
        @prev_frame: prev frame
        @rect: bounding box of previous frame
        """
        prev_bbox = rect

        target_pad, _, _, _ = cropPadImage(prev_bbox, prev_frame)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(prev_bbox, curr_frame)

        if self._dbg:
            self._viz.plot_image_opencv(target_pad, 'target')
            self._viz.plot_image_opencv(cur_search_region, 'current')

        target_pad_in = self.preprocess(target_pad, mean=None).unsqueeze(0)
        cur_search_region_in = self.preprocess(cur_search_region,
                                               mean=None).unsqueeze(0)
        pred_bb = self._model.forward(target_pad_in,
                                      cur_search_region_in)
        if self._dbg:
            prev_bbox.scale(prev_frame)
            x1, y1, x2, y2 = prev_bbox.x1, prev_bbox.y1, prev_bbox.x2, prev_bbox.y2
            prev_bbox = torch.tensor([x1, y1, x2, y2]).unsqueeze(0)
            target_dbg = target_pad_in.clone()
            cur_search_region_dbg = cur_search_region_in.clone()
            self.vis_images(target_dbg,
                            cur_search_region_dbg, prev_bbox, pred_bb)

        pred_bb = BoundingBox(*pred_bb[0].cpu().detach().numpy().tolist())
        pred_bb.unscale(cur_search_region)
        pred_bb.uncenter(curr_frame, search_location, edge_spacing_x, edge_spacing_y)
        x1, y1, x2, y2 = int(pred_bb.x1), int(pred_bb.y1), int(pred_bb.x2), int(pred_bb.y2)
        pred_bb = BoundingBox(x1, y1, x2, y2)
        return pred_bb

    def preprocess(self, im, mean=None):
        """preprocess image before forward pass, this is the same
        preprocessing used during training, please refer to collate function
        in train.py for reference
        @image: input image
        """
        # preprocessing for all pretrained pytorch models
        if mean:
            im = resize(im, (227, 227)) - mean
        else:
            mean = np.array([104, 117, 123])
            im = resize(im, (227, 227)) - mean
        im = image_io.image_to_tensor(im)
        return im

    # Added get bounding box from masks
    def get_bb(self, png_mask, pix_oversize):
        _, thresh = cv2.threshold(png_mask, 127, 255, 0)
        contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # not copying here will throw an error
        
        # Concatenate all contours
        contours = np.concatenate(contours)

        # Determine and draw bounding rectangle
        x_, y_, h_, w_ = cv2.boundingRect(contours)

        x1 = x_
        y1 = y_
        x2 = x_+h_
        y2 = y_+w_
        d_px = pix_oversize
        bbox = BoundingBox(x1-d_px, y1-d_px, x2+d_px, y2+d_px)
        return bbox

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA.x1, boxB.x1)
        yA = max(boxA.y1, boxB.y1)
        xB = min(boxA.x2, boxB.x2)
        yB = min(boxA.y2, boxB.y2)

        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
        boxBArea = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def bb_centroid_dist(self, boxA, boxB):
        center_A_x = (boxA.x1 + boxA.x2)/2
        center_A_y = (boxA.y1 + boxA.y2)/2
        center_B_x = (boxB.x1 + boxB.x2)/2
        center_B_y = (boxB.y1 + boxB.y2)/2       
        return np.sqrt((center_A_x-center_B_x)**2 + (center_A_y-center_B_y)**2)
        
    def track(self):
        """Track"""
        vid_frames = self._vid_frames[0]
        num_frames = len(vid_frames)
        f_path = vid_frames[0]

        # Added mask frames in tracking
        mask_frames = self._mask_frames[0]
        m_path = mask_frames[0]
        mask_0 = cv2.imread(m_path,cv2.IMREAD_GRAYSCALE)#[::-1] # not working
        bbox_0 = self.get_bb(np.asarray(mask_0), pix_oversize=15)
        
        frame_0 = image_io.load(f_path)
        prev = np.asarray(frame_0)#[::-1] #not working
        global image
        image = prev

        while True:
            # prev_out = cv2.cvtColor(prev, cv2.COLOR_RGB2BGR)
            prev_out = np.copy(prev)
            cv2.imshow('image', prev_out)


            # Added draw bounding box ground truth
            curr_dbg = np.copy(prev_out)
            curr_dbg = cv2.rectangle(curr_dbg, (int(bbox_0.x1),
                                                int(bbox_0.y1)),
                                     (int(bbox_0.x2), int(bbox_0.y2)), (255, 255, 0), 2)
            cv2.imshow('image', curr_dbg)
            cv2.waitKey(20)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # (x1, y1), (x2, y2) = refPt[0], refPt[1]
                #bbox_0 = BoundingBox(x1, y1, x2, y2)
                break
            elif key == ord('r'):
                #(x1, y1), (x2, y2) = refPt[0], refPt[1]
                #bbox_0 = BoundingBox(x1, y1, x2, y2)
                break
        

        for i in range(1, num_frames):
            f_path = vid_frames[i]
            frame_1 = image_io.load(f_path)

            # Added masks in each iteration
            m_path = mask_frames[i]
            mask_1 = cv2.imread(m_path,cv2.IMREAD_GRAYSCALE)#[::-1] # not working
            bbox_true = self.get_bb(np.asarray(mask_1), pix_oversize=0)
            
            curr = np.asarray(frame_1)#[::-1]
            bbox_0 = self._track(curr, prev, bbox_0)
            bbox = bbox_0
            prev = curr

            IOU_val = self.bb_intersection_over_union(bbox_true, bbox)
            centroid_dist = self.bb_centroid_dist(bbox_true, bbox)
            print(centroid_dist)

            if cv2.waitKey(1) & 0xFF == ord('p'):
                while True:
                    image = curr
                    cv2.imshow("image", curr)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("s"):
                        (x1, y1), (x2, y2) = refPt[0], refPt[1]
                        bbox_0 = BoundingBox(x1, y1, x2, y2)
                        break

            curr_dbg = np.copy(curr)
            curr_dbg = cv2.rectangle(curr_dbg, (int(bbox.x1),
                                                int(bbox.y1)),
                                     (int(bbox.x2), int(bbox.y2)), (255, 255, 0), 2)
            # draw ground truth
            curr_dbg = cv2.rectangle(curr_dbg, (int(bbox_true.x1),
                                                int(bbox_true.y1)),
                                     (int(bbox_true.x2), int(bbox_true.y2)), (0, 255, 255), 1)

            # curr_dbg = cv2.cvtColor(curr_dbg, cv2.COLOR_RGB2BGR)
            cv2.imshow('image', curr_dbg)
            # cv2.imwrite('./output/{:04d}.png'.format(i), curr_dbg)
            cv2.waitKey(20)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',
                    required=True, help='path to input folder containing all the frames')
    ap.add_argument('--model_dir',
                    required=True, help='model directory')
    # Added argument for masks directory
    ap.add_argument('--ground_truth',
                    required=True, help='ground truth masks directory')

    args = ap.parse_args()
    objG = GoturnTracker(args, dbg=False)
    objG.track()
