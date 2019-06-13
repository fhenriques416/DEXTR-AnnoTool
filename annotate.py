#!/usr/bin/env python

import argparse
from os import path, makedirs, listdir

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

import tensorflow as tf
from keras import backend as K

# ---user-defined modules---
# repo defined
from networks.dextr import DEXTR
from mypath import Path
from helpers import helpers as helpers
from helpers import annotools as tools

modelName = 'dextr_pascal-sbd'
pad = 50
thres = 0.8
gpu_id = 0

if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='annotation tool based on DEXTR for annotating segmentation masks for'
                                                 ' images')
    required_arguments = parser.add_argument_group('Required arguments')
    required_arguments.add_argument('-c', '--config', type=str, help='path to the yaml file that holds the'
                                                                     ' configuration', required=True)
    args = parser.parse_args()

    # ---------------------------------Setting up based on yaml config--------------------------------------------------

    # load (and check) the user defined config
    cfg = tools.load_config(args.config)
    # setup progress file: check config to detect what should be done, return files that need to be annotated
    cfg, files_to_annotate = tools.config_prog_log(cfg, args.config)

    if not files_to_annotate:
        print("Yippie...finished annotating all files in the source folder. Exiting...")
        exit(0)

    # ------------------------------------Setup TensorFlow session------------------------------------------------------
    # Handle input and output args
    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        net = DEXTR(nb_classes=1, resnet_layers=101, input_shape=(512, 512), weights=modelName,
                    num_input_channels=4, classifier='psp', sigmoid=True)

        # -----------------------------------Start annotation interface-------------------------------------------------
        _KEEPannotating = True  # annotation status flag

        for filename in files_to_annotate:

            if _KEEPannotating:
                # read image
                imagepath = path.join(cfg['required']['source'], filename)
                image = np.array(Image.open(imagepath))

                # image level status flags
                _ANNOinit = False
                _ANNOwrong = False

                # setup annotation UI
                if not _ANNOinit:
                    plt.switch_backend('TkAgg') # force 'TkAgg' backend
                    ui = plt.figure('Annotation Interface')
                    ui.tight_layout()

                # maximize window and setup interactive mode
                manager = plt.get_current_fig_manager()
                manager.resize(*manager.window.maxsize())
                plt.ion()
                # setup gridspec
                gs = gridspec.GridSpec(2, 2, width_ratios=[5,1], height_ratios=[11,1])
                # setup window for images
                ui_img = ui.add_subplot(gs[0])
                ui_img.axis('off')
                ui_img.imshow(image)
                # window for annotation instructions
                ui_instruct = ui.add_subplot(gs[1])
                ui_instruct.axis('off')
                instruct_msg = ui_instruct.text(0, 0,
                                                'Instructions:\n1. Use the left mouse button to select the following'
                                                ' four extreme points for an object in the image:\n-left,'
                                                '\n-right,\n-top, and\n-bottom.\n2. In case of misselection, use'
                                                ' right mouse button to undo.\n3. When the 4 points are selected,'
                                                ' the resulting masks are overlayed.\n4. Status messages are '
                                                'displayed at bottom of the image; perform next actions based on'
                                                'them.', fontsize = 14, wrap = True,
                                                transform = ui_instruct.transAxes)

                ui_status = ui.add_subplot(gs[2]) # window for displaying status
                ui_status.axis('off')


                # hold predictions from image
                image_results = []

                # keep running till user says he had finished annotating all objects in an image
                while True:

                    if not _ANNOinit:
                        ui_msg = ui_status.text(0.0, 0.0, 'Start annotation...', color='green', fontsize=14)
                        _ANNOinit = True

                    elif _ANNOinit:
                        if not _ANNOwrong:
                            ui_msg = ui_status.text(0.0, 0.0, 'Continue annotation...', color='green',
                                             fontsize=14)
                        elif _ANNOwrong:
                            ui_img.imshow(helpers.overlay_masks(image / 255, image_results))
                            ui_msg = ui_status.text(0.0, 0.0, 'Ok. redo annotation...', color='green',
                                             fontsize=14)
                            _ANNOwrong = False

                    extreme_points_ori = np.array(plt.ginput(4, timeout=0, mouse_add=1)).astype(np.int)
                    ui_msg.remove()

                    # Crop image to the bounding box from the extreme points and resize
                    bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=True)
                    crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
                    resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

                    # Generate extreme point heat map normalized to image values
                    extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]),
                                                           np.min(extreme_points_ori[:, 1])] + [pad, pad]
                    extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(
                        np.int)
                    extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
                    extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

                    # Concatenate inputs and convert to tensor
                    input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)

                    # Run a forward pass
                    pred = net.model.predict(input_dextr[np.newaxis, ...])[0, :, :, 0]
                    result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

                    # Mask correct or not
                    image_results.append(result)  # add result to list of results even if incorrect for vis purposes
                    plot_mask = ui_img.imshow(helpers.overlay_masks(image / 255, image_results))
                    ui_msg = ui_status.text(0.0, 0.0, 'Is the generated mask correct? Click mouse for yes,'
                                                 ' press r to redo', color='orange', fontsize=14)
                    not_anno_correct = plt.waitforbuttonpress()
                    ui_msg.remove()

                    if not_anno_correct:
                        wrong_anno_mask = image_results.pop()
                        _ANNOwrong = True
                        continue

                    # Add extreme points to show annotation done for the object
                    plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
                    # Check what user wants to do next
                    ui_msg = ui_status.text(0.0, 0.0,
                                                 'Next...Press d if done annotating this image or click mouse to'
                                                 ' continue annotating more objects in this image', color='orange',
                                      fontsize=14)
                    usr_choice = plt.waitforbuttonpress()
                    ui_msg.remove()

                    if usr_choice:  # if user presses d then break
                        ui_msg = ui.text(0.0, 0.0, 'Annotation of this image stopped, mask image saved'
                                                                   ' and progress updated', color='red', fontsize=14)
                        break

                # Show and save the generated masks
                generated_mask = tools.get_img_segmasks(image, image_results)
                mask_uint8 = tools.save_mask(generated_mask, cfg, filename)

                # update figure to show subplots
                ui.clear()
                ax1, ax2 = ui.subplots(1,2)
                ui.tight_layout()
                # subplot-1: image
                ax1.axis('off')
                ax1.imshow(image)
                ax1.set_title('Original Image')
                # subplot-2: mask
                ax2.axis('off')
                ax2.imshow(mask_uint8, cmap='gray')
                ax2.set_title('Mask Image\n Note: Mask value scaled to [0 255] for visualisation')
                # update message at bottom
                ax1_lowercorner = ax1.get_position().corners()
                ui_msg= ui.text(ax1_lowercorner[0][0], (ax1_lowercorner[0][1] - 0.02),
                                'Annotate next image? Click mouse for yes, press e to exit annotation and resume later',
                                color='orange', fontsize=14)
                # ask for user response
                usr_choice = plt.waitforbuttonpress()
                if usr_choice:  # if user presses e then break
                    _KEEPannotating = False
                ui_msg.remove()
            else:
                print("Annotation of images stopped. Your progress has been saved. Run script again to resume "
                      "annotation. Exiting...")
                exit(0)

        print("Well Done...Finished annotating all files in the source folder. Exiting...")

