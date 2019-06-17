#!/usr/bin/env python

import argparse
from os import path

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

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

        # Initialise the UI
        plt.switch_backend('TkAgg')  # force 'TkAgg' backend
        ui = plt.figure('Annotation Interface')
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize()) # maximize window
        plt.ion()  # setup interactive mode

        for filename in files_to_annotate:

            if _KEEPannotating:

                # read image
                imagepath = path.join(cfg['required']['source'], filename)
                image = np.array(Image.open(imagepath))

                # image level status flags
                _ANNOinit = False
                _lastANNOwrong = False
                _SKIP_IMAGE = False

                # -----Setup figure for annotating -----
                ui.clear()

                gs = ui.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[11, 1])
                # setup window for images
                ui_img = ui.add_subplot(gs[0])
                ui_img.axis('off')
                ui_img.imshow(image)
                # window for annotation instructions
                ui_instruct = ui.add_subplot(gs[1], sharex=ui_img, sharey=ui_img)
                ui_instruct.axis('off')
                instruct_msg = ui_instruct.text(0, 0,
                                                'Instructions:\n'
                                                '1. Use the left mouse button to select the\n'
                                                'following four extreme points for an object:\n'
                                                '- left,\n'
                                                '- right,\n'
                                                '- top, and\n'
                                                '- bottom.\n\n'
                                                '2. In case of misselection, use right mouse\n'
                                                'button to undo.\n\n'
                                                '3. When the 4 points are selected, the\n'
                                                'resulting masks are overlayed.\n\n'
                                                '4. Status messages are displayed at bottom\n'
                                                'of the image; provide input based on them.\n\n'
                                                '5. Close window to exit.',
                                                fontsize=12, wrap=True, va='top', ha='left')
                # window for displaying status
                ui_status = ui.add_subplot(gs[2], sharey=ui_img)  # window for displaying status
                ui_status.axis('off')

                # hold predictions from image
                image_results = []

                # keep running till user says he had finished annotating all objects in an image
                while True:

                    if not _ANNOinit:
                        ui_msg = ui_status.text(0, 0, 'Start annotation...', color='green', wrap=True, fontsize=14,
                                                va='top', ha='left')
                        _ANNOinit = True

                    elif _ANNOinit:
                        if not _lastANNOwrong:
                            ui_msg = ui_status.text(0, 0, 'Continue annotation...', color='green', wrap=True,
                                                    fontsize=14, va='top', ha='left')
                        elif _lastANNOwrong:
                            ui_img.imshow(helpers.overlay_masks(image / 255, image_results))
                            ui_msg = ui_status.text(0, 0, 'Ok. try again...', color='green', wrap=True,
                                                    fontsize=14, va='top', ha='left')


                    # user input required
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

                    if _lastANNOwrong:
                        ui_msg = ui_status.text(0, 0, 'Is the mask now correct? Click left mouse for yes, press \'n\' '
                                                      'for no.',
                                                color='orange', wrap=True, fontsize=14, va='top', ha='left')
                        usr_choice = plt.waitforbuttonpress()
                        if usr_choice:
                            ui_msg.remove()
                            ui_msg = ui_status.text(0, 0, 'Okay, to try annotating one more time click left mouse or '
                                                          'press \'n\' to skip to the new image.',
                                                    color='red', wrap=True, fontsize=14, va='top', ha='left')
                            usr_choice_skip = plt.waitforbuttonpress()
                            if usr_choice_skip:
                                ui_msg.remove()
                                _SKIP_IMAGE = True
                                _lastANNOwrong = False
                                break
                            else:
                                ui_msg.remove()
                                wrong_anno_mask = image_results.pop()
                                _lastANNOwrong = True
                                continue
                        else:
                            _lastANNOwrong = False
                            ui_msg.remove()
                    else:
                        ui_msg = ui_status.text(0, 0, 'Is the generated mask correct? Click left mouse for yes, press'
                                                      ' \'r\' to redo.',
                                                color='orange', wrap=True, fontsize=14, va='top', ha='left')
                        not_anno_correct = plt.waitforbuttonpress()
                        ui_msg.remove()
                        if not_anno_correct:
                            wrong_anno_mask = image_results.pop()
                            _lastANNOwrong = True
                            continue

                    # Add extreme points to show annotation done for the object
                    ui_img.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')
                    # Check what user wants to do next
                    ui_msg = ui_status.text(0, 0, 'Click left mouse to continue annotating this image or press \'d\''
                                                  ' if done.',
                                            color='orange', wrap=True, fontsize=14, va='top', ha='left')
                    usr_choice = plt.waitforbuttonpress()
                    ui_msg.remove()

                    if usr_choice:  # if user presses d then break
                        ui_msg = ui_status.text(0, 0, 'Annotation of this image stopped, mask image saved and progress'
                                                      ' updated',
                                                color='red', wrap=True, fontsize=14, va='top', ha='left')
                        break

                if _SKIP_IMAGE:  # check if the break was initialised by skipping image
                    tools.prog_update_skip(cfg, filename) # update prog file
                    continue  # if so, then continue on to next image

                # Show and save the generated masks
                generated_mask = tools.get_img_segmasks(image, image_results)
                mask_uint8 = tools.save_mask(generated_mask, cfg, filename)

                # -----Setup figure for showing annotation results -----
                ui.clear()
                # update figure to show subplots
                gs_res = ui.add_gridspec(2, 2, height_ratios=[11, 1])
                # subplot-1: image
                ui_res_img = ui.add_subplot(gs_res[0])
                ui_res_img.axis('off')
                ui_res_img.imshow(image)
                ui_res_img.set_title('Original Image')
                # subplot-2: mask image
                ui_res_mask = ui.add_subplot(gs_res[1])
                ui_res_mask.axis('off')
                ui_res_mask.imshow(mask_uint8, cmap='gray')
                ui_res_mask.set_title('Mask Image\n Note: Mask value scaled to [0 255] for visualisation')
                # subplot-3: status message
                ui_res_status = ui.add_subplot(gs_res[2], sharey=ui_res_img)  # window for displaying status
                ui_res_status.axis('off')
                ui_res_msg = ui_res_status.text(0.0, 0.0, 'Annotate next image? Click left mouse for yes or press \'e\''
                                                          ' to exit annotation and resume later',
                                                color='orange', wrap=True, fontsize=14, va='top', ha='left')

                # user input required
                usr_choice = plt.waitforbuttonpress()
                if usr_choice:  # if user presses e then break
                    _KEEPannotating = False
                ui_res_msg.remove()
            else:
                print("Annotation of images stopped. Your progress has been saved. Run script again to resume "
                      "annotation. Exiting...")
                exit(0)

        print("Well Done...Finished annotating all files in the source folder. Exiting...")
