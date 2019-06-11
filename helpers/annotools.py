import sys
import io
from ruamel.yaml import YAML
import ruamel.yaml.util as ymlutil
from os import path, makedirs, listdir
import warnings
from PIL import Image
import numpy as np


# configure ruamel.yaml
yaml = YAML()
yaml.default_flow_style = False


# setup custom exception classes
class ConfigFileWarning(UserWarning):
    pass


class ConfigFileError(Exception):
    pass


# ----------------------------------------General helper functions------------------------------------------------------

def _query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def _files_to_annotate(source):
    dir_contains = listdir(source)
    to_annotate = []
    for name in dir_contains:
        if name.endswith('.jpg') or name.endswith('.png'):
            to_annotate.append(name)

    return to_annotate


# -------------------------------------Functions for progress tracking--------------------------------------------------

def _update_filecache(config, filename):
    # parse file names already in the progress logging file
    with io.open(filename, 'r') as progress_file:
        _progress = yaml.load(progress_file)

    curr_files = list(_progress['annotated'])
    all_files = _files_to_annotate(config['required']['source'])
    # make new dict based on files not in currently listed files (curr_files)
    new_files = [f for f in all_files if f not in curr_files]
    new_files.sort()  # sort the list
    tmp_extra_dict = {f: False for f in new_files}
    # append this new dict to current dict
    _progress['annotated'].update(tmp_extra_dict)

    # write this to file
    with io.open(filename, 'w+') as progress_file:
        yaml.dump(_progress, progress_file)


def _get_to_annotate(filename):
    # load progress file again and return list of keys (filenames) from annotated section
    with io.open(filename, 'r') as progfile:
        progress = yaml.load(progfile)

    files_anno = []  # to hold file names for files to be annotated
    for file, status in list(progress['annotated'].items()):
        if status is False:
            files_anno.append(file)

    return files_anno


def _update_prog(config, source_filename):

    # open and edit the progress file
    progressfile_path = path.join(config['required']['source'], config['admin']['progress_file'])

    with io.open(progressfile_path, 'r') as progress_file:
        progress = yaml.load(progress_file)

    progress['annotated'][source_filename] = True

    # save update to progress file
    with io.open(progressfile_path, 'w+') as progress_file:
        yaml.dump(progress, progress_file)


def _make_progfile(config, filename):
    # parse the folder and add filename to a dict
    files = _files_to_annotate(config['required']['source'])
    files.sort()  # return sorted list

    if not files:
        raise ConfigFileError("No jpg or png files found in the source folder specified in the config file. Aborting "
                              "creation of progress file and exiting...")

    # make a temporary dict to store file names and annotation status
    tmp_anno_dict = {f: False for f in files}
    # make a temporary dict to get come info from the config file dict with the the same section headers
    tmp_info_dict = config['required']

    # join the dicts into common dict to be saved into the progress tracking file
    _progress = {'required': tmp_info_dict, 'annotated': tmp_anno_dict}

    with io.open(filename, 'w') as progfile:
        yaml.dump(_progress, progfile)

    print("Annotation progress tracking file created.\n\n"
          "********************************************************************************\n"
          "Tip: if the source directory changes (i.e. if files are added/removed), in config\n"
          "file set 'update: True' to update the file-names cache in the progress tracker.\n"
          "********************************************************************************")


def config_prog_log(config, cfg_file):

    # progress file settings
    _progress_filename = path.join(config['required']['source'], config['admin']['progress_file'])

    # when progress file doesn't exist
    if not path.isfile(_progress_filename):
        # idiot check
        if config['admin']['reset_progress']:
            warnings.warn("You are resetting the progress file before it was setup. Ignoring 'reset: True' and setting"
                          "'reset: False' in config file to prevent any errors later.", ConfigFileWarning)
            sys.stderr.flush()  # to print warning immediately
            _modify_config(cfg_file, 'reset_progress', False)
        if config['optional']['update']:
            warnings.warn("You are updating the progress file before it was setup. Ignoring 'update: True' and"
                          " continuing to make the progress file. If required, please change 'update:' status in the"
                          " config file.", ConfigFileWarning)
            sys.stderr.flush()  # to print warning immediately

        # prompt user that file will be created
        print("Progress file that tracks your annotation history doesn't exist. So:\n"
              "(1) either this is the first time you are annotating files in the provided source folder, or\n"
              "(2) you deleted the file. Please recover it if you want to continue from last annotated image.\n"
              "In either case, will try to create a new progress tracking file now...")
        _make_progfile(config, _progress_filename)

    # when progress file exists
    elif path.isfile(_progress_filename):
        # idiot check
        if config['admin']['reset_progress'] and config['optional']['update']:
            raise ConfigFileError("You've set 'reset: True' and 'update: True'. What do you want? If you want to...\n"
                  "- reset annotation progress, then set 'reset: True' and 'update: False'.\n"
                  "- update annotation progress file-cache, then set 'update: True' and 'reset: False'.\n"
                  "Exiting now...")
        elif config['admin']['reset_progress']:
            reset_prompt = _query_yes_no("In config file 'reset: True'. Thus, you'll lose your annotation progress so "
                                        "far.\n Are you sure? Enter 'yes' to continue resetting or 'no' otherwise")
            if reset_prompt:
                print("Okay then. Remember you've been warned. Resetting...")
                _make_progfile(config, _progress_filename)
            elif not reset_prompt:
                print("Phew...be careful don't modify stuff you don't understand. Ignoring resetting...")
        elif config['optional']['update']:
            print("In config file 'update: True', so source folder will be scanned again and new files will be added to"
                  " progress tracking cache.")
            _update_filecache(config, _progress_filename)

    # load config file again in case any modifications were made
    with io.open(cfg_file, 'r') as conffile:
        updated_config = yaml.load(conffile)

    to_annotate = _get_to_annotate(_progress_filename)

    return updated_config, to_annotate


# -------------------------------------Configuration file operations----------------------------------------------------

def _modify_config(cfg_file, param, param_new_value=False):
    # open file
    config, ind, bsi = ymlutil.load_yaml_guess_indent(open(cfg_file))

    if param is 'reset_progress':
        config['admin']['reset_progress'] = param_new_value
    elif param is 'update':
        config['optional']['update'] = param_new_value

    with io.open(cfg_file, 'w') as conffile:
        yaml.dump(config, conffile)


def load_config(cfg_file):

    # load config from specified yaml file
    with io.open(cfg_file, 'r') as conffile:
        config = yaml.load(conffile)

    # source folder exists?
    if not path.exists(config['required']['source']):
        raise ConfigFileError("The source folder doesn't exist. Please properly set folder in config file. Exiting...")

        # destination folder exists? No, make it.
    if not path.exists(config['required']['destination']):
        warnings.warn("The specified destination folder for segmasks doesn't exist, making it.", ConfigFileWarning)
        sys.stderr.flush()  # to print warning immediately
        makedirs(config['required']['destination'])

    return config


# --------------------------------Functions for generating and saving mask images---------------------------------------

def get_img_segmasks(im, masks):
    orig_img = im.copy()
    # not sure if this is required
    if isinstance(masks, np.ndarray):
        masks = [masks]
    # full image mask
    img_mask = np.zeros([im.shape[0], im.shape[1]])  # init full image mask with zeros
    for n_mask in range(len(masks)):
        curr_mask = masks[n_mask]
        img_mask[curr_mask] = n_mask + 1

    return img_mask


def save_mask(mask_array, config, source_filename):
    # get the mask image name
    source_file_initial, _ = source_filename.split('.')
    mask_file_name = source_file_initial + config['optional']['mask_suffix']

    # make mask image from mask array and save it to disk
    mask_array_mod = mask_array.astype(np.uint8)  # convert array to uint8 before saving

    if config['optional']['save_format'] != 'png' and config['optional']['save_format'] != 'npy':
        raise ConfigFileError("Mask save format can only be 'png' or 'npy'. Change setting in config file.")
    elif config['optional']['save_format'] == 'png':
        mask_img = Image.fromarray(mask_array_mod, mode='L')  # mode L is for 8-bit pixels, black and white
        mask_name_full = mask_file_name + '.png'
        mask_save_path = path.join(config['required']['destination'], mask_name_full)
        mask_img.save(mask_save_path)
    elif config['optional']['save_format'] == 'npy':
        mask_save_path = path.join(config['required']['destination'], mask_file_name)
        np.save(mask_save_path, mask_array_mod)

    # now update the progress file
    _update_prog(config, source_filename)

    return mask_array_mod
