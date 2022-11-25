from imaris_ims_file_reader.ims import ims

from napari_plugin_engine import napari_hook_implementation
import os


def reader(path):
    img = ims(path)
    img = img[0]
    return img


@napari_hook_implementation
def napari_get_reader(path):
    if isinstance(path,str) and os.path.splitext(path)[1].lower() == '.ims':
        return reader
