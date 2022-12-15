import time

import numpy as np
import napari
from scipy import ndimage
from qtpy.QtWidgets import QPushButton, QLabel, QSlider, QWidget, QCheckBox
from qtpy.QtCore import *
import skimage
from skimage.measure import label, regionprops, regionprops_table
from napari_mclabel.utils import fill_label_holes
from skimage.util import map_array
from enum import Enum
from napari_plugin_engine import napari_hook_implementation
import imageio
import csv
import os

class State(Enum):
    DRAW = 1
    COMPUTE = 2
    NO_INIT = 3


def on_label_change(event):
    # old_label = event.old_label  # the old label
    # new_label = event.new_label  # the new label
    #
    # if new_label == old_label + 1:  # check if the label increased by 1
    #     print("Label increased by 1")
    # else:
    #     print("Label changed but didn't increase by 1")
    #print(vars(event._sources[0]))
    pass


class McLabel(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.layers.events.inserted.connect(self.init_layers)

        self.label_helper = None
        self.image_layer = None
        self.label_layer = None

        # start_button
        self.start_button = QPushButton('Draw Label')
        self.start_button.clicked.connect(self.btn_click)

        # Timer for user study
        self.timer_start_stop = QPushButton('Start Timer')
        self.timer_start_stop.clicked.connect(self.timer_fn)

        # Checkbox for user study
        self.manual_mode = False
        self.manual_mode_cb = QCheckBox("Manual Annotation")
        self.manual_mode_cb.setChecked(self.manual_mode)
        self.manual_mode_cb.toggled.connect(self.change_manual_mode)

        # preprocess button
        self.preproc_button = QPushButton('Convert Images')
        self.preproc_button.clicked.connect(self.preproc_fn)

        # Slider for manual adjustment of threshold
        self.threshold_slider_lbl = QLabel("Threshold: ")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setVisible(False)
        self.threshold_slider_lbl.setVisible(False)
        self.threshold_slider.setRange(0, 50)
        self.threshold_slider.setPageStep(1)
        self.threshold_slider.setValue(15)
        self.threshold_slider.valueChanged.connect(self.manual_threshold_adjustment)

        self.viewer.window.add_dock_widget(
            [self.manual_mode_cb, self.start_button, self.preproc_button,self.timer_start_stop, self.threshold_slider_lbl, self.threshold_slider], area='left',
            name='McLabel')

        self.current_max_lbl = 0
        self.start_button.setEnabled(False)

        self.state = State.NO_INIT

        self.normalize = False

        self.timer_active = False
        self.start_time = 0
        self.end_time = 0
        self.duration = 0
        self.logfolder = McLabel.create_logfolder()
        self.filename = os.path.join(self.logfolder, 'log.csv')

        self.logline = "label, minr, minc, maxc, duration"
        self.write_logline(self.logline)

    def change_manual_mode(self):
        self.manual_mode = self.manual_mode_cb.isChecked()
        print(f'Set manual mode to {self.manual_mode}')
        if self.label_helper is not None:
            self.viewer.layers.remove(self.label_helper)
            self.label_helper = None
            self.viewer.layers.selection.active = self.viewer.layers["OutputLabel"]
            self.start_button.setEnabled(False)
            self.label_layer.brush_size = 3
    @staticmethod
    def create_logfolder():
        home_dir = os.path.expanduser("~")
        folder_name = "mclabel"
        folder_path = os.path.join(home_dir, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created log folder at {folder_path}")
        return folder_path

    def init_layers(self, event):
        if len(self.viewer.layers) == 1:
            # If an image was added we can now allow labelling (i.e. enable the button)
            self.start_button.setEnabled(True)

    def write_logline(self, logline):
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([logline])
        self.logline = ""


    def timer_fn(self):
        if self.timer_active:
            self.end_time = time.time()
            self.duration = round(time.time() - self.start_time)
            print(f'It took {self.duration} seconds')
            if self.manual_mode:
                self.logline += str(f'{self.label_layer.selected_label},')
            self.logline += str(self.duration)
            print(self.logline)
            self.write_logline(self.logline)
            self.timer_active = False
            self.timer_start_stop.setText("Start Timer")
        else:
            self.start_time = time.time()
            self.timer_active = True
            self.timer_start_stop.setText("Stop Timer")

    def preproc_fn(self):
        if isinstance(self.viewer.layers[0].data, napari.layers._multiscale_data.MultiScaleData):
            # user manually selects all layers which shall be considered for MIP
            num_layers_selected = len(self.viewer.layers.selection)
            assert num_layers_selected > 0, "At least one layer must be selected"

            singleton = self.viewer.layers[0].data.shape[
                            0] == 1  # sometimes the ims loader returns the array with singleton

            if singleton:  # maybe we do not need this
                _, z, y, x = self.viewer.layers[0].data[0].shape
            else:
                z, y, x = self.viewer.layers[0].data[0].shape

            max_intensity_img = np.zeros((2, y, x))
            # TODO:
            # merge max value from all selected layers into one channel
            # channels that are not selected will remain as distinct channels
            for selected in self.viewer.layers.selection:
                sel = np.asarray(selected.data[0])[0:self.viewer.layers[0].data[0].shape[0]]
                current_mip = np.amax(sel, axis=0)
                np.copyto(max_intensity_img[0], current_mip, where=current_mip > max_intensity_img[0])

            probably_nuc_channel = np.asarray(
                [l for l in self.viewer.layers if l not in self.viewer.layers.selection][0].data[0].copy())[
                                   0:self.viewer.layers[0].data[0].shape[0]]
            max_intensity_img[1] = np.amax(probably_nuc_channel, axis=0)

        else:  # no ims file, image presumed to be already processed
            max_intensity_img = self.viewer.layers[0].data.copy()

        self.viewer.layers.clear()
        self.viewer.add_image(max_intensity_img,
                              channel_axis=0,
                              name=["macrophages", "nuclei"],
                              colormap=["gray", "magenta"],
                              contrast_limits=[[0, 75], [0, 75]],
                              )
        self.image_layer = self.viewer.layers['macrophages']
        self.label_layer = self.viewer.add_labels(np.zeros_like(self.image_layer.data, dtype='int32'),
                                                  name='OutputLabel')
        self.label_layer.events.selected_label.connect(on_label_change)
        # self.label_layer.selected_label = 0
        self.viewer.reset_view()
        self.preproc_button.setVisible(False)
        self.start_button.setText("Compute Label")
        self.draw_fn()

    def btn_click(self):
        if self.state == State.DRAW:
            self.start_button.setText("Draw Label")
            self.compute_fn()
        elif self.state == State.COMPUTE:
            self.start_button.setText("Compute Label")
            self.draw_fn()

    def draw_fn(self):
        self.state = State.DRAW
        labels = np.zeros_like(self.image_layer.data, dtype='int32')
        self.label_helper = self.viewer.add_labels(labels, name='label_helper')
        self.label_helper.brush_size = 3
        self.viewer.layers[0].selected = False
        self.viewer.layers['label_helper'].selected = True

        self.label_helper.mode = "PAINT"
        # print("Hello Not-Foo!")

    def compute_fn(self):
        self.state = State.COMPUTE
        # Get data
        img_patch, (minr, minc, maxr, maxc) = self.get_patch_from_layer()

        # Apply image processing
        filtered_label = self.compute_label_from_patch(img_patch).astype('int32')

        # Adapt counting of filtered label to current_max_lbl + 1
        self.current_max_lbl += 1
        self.logline += str(self.current_max_lbl) + ","
        self.logline += f'{minr}, {minc}, {maxr}, {maxc},'
        filtered_label[filtered_label != 0] = self.current_max_lbl

        # Refresh layers
        label_patch = self.label_layer.data[minr:maxr, minc:maxc].copy()
        out_patch_1 = np.where(label_patch == 0, filtered_label, label_patch)
        self.label_layer.data[minr:maxr, minc:maxc] = out_patch_1

        self.label_layer.refresh()

        self.remove_helper()

        self.threshold_slider_lbl.setVisible(True)
        self.threshold_slider.setVisible(True)
        print("Done")

    @staticmethod
    def apply_threshold(patch, threshold):
        binary = patch > threshold
        binary = ndimage.binary_fill_holes(binary).astype('int32')
        return binary

    @staticmethod
    def connected_component(binary_image):
        label_image = label(binary_image)
        # label_image = fill_label_holes_cv2(label_image)
        label_image = fill_label_holes(label_image)
        return label_image

    def remove_helper(self):
        [self.viewer.layers.remove(str(layer)) for layer in reversed(self.viewer.layers) if
         "label_helper" in str(layer)]

    def change_state(self, state):
        """Change state of McLabel"""
        # TODO: write TODO for this function
        pass

    def apply_filter(self, lbl_img, condition='area', min_value=100):
        table = regionprops_table(lbl_img, properties=('label', 'area'))
        print(f"DEBUG: {table['area'].max()=}")
        # filt = table[condition] > min_value
        filt = table[condition] == table['area'].max()
        input_label = table['label']
        output_label = input_label * filt
        filtered_label = map_array(lbl_img, input_label, output_label)
        return filtered_label.astype('int32')

    def get_patch_from_layer(self):
        labeled_macro = self.label_helper.data.copy()
        labeled_macro = ndimage.binary_fill_holes(labeled_macro).astype('int32')
        props = regionprops(labeled_macro)
        print(f'{len(props)=}')
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            print(f'{minr=},{minc=},{maxr=},{maxc=}')
            print(prop.area)
        img_patch = self.image_layer.data[minr:maxr, minc:maxc].copy()
        img_patch[labeled_macro[minr:maxr, minc:maxc] == 0] = 0  # removes parts outside hand-drawn region.

        imageio.imsave(os.path.join(self.logfolder, f'{self.current_max_lbl}.png'), img_patch)
        return img_patch, (minr, minc, maxr, maxc)

    def compute_label_from_patch(self, img_patch, thresh=None, min_area=None):
        if thresh is None:
            thresh = skimage.filters.threshold_triangle(img_patch, nbins=64)
        print(f'DEBUG: {thresh=}')
        binary = McLabel.apply_threshold(img_patch, thresh)
        label_image = McLabel.connected_component(binary)
        if min_area is not None:
            filtered_label = self.apply_filter(label_image, min_value=min_area)
        else:
            filtered_label = self.apply_filter(label_image)
        return filtered_label

    def manual_threshold_adjustment(self, thresh):
        img_patch, (minr, minc, maxr, maxc) = self.get_patch_from_layer()
        filtered_label = self.compute_label_from_patch(img_patch, thresh=thresh)
        label_patch = self.label_layer.data[minr:maxr, minc:maxc].copy()
        # make sure that we keep original label id when changing threshold
        if label_patch.max():
            # filtered_label[filtered_label != 0] = label_patch.max()
            filtered_label[filtered_label != 0] = self.current_max_lbl

        out_patch = label_patch.copy()
        np.copyto(out_patch, filtered_label, where=label_patch == 0)
        np.copyto(out_patch, filtered_label, where=label_patch == self.current_max_lbl)
        self.label_layer.data[minr:maxr, minc:maxc] = out_patch
        self.label_layer.refresh()
        print(f'DEBUG: {thresh=}')


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return McLabel

def main():
    # Load sample image
    viewer = napari.Viewer()
    win = McLabel(viewer)
    input('Press ENTER to exit')


if __name__ == "__main__":
    main()
