import numpy as np
import napari
import tifffile
from scipy import ndimage
from skimage import data
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QPushButton, QMainWindow, QCheckBox, QSpinBox, QLabel, QSlider, QShortcut, QWidget
from qtpy.QtCore import *
from qtpy.QtGui import QKeySequence
import skimage
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square
from utils import fill_label_holes, fill_label_holes_cv2
from skimage.util import map_array
from enum import Enum
from napari_plugin_engine import napari_hook_implementation


class State(Enum):
    DRAW = 1
    COMPUTE = 2
    NO_INIT = 3


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
            [self.start_button, self.preproc_button, self.threshold_slider_lbl, self.threshold_slider], area='left',
            name='McLabel')

        self.current_max_lbl = 0
        self.start_button.setEnabled(False)

        self.state = State.NO_INIT

        self.normalize = False

    def init_layers(self, event):
        if len(self.viewer.layers) == 1:
            # If an image was added we can now allow labelling (i.e. enable the button)
            self.start_button.setEnabled(True)

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
        img_patch[labeled_macro[minr:maxr, minc:maxc] == 0] = 0  # removes parts outside of handdrawn region.

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
