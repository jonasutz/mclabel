import numpy as np
import napari
import tifffile
from scipy import ndimage
from skimage import data
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QPushButton, QMainWindow, QCheckBox, QSpinBox, QLabel, QSlider, QShortcut
from qtpy.QtCore import *
from qtpy.QtGui import QKeySequence
import skimage
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square
from utils import fill_label_holes, fill_label_holes_cv2
from skimage.util import map_array
from enum import Enum

class State(Enum):
    IDLE = 1
    DRAW = 2
    COMPUTE = 3

class McLabel(QMainWindow):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_layer = self.viewer.layers['macrophages']
        self.label_layer = self.viewer.add_labels(np.zeros_like(self.image_layer.data, dtype='int32'),
                                                  name='OutputLabel')
        self.label_helper = None
        # start_button
        self.start_button = QPushButton('Draw new Label')
        self.start_button.clicked.connect(self.select_macrophage)
        # self.viewer.window.add_dock_widget(self.start_button, area='left')
        # finish buttono
        self.finish_button = QPushButton('Compute Label')
        self.finish_button.clicked.connect(self.finish_fn)



        self.compute_label_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_P), self)
        self.compute_label_shortcut.activated.connect(self.finish_fn)


        # Checkbox for preserve labels
        self.preserve_checkbox = QCheckBox("Preserve Label")

        # Slider for area which will be discarded
        self.area_slider_lbl = QLabel('Min Area: ')
        self.area_slider = QSlider(Qt.Orientation.Horizontal)
        self.area_slider.setRange(250, 250000)
        self.area_slider.setValue(100)
        self.area_slider.setPageStep(250)
        self.area_slider.setVisible(False)
        self.area_slider_lbl.setVisible(False)
        self.area_slider.valueChanged.connect(self.manual_area_adjustment)

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
            [self.start_button, self.finish_button, self.preserve_checkbox, self.area_slider_lbl, self.area_slider,
             self.threshold_slider_lbl, self.threshold_slider], area='left', name='McLabel')

        self.current_max_lbl = 0
        self.finish_button.setEnabled(False)
        self.select_macrophage()
        # TODO:
        """
        Implement global flags for draw mode. If draw mode is active, the button changes into an undo button 
        which deletes the label helper if pressed and retains the original state
        
        save bbox coordinates in dict for later adaption
        """

    def select_macrophage(self):
        self.finish_button.setEnabled(True)
        labels = np.zeros_like(self.image_layer.data, dtype='int32')
        self.label_helper = self.viewer.add_labels(labels, name='label_helper')
        self.label_helper.mode = "PAINT"
        # print("Hello Not-Foo!")

    def finish_fn(self):
        # TODO: implement error handling

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
        self.area_slider_lbl.setVisible(True)
        self.area_slider.setVisible(True)
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
        img_patch[labeled_macro[minr:maxr, minc:maxc] == 0] = 0 # removes parts outside of handdrawn region.

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

    def manual_area_adjustment(self, area):
        img_patch, (minr, minc, maxr, maxc) = self.get_patch_from_layer()
        lbl_patch = self.label_layer.data[minr:maxr, minc:maxc]
        filtered_label = self.apply_filter(lbl_patch, min_value=area)
        label_patch = self.label_layer.data[minr:maxr, minc:maxc]
        out_patch_1 = np.where(label_patch == 0, filtered_label, label_patch)
        out_patch_2 = np.where(label_patch == self.current_max_lbl, filtered_label, label_patch)
        out_patch = out_patch_1 + out_patch_2
        self.label_layer.data[minr:maxr, minc:maxc] = out_patch
        self.label_layer.refresh()
        print(f'DEBUG: {area=}')



def main():
    # Load sample image
    img = tifffile.imread(
        "/Users/jonas/Library/Mobile Documents/com~apple~CloudDocs/FAU/AIMI/PhD/03_Datasets/ouma_macrophages/TIF/CD115_l_Ctrl _Merged Image 12.tiff")

    viewer = napari.view_image(
        img,
        channel_axis=0,
        name=["macrophages", "nuclei"],
        colormap=["gray", "magenta"],
        contrast_limits=[[0, 75], [0, 75]],
    )
    _ = McLabel(viewer)
    input('Press ENTER to exit')


if __name__ == "__main__":
    main()
