import numpy as np
import napari
import tifffile
from skimage import data
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QPushButton, QMainWindow, QCheckBox, QSpinBox, QLabel, QSlider
from qtpy.QtCore import *
import skimage
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square
from utils import fill_label_holes
from skimage.util import map_array


class McLabel(QMainWindow):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_layer = self.viewer.layers['macrophages']
        self.label_layer = self.viewer.add_labels(np.zeros_like(self.image_layer.data, dtype='int32'),
                                                  name='OutputLabel')
        self.label_helper = None
        # start_button
        self.start_button = QPushButton('Draw Mode')
        self.start_button.clicked.connect(self.select_macrophage)
        # self.viewer.window.add_dock_widget(self.start_button, area='left')
        # finish button
        self.finish_button = QPushButton('Compute Label')
        self.finish_button.clicked.connect(self.finish_fn)

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
        self.threshold_slider.setRange(0,50)
        self.threshold_slider.setPageStep(1)
        self.threshold_slider.setValue(15)
        self.threshold_slider.valueChanged.connect(self.manual_threshold_adjustment)

        self.viewer.window.add_dock_widget(
            [self.start_button, self.finish_button, self.preserve_checkbox, self.area_slider_lbl, self.area_slider,
             self.threshold_slider_lbl, self.threshold_slider], area='left', name='McLabel')

        # TODO:
        """
        Implement global flags for draw mode. If draw mode is active, the button changes into an undo button 
        which deletes the label helper if pressed and retains the original state
        
        save bbox coordinates in dict for later adaption
        """

    def select_macrophage(self):
        labels = np.zeros_like(self.image_layer.data, dtype='int32')
        self.label_helper = self.viewer.add_labels(labels, name='label_helper')
        self.label_helper.mode = "PAINT"
        # print("Hello Not-Foo!")

    def finish_fn(self):
        # TODO: implement error handling

        # Get data
        img_patch, (minr, minc, maxr, maxc) = self.get_patch_from_layer()

        # Apply image processing
        filtered_label = self.compute_label_from_patch(img_patch)

        # Refresh layers
        self.label_layer.data[minr:maxr, minc:maxc] = filtered_label  # TODO: affect only labeled pixels
        self.label_layer.refresh()
        self.viewer.layers.remove('label_helper')  # TODO: remove potentially multiple layers
        self.threshold_slider_lbl.setVisible(True)
        self.threshold_slider.setVisible(True)
        self.area_slider_lbl.setVisible(True)
        self.area_slider.setVisible(True)
        print("Done")

    @staticmethod
    def apply_threshold(patch, threshold):
        return patch > threshold

    @staticmethod
    def connected_component(binary_image):
        label_image = label(binary_image)
        label_image = fill_label_holes(label_image)
        return label_image

    def apply_filter(self, lbl_img, condition='area', min_value=100):
        table = regionprops_table(lbl_img, properties=('label', 'area'))
        filt = table[condition] > min_value
        input_label = table['label']
        output_label = input_label * filt
        filtered_label = map_array(lbl_img, input_label, output_label)
        return filtered_label

    def get_patch_from_layer(self):
        labeled_macro = self.label_helper.data
        props = regionprops(labeled_macro)
        print(f'{len(props)=}')
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            print(f'{minr=},{minc=},{maxr=},{maxc=}')
            print(prop.area)
        img_patch = self.image_layer.data[minr:maxr, minc:maxc]
        img_patch[labeled_macro[minr:maxr, minc:maxc] == 0] = 0

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
        self.label_layer.data[minr:maxr, minc:maxc] = filtered_label  # TODO: affect only labeled pixels
        self.label_layer.refresh()
        print(f'DEBUG: {thresh=}')

    def manual_area_adjustment(self, area):
        img_patch, (minr, minc, maxr, maxc) = self.get_patch_from_layer()
        lbl_patch = self.label_layer.data[minr:maxr, minc:maxc]
        filtered_label = self.apply_filter(lbl_patch, min_value=area)
        self.label_layer.data[minr:maxr, minc:maxc] = filtered_label  # TODO: affect only labeled pixels
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
