import numpy as np
import napari
import tifffile
from skimage import data
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QPushButton, QMainWindow, QCheckBox, QSpinBox, QLabel
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
        self.label_layer = self.viewer.add_labels(np.zeros_like(self.image_layer.data, dtype='int32'), name='OutputLabel')
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

        # Spinner for number of bins
        self.nbins_spinbox = QSpinBox()
        self.nbins_spinbox.setValue(64)

        # Spinner for area which will be discarded
        self.area_spinbox_lbl = QLabel('Min Area: ')
        self.area_spinbox = QSpinBox()
        self.area_spinbox.setValue(100)
        self.area_spinbox.setRange(5,250)
        #self.area_spinbox.setText("Min. Area: ")
        self.viewer.window.add_dock_widget([self.start_button, self.finish_button, self.preserve_checkbox, self.nbins_spinbox, self.area_spinbox_lbl, self.area_spinbox], area='left', name='McLabel')

        # TODO:
        """
        Implement global flags for draw mode. If draw mode is active, the button changes into an undo button 
        which deletes the label helper if pressed and retains the original state
        """

    def select_macrophage(self):
        labels = np.zeros_like(self.image_layer.data, dtype='int32')
        self.label_helper = self.viewer.add_labels(labels, name='label_helper')
        self.label_helper.mode = "PAINT"
        #print("Hello Not-Foo!")

    def finish_fn(self):
        # TODO: implement error handling
        labeled_macro = self.label_helper.data
        props = regionprops(labeled_macro)
        print(f'{len(props)=}')
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            print(f'{minr=},{minc=},{maxr=},{maxc=}')
        img_patch = self.image_layer.data[minr:maxr, minc:maxc]
        img_patch[labeled_macro[minr:maxr, minc:maxc] == 0] = 0
        thresh = skimage.filters.threshold_triangle(img_patch, nbins=self.nbins_spinbox.value())
        print(f'{thresh=}')
        print(f'{img_patch.max()=}')
        binary = img_patch > thresh
        # binary = closing(binary, square(3))
        #cleared = clear_border(binary)
        label_image = label(binary)
        label_image = fill_label_holes(label_image)
        table = regionprops_table(label_image, properties=('label', 'area'))
        condition = table['area'] > self.area_spinbox.value()
        input_label = table['label']
        output_label = input_label * condition
        filtered_label = map_array(label_image, input_label, output_label)
        self.label_layer.data[minr:maxr, minc:maxc] = filtered_label # TODO: affect only labeled pixels
        self.label_layer.refresh()
        self.viewer.layers.remove('label_helper') # TODO: remove potentially multiple layers
        print("Done")

def foo():
    print("Hello World")

def main():
    # Load sample image
    img = tifffile.imread("/Users/jonas/Library/Mobile Documents/com~apple~CloudDocs/FAU/AIMI/PhD/03_Datasets/ouma_macrophages/TIF/CD115_l_Ctrl _Merged Image 12.tiff")

    viewer = napari.view_image(
        img,
        channel_axis=0,
        name=["macrophages", "nuclei"],
        colormap=["gray", "magenta"],
        contrast_limits=[[0, 75], [0, 75]],
    )
    btn = McLabel(viewer)
    #btn = QPushButton('Do stuff')
    #btn.clicked.connect(foo)
    #viewer.window.add_dock_widget(btn, area='left')
    #viewer.window.add_dock_widget(QPushButton('Do stuff'), area='left')
    labels = np.zeros_like(img[0], dtype='int32')
    #viewer.add_labels(labels, name="labels")





    input('Press ENTER to exit')
    pass


if __name__ == "__main__":
    main()