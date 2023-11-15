import numpy as np
import napari
from scipy import ndimage
from qtpy.QtWidgets import QPushButton, QLabel, QSlider, QWidget, QCheckBox, QComboBox
from qtpy.QtCore import *
from qtpy import QtCore
import skimage.filters
from skimage.measure import label, regionprops, regionprops_table
from napari_mclabel.utils import fill_label_holes
from skimage.util import map_array
from enum import Enum
from napari_plugin_engine import napari_hook_implementation
from dataclasses import dataclass


class State(Enum):
    DRAW = 1
    COMPUTE = 2
    NO_INIT = 3


class ImageType(Enum):
    SC_2D = 1  # Single Channel 2D (Y,X)
    MCF_2D = 2  # Multi Channel First 2D (C,Y,X)
    MCL_2D = 3  # Multi Channel Last 2D (Y,X,C) (RGB,RGBA)
    SC_3D = 4  # Single Channel 3D (Z,Y,X)
    MC_3D = 5  # Multi Channel 3D (C,Z,Y,X)
    UNKNOWN = -1


@dataclass
class ImageShapeHelper:
    img_type: ImageType
    lbl_helper_shape: tuple
    patch_slice: tuple


def on_label_change(event):
    # old_label = event.old_label  # the old label
    # new_label = event.new_label  # the new label
    #
    # if new_label == old_label + 1:  # check if the label increased by 1
    #     print("Label increased by 1")
    # else:
    #     print("Label changed but didn't increase by 1")
    # print(vars(event._sources[0]))
    pass


class McLabel(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.layers.events.inserted.connect(self.init_layers)

        self.label_helper = None
        self.image_layer = None
        self.label_layer = None

        self.layer_types = {"image": napari.layers.image.image.Image, "labels": napari.layers.labels.labels.Labels}

        # draw_button
        self.draw_compute_btn = QPushButton('Draw Label')
        self.draw_compute_btn.clicked.connect(self.btn_click)

        # List of layers to (select layer to segment)
        self.layer_selection_lbl = QLabel("Select layer to segment:")
        self.layer_selection_cb = QComboBox()
        self.layer_selection_cb.addItems(self.get_layer_names("image"))
        self.layer_selection_cb.currentTextChanged.connect(self.on_image_change)

        self.algo_selection_lbl = QLabel("Select algo to segment:")
        self.algo_selection_cb = QComboBox()
        self.algo_selection_cb.addItems(self.get_supported_algorithms())
        self.algo_selection_cb.currentTextChanged.connect(self.on_algo_change)
        self.algo_selection_cb.setCurrentText("triangle")
        self.algo = skimage.filters.threshold_triangle

        # In future we might add more comboboxes, e.g., for label layers.
        # Therefore we store all comboboxes in a list of dicts
        self.comboboxes = [{"combobox": self.layer_selection_cb, "layer_type": "image"}]
        # {"combobox": self.cb_label_layers, "layer_type": "labels"}]
        self.init_comboboxes()

        # Slider for manual adjustment of threshold
        self.threshold_slider_lbl = QLabel("Threshold: ")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setVisible(False)
        self.threshold_slider_lbl.setVisible(False)
        self.threshold_slider.setRange(0, 65000)  # TODO: determine range from selected image
        self.threshold_slider.setPageStep(1)
        self.threshold_slider.setValue(15)
        self.threshold_slider.valueChanged.connect(self.manual_threshold_adjustment)

        self.viewer.window.add_dock_widget(
            [self.draw_compute_btn, self.layer_selection_lbl, self.layer_selection_cb, self.algo_selection_lbl,
             self.algo_selection_cb,
             self.threshold_slider_lbl, self.threshold_slider],
            area='right',
            name='McLabel')

        self.current_max_lbl = 0
        self.draw_compute_btn.setEnabled(True)

        self.state = State.NO_INIT
        self.img_type = ImageType.UNKNOWN

        self.normalize = False

    def init_layers(self, event):
        if len(self.viewer.layers) == 1:
            # If an image was added we can now allow labelling (i.e. enable the button)
            self.draw_compute_btn.setEnabled(True)

    # def preproc_fn(self):
    #     if isinstance(self.viewer.layers[0].data, napari.layers._multiscale_data.MultiScaleData):
    #         # user manually selects all layers which shall be considered for MIP
    #         num_layers_selected = len(self.viewer.layers.selection)
    #         assert num_layers_selected > 0, "At least one layer must be selected"
    #
    #         singleton = self.viewer.layers[0].data.shape[
    #                         0] == 1  # sometimes the ims loader returns the array with singleton
    #
    #         if singleton:  # maybe we do not need this
    #             _, z, y, x = self.viewer.layers[0].data[0].shape
    #         else:
    #             z, y, x = self.viewer.layers[0].data[0].shape
    #
    #         max_intensity_img = np.zeros((2, y, x))
    #         # TODO:
    #         # merge max value from all selected layers into one channel
    #         # channels that are not selected will remain as distinct channels
    #         for selected in self.viewer.layers.selection:
    #             sel = np.asarray(selected.data[0])[0:self.viewer.layers[0].data[0].shape[0]]
    #             current_mip = np.amax(sel, axis=0)
    #             np.copyto(max_intensity_img[0], current_mip, where=current_mip > max_intensity_img[0])
    #
    #         probably_nuc_channel = np.asarray(
    #             [l for l in self.viewer.layers if l not in self.viewer.layers.selection][0].data[0].copy())[
    #                                0:self.viewer.layers[0].data[0].shape[0]]
    #         max_intensity_img[1] = np.amax(probably_nuc_channel, axis=0)
    #
    #     else:  # no ims file, image presumed to be already processed
    #         max_intensity_img = self.viewer.layers[0].data.copy()
    #
    #     # self.viewer.layers.clear()
    #     # self.viewer.add_image(max_intensity_img,
    #     #                       channel_axis=0,
    #     #                       name=["macrophages", "nuclei"],
    #     #                       colormap=["gray", "magenta"],
    #     #                       contrast_limits=[[0, 75], [0, 75]],
    #     #                       )
    #
    #     self.image_layer = self.viewer.layers[self.layer_selection_cb.currentText()]
    #
    #     # if self.image_layer.ndim == 3 and not self.image_layer.rgb: # if 3d image:
    #     # 3d image 2d layer: np.zeros(self.image_layer.data.shape[self.viewer.dims.ndim - self.viewer.dims.ndisplay:], dtype='int32')
    #     self.label_layer = self.viewer.add_labels(
    #         np.zeros(self.image_layer.data.shape, dtype='int32'),
    #         name='OutputLabel')
    #
    #     self.label_layer.events.selected_label.connect(on_label_change)
    #     # self.label_layer.selected_label = 0
    #     self.viewer.reset_view()
    #     self.threshold_slider.setRange(0, int(self.image_layer.data.max() // 2))
    #     self.preproc_button.setVisible(False)
    #     self.draw_compute_btn.setText("Compute Label")
    #     self.draw_fn()

    def determine_image_type(self):
        """
        Determine image type based on shape
        :return:
        """
        image_shape = self.image_layer.data.shape
        if len(image_shape) >= 4:
            self.img_type = ImageType.MC_3D
            return
        if len(image_shape) == 3:
            if image_shape[-1] in (3, 4):
                # 2D RGB(A): (Y,X,3) or (Y,X,4)
                self.img_type = ImageType.MCL_2D
                return
            elif image_shape[-1] == 1:
                # 2D single channel with singleton (Y,X,1)
                self.img_type = ImageType.SC_2D
                return
            else:
                # TODO: this could be also ImageType.MCF_2D
                # 3D single channel: (Z,Y,X)
                self.img_type = ImageType.SC_3D
                return
        if len(image_shape) == 2:
            # 2D single channel: (Y,X)
            self.img_type = ImageType.SC_2D
            return
        self.img_type = ImageType.UNKNOWN
        raise NotImplementedError  # TODO!

    def btn_click(self):
        if self.state == State.DRAW:
            self.draw_compute_btn.setText("Draw Label")
            self.compute_fn()
        elif self.state == State.COMPUTE or self.state == State.NO_INIT:
            self.draw_compute_btn.setText("Compute Label")
            self.draw_fn()

    def draw_fn(self):
        if self.state == State.NO_INIT:
            self.image_layer = self.viewer.layers[self.layer_selection_cb.currentText()]
            if self.image_layer.data.shape[-1] in (3, 4):  # cheap heuristic for RGB(A)
                self.label_layer = self.viewer.add_labels(np.zeros(self.image_layer.data.shape[0:2], dtype='int32'),
                                                          name="Output Label")
            else:
                self.label_layer = self.viewer.add_labels(
                    np.zeros(self.image_layer.data.shape, dtype='int32'),
                    name='Output Label')

            self.label_layer.events.selected_label.connect(on_label_change)
            self.threshold_slider.setRange(0, int(self.image_layer.data.max() // 2))
            self.draw_compute_btn.setText("Compute Label")
        self.state = State.DRAW

        image_shape = self.image_layer.data.shape
        if len(image_shape) >= 4:
            # 3D with channels: (C,Z,Y,X)
            labels = np.zeros(self.image_layer.data.shape[self.viewer.dims.ndim - self.viewer.dims.ndisplay:],
                              dtype='int32')  # label helper is always 2D
            self.img_type = ImageType.MC_3D
        elif len(image_shape) == 3:
            if image_shape[-1] in (3, 4):
                # 2D RGB(A): (Y,X,3) or (Y,X,4)
                labels = np.zeros((image_shape[0], image_shape[1]), dtype='int32')
                # self.image_layer.data = self.image_layer.data[:, :, 0].copy()
                # self.image_layer.refresh()
                self.img_type = ImageType.MCL_2D
            elif image_shape[-1] == 1:
                # 2D single channel with singleton (Y,X,1)
                labels = np.zeros((image_shape[0], image_shape[1]), dtype='int32')
                self.img_type = ImageType.SC_2D
            else:
                # TODO: this could be also ImageType.MCF_2D
                # 3D single channel: (Z,Y,X)
                labels = np.zeros((image_shape[1], image_shape[2]), dtype='int32')
                self.img_type = ImageType.SC_3D
        elif len(image_shape) == 2:
            # 2D single channel: (Y,X)
            labels = np.zeros((image_shape[0], image_shape[1]), dtype='int32')
            self.img_type = ImageType.SC_2D
        else:
            # No support for other shapes at the moment
            self.img_type = ImageType.UNKNOWN
            raise NotImplementedError
        self.label_helper = self.viewer.add_labels(labels, name='_label_helper')
        self.label_helper.brush_size = 17
        self.viewer.layers[0].selected = False
        self.viewer.layers['_label_helper'].selected = True

        self.label_helper.mode = "PAINT"

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
        if self.img_type in (ImageType.SC_3D, ImageType.SC_3D):
            label_patch = self.label_layer.data[self.viewer.dims.current_step[0], minr:maxr, minc:maxc].copy()
        elif self.img_type == ImageType.MC_3D:
            label_patch = self.label_layer.data[self.viewer.dims.current_step[0], self.viewer.dims.current_step[1],
                          minr:maxr, minc:maxc].copy()
        # elif self.img_type == ImageType.MCL_2D:
        #     label_patch = self.label_layer.data[minr:maxr, minc:maxc, 0].copy()
        else:
            label_patch = self.label_layer.data[minr:maxr, minc:maxc].copy()
        out_patch_1 = np.where(label_patch == 0, filtered_label, label_patch)

        if self.img_type in (ImageType.SC_3D, ImageType.SC_3D):
            self.label_layer.data[self.viewer.dims.current_step[0], minr:maxr, minc:maxc] = out_patch_1
        elif self.img_type == ImageType.MC_3D:
            self.label_layer.data[self.viewer.dims.current_step[0], self.viewer.dims.current_step[1], minr:maxr,
            minc:maxc] = out_patch_1
        # elif self.img_type == ImageType.MCL_2D:
        #     self.label_layer.data[minr:maxr, minc:maxc, 0] = out_patch_1
        else:
            self.label_layer.data[minr:maxr, minc:maxc] = out_patch_1

        self.label_layer.refresh()

        self.remove_helper()

        self.threshold_slider_lbl.setVisible(True)
        self.threshold_slider.setVisible(True)

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
        pass

    def apply_filter(self, lbl_img, condition='area'):
        table = regionprops_table(lbl_img, properties=('label', 'area'))
        filt = table[condition] == table['area'].max()
        input_label = table['label']
        output_label = input_label * filt
        filtered_label = map_array(lbl_img, input_label, output_label)
        return filtered_label.astype('int32')

    def get_patch_from_layer(self):
        labeled_macro = self.label_helper.data.copy()
        labeled_macro = ndimage.binary_fill_holes(labeled_macro).astype('int32')
        props = regionprops(labeled_macro)

        for prop in props:
            minr, minc, maxr, maxc = prop.bbox

        if self.img_type == ImageType.SC_2D:
            img_patch = self.image_layer.data[minr:maxr, minc:maxc].copy()
            img_patch[
                labeled_macro[minr:maxr, minc:maxc] == 0] = 0  # removes parts outside hand-drawn region
        if self.img_type == ImageType.SC_3D:
            img_patch = self.image_layer.data[
                        self.viewer.dims.current_step[0],  # z
                        minr:maxr,  # y
                        minc:maxc].copy()  # x
            img_patch[labeled_macro[minr:maxr, minc:maxc] == 0] = 0
        if self.img_type == ImageType.MC_3D:
            img_patch = self.image_layer.data[
                        self.viewer.dims.current_step[0],  # C
                        self.viewer.dims.current_step[1],  # Z
                        minr:maxr,
                        minc:maxc
                        ].copy()
            img_patch[labeled_macro[minr:maxr, minc:maxc] == 0] = 0
        if self.img_type == ImageType.MCL_2D:
            img_patch = self.image_layer.data[minr:maxr, minc:maxc, 0].copy()
            img_patch[
                labeled_macro[minr:maxr, minc:maxc] == 0] = 0

        return img_patch, (minr, minc, maxr, maxc)

    def compute_label_from_patch(self, img_patch, thresh=None, min_area=None):
        if thresh is None:
            # thresh = skimage.filters.threshold_triangle(img_patch, nbins=32)
            thresh = self.algo(img_patch, nbins=32)
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
        #label_patch = self.label_layer.data[self.viewer.dims.current_step[0], minr:maxr, minc:maxc].copy()
        if self.img_type in (ImageType.SC_3D, ImageType.SC_3D):
            label_patch = self.label_layer.data[self.viewer.dims.current_step[0], minr:maxr, minc:maxc].copy()
        elif self.img_type == ImageType.MC_3D:
            label_patch = self.label_layer.data[self.viewer.dims.current_step[0], self.viewer.dims.current_step[1],
                          minr:maxr, minc:maxc].copy()
        # elif self.img_type == ImageType.MCL_2D:
        #     label_patch = self.label_layer.data[minr:maxr, minc:maxc, 0].copy()
        else:
            label_patch = self.label_layer.data[minr:maxr, minc:maxc].copy()
        # make sure that we keep original label id when changing threshold
        if label_patch.max():
            # filtered_label[filtered_label != 0] = label_patch.max()
            filtered_label[filtered_label != 0] = self.current_max_lbl

        out_patch = label_patch.copy()
        np.copyto(out_patch, filtered_label, where=label_patch == 0)
        np.copyto(out_patch, filtered_label, where=label_patch == self.current_max_lbl)
        #self.label_layer.data[self.viewer.dims.current_step[0], minr:maxr, minc:maxc] = out_patch  # TODO!
        if self.img_type in (ImageType.SC_3D, ImageType.SC_3D):
            self.label_layer.data[self.viewer.dims.current_step[0], minr:maxr, minc:maxc] = out_patch
        elif self.img_type == ImageType.MC_3D:
            self.label_layer.data[self.viewer.dims.current_step[0], self.viewer.dims.current_step[1], minr:maxr,
            minc:maxc] = out_patch
        # elif self.img_type == ImageType.MCL_2D:
        #     self.label_layer.data[minr:maxr, minc:maxc, 0] = out_patch_1
        else:
            self.label_layer.data[minr:maxr, minc:maxc] = out_patch
        self.label_layer.refresh()

    def on_image_change(self):
        # self.image_layer = self.viewer.layers[self.layer_selection_cb.currentText()]
        pass

    def on_algo_change(self):
        if not self.algo_selection_cb.currentText() == "SAM":
            self.algo = getattr(skimage.filters, "threshold_" + self.algo_selection_cb.currentText())
        else:
            self.algo = "SAM"

    def init_comboboxes(self):
        for combobox_dict in self.comboboxes:
            # If current active layer is of the same type of layer that the combobox accepts then set it as selected layer in the combobox.
            active_layer = self.viewer.layers.selection.active
            if combobox_dict["layer_type"] == "all" or isinstance(active_layer,
                                                                  self.layer_types[combobox_dict["layer_type"]]):
                index = combobox_dict["combobox"].findText(active_layer.name, QtCore.Qt.MatchFixedString)
                if index >= 0:
                    combobox_dict["combobox"].setCurrentIndex(index)

        # Inform all comboboxes on layer changes with the viewer.layer_change event
        self.viewer.events.layers_change.connect(self._on_layers_changed)

        # Register an event to all existing layers
        for layer_name in self.get_layer_names():
            layer = self.viewer.layers[layer_name]

            @layer.events.name.connect
            def _on_rename(name_event):
                self._on_layers_changed()

        # Register an event to all layers that will be created
        @self.viewer.layers.events.inserted.connect
        def _on_insert(event):
            layer = event.value

            @layer.events.name.connect
            def _on_rename(name_event):
                self._on_layers_changed()

        self._init_comboboxes_callback()

    def _on_layers_changed(self):
        for combobox_dict in self.comboboxes:
            layer = combobox_dict["combobox"].currentText()
            layers = self.get_layer_names(combobox_dict["layer_type"])
            combobox_dict["combobox"].clear()
            combobox_dict["combobox"].addItems(layers)
            index = combobox_dict["combobox"].findText(layer, QtCore.Qt.MatchFixedString)
            if index >= 0:
                combobox_dict["combobox"].setCurrentIndex(index)
        self._on_layers_changed_callback()

    def get_layer_names(self, type="all", exclude_hidden=True):
        layers = self.viewer.layers
        filtered_layers = []
        for layer in layers:
            if (type == "all" or isinstance(layer, self.layer_types[type])) and (
                    (not exclude_hidden) or (exclude_hidden and "<hidden>" not in layer.name)):
                filtered_layers.append(layer.name)
        return filtered_layers

    def get_supported_algorithms(self):
        algos = [func_name.split('_')[-1] for func_name in dir(skimage.filters) if func_name.startswith('threshold_')]
        return algos

    def _init_comboboxes_callback(self):
        self.on_image_change()

    def _on_layers_changed_callback(self):
        pass


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
