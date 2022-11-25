import napari

from qtpy.QtWidgets import QPushButton, QMainWindow, QCheckBox, QSpinBox, QLabel, QSlider, QShortcut
from qtpy.QtCore import *
from qtpy.QtGui import QKeySequence


class McExperiment(QMainWindow):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.layers.events.inserted.connect(self.print_layer_name)
        # start_button
        self.start_button = QPushButton('Draw new Label')
        #self.start_button.clicked.connect(self.select_macrophage)
        # self.viewer.window.add_dock_widget(self.start_button, area='left')
        # finish buttono
        self.finish_button = QPushButton('Compute Label')
        #self.finish_button.clicked.connect(self.finish_fn)

        self.compute_label_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_P), self)
        #self.compute_label_shortcut.activated.connect(self.finish_fn)

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
        #self.area_slider.valueChanged.connect(self.manual_area_adjustment)

        # Slider for manual adjustment of threshold
        self.threshold_slider_lbl = QLabel("Threshold: ")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setVisible(False)
        self.threshold_slider_lbl.setVisible(False)
        self.threshold_slider.setRange(0, 50)
        self.threshold_slider.setPageStep(1)
        self.threshold_slider.setValue(15)
        #self.threshold_slider.valueChanged.connect(self.manual_threshold_adjustment)

        self.viewer.window.add_dock_widget(
            [self.start_button, self.finish_button, self.preserve_checkbox, self.area_slider_lbl, self.area_slider,
             self.threshold_slider_lbl, self.threshold_slider], area='left', name='McLabel')

        self.current_max_lbl = 0
        self.finish_button.setEnabled(False)

    def print_layer_name(self, event):
        if len(self.viewer.layers) == 1:
            print(f"Layer added")
            self.viewer.layers.remove(0)
            print(f"Layer removed")


def main():
    # Load sample image
    viewer = napari.Viewer()


    win = McExperiment(viewer)
    input('Press ENTER to exit')


if __name__ == "__main__":
    main()