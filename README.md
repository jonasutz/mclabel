# McLabel Instructions

## Installation

The plugin can be installed using the following command:

```bash
pip install git+https://gitlab.cs.fau.de/xo04syge/mclabel.git
```

Required packages are installed automatically. After successfull installation the plugin will appear in the plugins menu of napari 

<img src="/Users/jonas/Library/Application Support/typora-user-images/image-20221125163916278.png" alt="image-20221125163916278" style="zoom:75%;" />

## Usage

After starting the plugin, using the Plugins menu, two new buttons appear on the side bar:

![image-20221125164105642](/Users/jonas/Library/Application Support/typora-user-images/image-20221125164105642.png)

### 1. Loading an Image

Using drag and drop a 3D imaris file can be placed on the main window to load the image into napari. Accordingly, the individual channels of the image appear in the layer list (left panel). 

**Now all layers with macrophage infomation (e.g. cytoplasm) must be selected.** Multiple layers can be selected with the `Ctrl`-key. 

![image-20221125164554682](/Users/jonas/Library/Application Support/typora-user-images/image-20221125164554682.png)

For this example image we choose only `Channel 0`, since `Channel 1` contains the nuclei (which are not required). 

**To start labeling press the button `Convert Images`**. 

The image is preprocessed and the output looks like this:

![image-20221125164832840](/Users/jonas/Library/Application Support/typora-user-images/image-20221125164832840.png)

### 2. Drawing around Macrophages

Now the Plugin is in 'DRAW' mode

Draw around a macrophage using the mouse:

![image-20221125172049834](/Users/jonas/Library/Application Support/typora-user-images/image-20221125172049834.png)

Press **`Compute Label`** to obtain a segmentation of the macrophage. 

### 3. Adjusting the Output of McLabel

#### 3.1 Global adjust of Threshold

By moving the slider: 
![image-20221125172231018](/Users/jonas/Library/Application Support/typora-user-images/image-20221125172231018.png)

the global threshold can be adjusted if automatic computation failed. 

#### 3.2 Refinement of structures with the brush

In some cases we are happy with the global threshold, but small structures were not captured by the semi-automatic process (see image for reference).
<img src="/Users/jonas/Library/Application Support/typora-user-images/image-20221128133156353.png" alt="image-20221128133156353" style="zoom:50%;" />

In this case we can add more pixels:

1. Select the layer `OutputLabel` from the layer list at the left pane
2. Choose the picker from the layer controls:
   ![image-20221128133351704](/Users/jonas/Library/Application Support/typora-user-images/image-20221128133351704.png)
3. Click inside the label that needs adjustment
4. Select the brush for adding or the eraser for removing pixels
   ![image-20221128133501238](/Users/jonas/Library/Application Support/typora-user-images/image-20221128133501238.png)

5. Label parts of the macrophage with the brush 

### 4. Saving the labels

When the OutputLabel layer is selected from the layer list, select File -> Save Selected Layer 

