# sparcnet (SPARrse annotation Classifier  uNET)
* [Pixel Classification](#pixel-classification)
* [Goals](#goals)
* [Step-by-step tutorial](#step-by-step-tutorial)
  * [I. installation conda](#i-installation-conda)
  * [II. Create and activate a new environment](#ii-create-and-activate-a-new-environment)
  * [III. Installation dependencies](#iii-installation-dependencies)
  * [IV. Organizing and preparing data](#iv-organizing-and-preparing-data)
  * [V. Extracting Images from CZI Files](#v-extracting-images-from-czi-files)
  * [VI. Napari Viewer and plugin interface](#vi-napari-viewer-and-plugin-interface)
  * [VII. Sparse annotations and training](#vii-sparse-annotations-and-training)
  * [VIII. Human in the loop like method](#viii-human-in-the-loop-like-method)
  * [IX. Predict entire 3D dataset](#ix-predict-entire-3d-dataset)
* [Tips and Tricks](#tips-and-tricks)
* [Results](#results)
* [Citation](#citation)

Recent Updates
(August 2025)
- Script now detects and prints the number of classes the model was trained to predict.
- You can now select the number of classes to include in the final output file and fully compatible with ImageJ, Napari, and other bioimage tools.
- Improved memory efficiency by writing predicted slices incrementally — no need to load full volumes in RAM.
- One unified script supports both classification and probability

Pixel Classification
------
Pixel classification in biological images involves assigning a class (e.g. vessel, nucleus, background) to each pixel based on its local appearance. This typically starts with manual annotation, where the user draws or colors selected regions (sparse labeling) to provide examples of each class. Tools like **Ilastik** or **Trainable Weka Segmentation** in Fiji, **QuPath** allow biologists to perform such classification using **Random Forest** classifiers. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/f5daa20b-5f1e-4b6c-ab6a-cf5548d1fd7a">
</p>


However, Random Forests have limitations, this motivates the shift toward **deep learning**, like **U-Net**.

Goals
------
This GitHub project demonstrates how to train a **2D U-Net** for semantic segmentation using only a few annotated ROIs across slices from a large 3D light sheet images (Lightsheet Z1 from Zeiss). Labels are sparsely drawn in **Napari**, using the **napari-easy-augment-batch-dl** plugin, and augmented to increase data diversity. The trained model is then applied **slice-by-slice** across the full 3D volume using MONAI’s, and the results are visualized in Napari. While a 3D model might improve performance, this 2D approach balances accuracy and resource efficiency, making it accessible and scalable for large imaging datasets.


Step-by-step tutorial
------
For windows

I. installation conda
------
Install Miniconda [link](https://www.anaconda.com/download/success) and check the [documentation](https://www.anaconda.com/docs/main) for more informations.
During the installation, check the box "Add Anaconda/Miniconda to my PATH environment variable".

II. Create and activate a new environment
------
Start Miniconda prompt and write <br />   `conda create -n sparcnet -c conda-forge python=3.10 -y` <br /> then <br />    `conda activate sparcnet`<br />

III. Installation dependencies
------
`pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124`<br />
`pip install "napari[all]" albumentations matplotlib scipy tifffile czifile notebook`<br />
`pip install ipykernel `<br />
`python -m ipykernel install --user --name sparcnet --display-name "Python (sparcnet)" `<br />
`pip install --upgrade git+https://github.com/Project-MONAI/MONAI`<br />
`pip install --upgrade git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git`<br />
`pip install --upgrade git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git`<br />


When launching Jupyter Notebook, it often opens in a default directory like `C:\Users\YourUsername`. To avoid navigation issues, it's recommended to create your project folder directly in this default location.Ten, download the following scripts, and place them inside that folder. <br /> [vessels_semantic_framework.py](https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/notebooks/imagesc/2025_03_19_vessel_3D_lightsheet/vessels_semantic_framework.py)
<br /> [semantic_helper.py](https://github.com/AlexHego/DL-pixel-classification/blob/main/semantic_helper.py)
<br /> [Extract 2D images and crop script](https://github.com/AlexHego/DL-pixel-classification/blob/main/01_extract%202D%20images%20and%20crop.ipynb)
<br /> [Training script](https://github.com/AlexHego/DL-pixel-classification/blob/main/02_Training_sparse_label.ipynb)
<br /> [Prediction script for 3D data](https://github.com/AlexHego/DL-pixel-classification/blob/main/03_segment_3D.ipynb)

IV. Organizing and preparing data
------
Place your `.czi` image files (with only one channel) in a folder named `data`. Please follow this structure.

```plaintext
parent_path/
├── data/
│   ├── image1.czi
│   ├── image2.czi
│   └── ...
```
Then the script will automatically create two new folders:

- **extracted_tiff**: contains the extracted 2D slices (every 25 planes) from the `.czi` files.
- **cropped_tiff**: contains the final cropped 2D images, resized to ensure the same dimensions in both **x** and **y**.

```plaintext
parent_path/
├── data/
│   ├── image1.czi
│   ├── image2.czi
│   └── ...
├── extracted_tiff/
│   ├── image1_z000.tif
│   ├── image1_z025.tif
│   └── ...
├── cropped_tiff/
│   ├── image1_z000.tif
│   ├── image1_z025.tif
│   └── ...
```
V. Extracting Images from CZI Files
------

Before launching Jupyter Notebook, ensure that the conda environment is activated. If it isn’t, open the Miniconda Prompt and run: <br />  `conda activate vessels_lightsheet`. <br />  Once the environment is activated, start Jupyter Notebook by typing: `jupyter notebook`.
<br /> Then, navigate to the folder where your scripts are saved and open the notebook titled "Extract 2D images and crop". This notebook crops the images to ensure they all have the same dimensions along the x and y axes. <br /> **Note** : You will have to change the parent_path

VI. Napari Viewer and plugin interface 
------
Open the notebook **`02_Training_sparse_label.ipynb`** and run the script. This will launch **Napari**, where you will draw sparse annotations and begin the training process.

### Important: "Number of Classes" Dialog

When prompted with the **"Number of classes"** dialog box, **leave it set to `1`**, even if you are using multiple pixel classes (e.g., background, class 1, class 2).

<img width="347" height="152" alt="image" src="https://github.com/user-attachments/assets/42a27380-ba09-42d1-b291-8ebcda5748da" />


## Napari Viewer tutorial:
<img width="250" alt="image" src="https://github.com/user-attachments/assets/52abe2f8-8800-4d12-b399-0c780544ef0e" align="right">

link to the [documentation](https://napari.org/dev/tutorials/fundamentals/viewer.html)
<br />

**Left Panel – Image Viewer & Layer Controls**
---
Used to adjust display settings of the selected layer:

- **Opacity**: Controls layer transparency.
- **Blending**: Current mode: `translucent_no_depth`.
- **Contrast Limits**: Adjusts brightness range.
- **Auto-contrast**:  
  - `once`: Set once when image is loaded.  
  - `continuous`: Updates automatically.
- **Gamma**: Controls image intensity midtones.
- **Colormap**: Current colormap is `gray`.
- **Interpolation**: Rendering mode (`nearest` selected).

**Layer List**
---
This section lists all visible layers in the napari viewer:

- `Label box`: Manual box. Where IA check label               
- `predictions_0`: Model-predicted semantic segmentation.    
- `labels_0`: Manual ground truth labels.                  
- `images`: Raw input image currently viewed.              

Note : Use the eye icon to toggle visibility for each layer.

---
##  Right Panel – Plugin Functionalities
<img width="250" alt="image" src="https://github.com/user-attachments/assets/62b3298d-7dd0-4372-9d00-bb1b8d0a58b8" align="right">

**1. Draw Labels**
- `Open image directory…`: Load raw image dataset.
- `Save results…`: Save drawing and prediction

**2. Augment Images**
all the augmentation can be use and check
- `Augment current image`: Apply only to the open image.
- `Augment all images`: Batch augment all loaded images.
- `Delete augmentations`: Remove generated augmented patches.
- `Settings…`: Open advanced augmentation options.

**3. Train / Predict**
Interface to train and use a deep learning segmentation model.

- **Model type**: `Vessels Semantic Model`
- `Load`: Load a pre-trained model.
- `Train network`: Start model training on labeled data.
- `Predict current image`: Run inference on the visible image.
- `Predict all images`: Apply prediction to all images in batch.

VII. Sparse annotations and training
------
##  Useful Napari Keyboard Shortcuts

Here are some helpful keyboard and mouse shortcuts to make navigating and annotating in Napari easier:

| Action                                      | Shortcut                           |
|--------------------------------------------|------------------------------------|
|  Move the image                          | Left-click + drag                  |
| Zoom in/out                              | Mouse scroll wheel                 |
|  Increase brush size                      | `Alt` + drag right                 |
|  Decrease brush size                      | `Alt` + drag left                  |
|  Navigate between layers                  | `Up` / `Down` arrows or `Ctrl` + `←`/`→` |
|  Show/hide a layer                        | Click the eye icon next to the layer |
|  Delete a shape (e.g., rectangle)         | Select shape + `Delete` key        |
| Reset the view                           | Press `R`                          |
| Activate brush tool                      | Press `B`                          |
| Activate fill tool (paint bucket)        | Press `F`                          |

## To begin annotating, follow these steps:

1. **Select the `Label box` layer** in the Napari viewer.  **Important**: Make sure the label box you draw is **larger than the patch size** (which is set to `256` by default).  
2. Use the shape tool (rectangle icon) to **draw a box** in the image. This box defines the region where the AI will look for your annotations.
3. After drawing the box, switch to the `labels_0` layer to **draw sparse annotations** inside the selected area.

This method uses **sparse labeling**, meaning you do **not** need to label every pixel. However, you **must label some background pixels** to help the model differentiate between background and unlabeled areas.

### Labeling Guide

If you're working with two foreground classes, use the following labels:

- `1` → Background  
- `2` → Class 1  
- `3` → Class 2  

Any pixels labeled as `0` are treated as **unlabeled** and will be **ignored during training**.

##  Training setup
<img width="250" alt="image" src="https://github.com/user-attachments/assets/940d14f4-2de5-43cc-86ad-0f31244ffa1e" align="right">

- **`depth`**  
  Number of layers in the encoder. Higher values increase model capacity to see larger area but also training time.  
 
- **`features_level_1`**  
  Number of feature channels in the first layer. Each subsequent layer doubles this value.  
  *Example:* If set to `32`, the encoder feature channels will be: `32 → 64 → 128 → 256 → 512`.

- **`weight_c1`, `weight_c2`, `weight_c3`**  
  Class weights used in the loss function for class balancing:
  - `weight_c1` → Background  
  - `weight_c2` → Class 1  
  - `weight_c3` → Class 2 (e.g., vessels)  
  *Tip:* Increase the weight for the **vessel class** to emphasize learning on vessels.

- **`num_epochs`**  
  Total number of training epochs. Higher values allow longer learning but take more time.

- **`learning_rate`**  
  Learning rate for the optimizer. A lower value like `0.0001` is usually safer and more stable.

- **`dropout`**  
  Dropout rate for regularization. Helps prevent overfitting. Recommended values are between `0.1` and `0.5`.

- **`model_name`**  
  The filename for saving the trained model (e.g., `vessel_segmentation.pth`).


##  Annotation, augmentation and training example

![part1](https://github.com/user-attachments/assets/0f23247f-d3be-4243-92a9-76d0909654c3)


VIII. Human in the loop like method
------
## 🔄 Human-in-the-Loop Training Loop

Once you have created your initial sparse annotations and trained the first version of the model, you can start a **human-in-the-loop method** to iteratively improve results with minimal effort.

### Loop Steps 

1. **Predict on a New Image**  
   Open a new image in Napari, click **"Predict current image"** using your trained model and draw a `Label box`.

2. **Convert Prediction to Labels**  
   After the label box, Napari will prompt: <br />
   <img width="412" height="151" alt="image" src="https://github.com/user-attachments/assets/ac769147-d219-4bb9-9f53-ed11310aa6f7" /> <br />
   Click **Yes**. This will create editable labels based on the model's prediction.

4. **Correct the Prediction**  
   - Use the `labels_0` layer to manually correct any mistakes (e.g., missing vessels, false positives).

5. **Augment and Retrain**  
   - Apply augmentations again and train the model to incorporate the newly corrected data.

6. **Repeat**  
   Continue this loop with additional images:
   - Predict → Convert → Correct → Retrain  
   to build a more accurate and generalizable model over time.

This approach drastically reduces manual annotation time while steadily improving model performance.  


IX. Predict entire 3D dataset
------
The prediction step for an entire 3D dataset is **performed outside of Napari**, so you can safely close the application once annotations and training are complete.

To run inference, use the script :

- **03_segment_3D.ipynb** – Generate **classification** or **probabilities** (segmentation output):  

**Note**: Use class label prediction for typical segmentation tasks, and probability maps if you want more flexibility or uncertainty-aware post-processing.


Tips and Tricks
------
Here are some optional strategies and considerations to help you get the most out of your training process:

### 1. Downsample the Image (Optional)

Downsampling your input images can significantly increase training speed and effectively expand the model’s **real-world receptive field**.  

- For example, if your network has a receptive field of `50×50` pixels, downsampling the image will make that field cover a larger area of the original image.
- This can help the model capture **large-scale structures** like major vessels or tissue architecture more effectively **BUT** you may lose finer details such as **small vessels**.

### 2.  Class Weights – Use with Care

During early training iterations with **sparse annotations**, using **class weights** is helpful to:
- Emphasize underrepresented or important classes (e.g., vessels).
- Help the model focus on sparse labels that might otherwise be overwhelmed by background.

However, after a few training cycles — once you’ve annotated a **larger amount of data** — continuing to use high class weights can become **counterproductive**:

- The model may **overfit** to those classes.
- It can start introducing **bias**


Results
------

Below are some example of results. 

### Before and after
<p align="center">
<img alt="brainrotation" src="https://github.com/user-attachments/assets/fc2ff58d-453d-48d7-9f75-c342ab546567" height="600" />
</p>


Citation
------
This project builds on the work by Brian Northan [True North Intelligent Algorithms](https://github.com/True-North-Intelligent-Algorithms). I reuse and adapt parts of his code and ideas to demonstrate semantic segmentation workflows for my lightsheet imaging.

If you use it successfully for your research please be so kind to cite theses Github: [3D Vessel Segmentation using 2D U-Net](https://github.com/True-North-Intelligent-Algorithms/tnia-python/tree/main/notebooks/imagesc/2025_03_19_vessel_3D_lightsheet)
</br> and this github if you use my script [sparcnet](https://github.com/AlexHego/DL-pixel-classification)



