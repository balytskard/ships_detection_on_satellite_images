# Ship Detection On Satellite Images

This document provides a comprehensive overview of the Python code for building and training a segmentation model to detect ships in satellite imagery.

## Data Download and Exploration:

1. **Import libraries**: Necessary libraries for data manipulation, image processing, and model building are imported.
2. **Download dataset**: The opendatasets library is used to download the Airbus Ship Detection dataset from Kaggle using the user's API key.
3. **Import training data**: The training data, including satellite images and ship segmentation information, are loaded from the downloaded folder.
4. **Image size**: A function `get_size` calculates the width and height of a randomly selected image to determine the input dimensions for the model.
5. **Data overview**:
   * The total number of images in the dataframe is compared to the unique number of images in the folder, revealing that some images may have multiple ship detections.
   * The combine_encoded_pixels function groups the dataframe by 'ImageId' and combines 'EncodedPixels' entries with spaces, preparing the data for mask creation.

### Data Preprocessing:

1. **Resize images**: The `resize_images` function resizes all images in a folder to a specified size (256x256 pixels in this case) for consistency in the model input.

### Data Creation for Training:

1. **Create masks**: The `create_masks` function iterates through the training images and their corresponding encoded pixel information to create binary masks representing ship locations.
   * If an image has no ship ('nan' encoded pixels), an empty mask is created.
   * Encoded pixels are decoded into pixel positions and run lengths to generate the mask.
   * Masks are rotated and flipped for data augmentation.
   * Finally, masks are resized to match the image size.

New data is saved with the following structure:

```
data/
|— train
|— masks
```
   

### Image Data Generation:

1. **Data generators**: Separate data generators are created for training and validation images and masks using ImageDataGenerator from Keras.
    * The generators perform data augmentation such as rescaling and random transformations.
2. **Batching**: Training and validation data are divided into batches for efficient model training. 
 
## Segmentation Model Building:

1. **Custom metrics**:
   * `iou_coef` calculates the Intersection over Union (IoU) coefficient, a metric for measuring the overlap between predicted and ground truth masks.
   * `dice_coef` calculates the Dice coefficient, another metric for assessing segmentation performance.
   * `dice_coef_loss` defines the loss function based on the Dice coefficient, which the model aims to minimize during training.
2. **U-Net architecture**: The model utilizes the InceptionResNetV2 pre-trained model as an encoder for feature extraction, followed by a decoder with upsampling and convolutional layers to generate the final segmentation mask.
3. **Model compilation**: The model is compiled with Adam optimizer, the custom dice coefficient loss, and additional metrics like IoU and dice coefficient for evaluation.
   
## Model Training:

1. **Model training**: The model is trained on the prepared training data using the ```fit``` function with specified epochs, batch size, and validation data for performance monitoring.
    * Early stopping and model checkpointing are implemented to prevent overfitting and save the best performing model based on the validation dice coefficient.
2. **Training progress visualization**: The `plot_hist_curves` function visualizes the training and validation loss and dice coefficient curves over epochs to analyze the learning process.

To train the model run the script `python train.py`

## Model Evaluation and Prediction:

1. **Predict on validation data**: Predictions are made on a sample of validation images, and the predicted masks are compared to the ground truth masks for qualitative evaluation.
2. **Visualization**: Predicted masks are visualized alongside the original images and ground truth masks to assess the model's ability to detect ships accurately.
3. **Encoding predictions**: Functions are defined to:
   * Validate the format of the encoded pixel string representing ship segmentation.
   * Encode predicted masks into the run-length encoded format required for submission.
4. **Prediction on test data**: The trained model is used to predict masks on the test data, and the predictions are encoded for submission to the competition.

After training, the models can be used to detect ships on satellite images. The `inference.py` script loads the trained models, processes the input images from the `test` directory, and outputs the predictions. Predictions include information on whether ships are detected on each image, and if so, their location.

Run the inference using: `python inference.py`

>The files from the data/test_images_jpg folder will be used to generate `submission.csv`.
>
>To specify your own directory, go to `inference.py` and change `test` to your directory at the very bottom of the code.
>
>```
>if __name__ == '__main__':
>   ...
>  model.compile(optimizer=Adam(learning_rate=1e-4),
>                  loss=dice_coef_loss,
>                  metrics=[dice_coef, iou_coef])
>
>   test = 'airbus-ship-detection/test_v2' # or path to your directory with images
>   output_file = 'submission.csv'
>   ...
> ```

This will create a `submission.csv` file with predictions and bounding boxes for the detected ships in the format required for the original hackathon.