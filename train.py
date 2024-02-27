import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionResNetV2


IMG_SIZE = (256, 256, 3)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 5
DATA = 'data'


def data_generator(data, img_size):
    """Builds data generators for training and validation from a directory.

    Args:
      data: str, path to the directory containing images and masks.
      img_size: tuple(int, int), desired target size (height, width) for images
        and masks.

    Returns:
      tuple(ImageDataGenerator, ImageDataGenerator), a tuple containing two
        ImageDataGenerator objects, the first for training and the second for
        validation.
    """
    image_datagen = ImageDataGenerator(rescale=1. / 255,
                                       validation_split=0.15)

    mask_datagen = ImageDataGenerator(rescale=1. / 255,
                                      validation_split=0.15)

    train_images = image_datagen.flow_from_directory(
        data,
        target_size=img_size[0:2],
        class_mode=None,
        classes=['train'],
        batch_size=BATCH_SIZE,
        seed=SEED,
        subset='training')

    train_masks = mask_datagen.flow_from_directory(
        data,
        target_size=img_size[0:2],
        class_mode=None,
        classes=['masks'],
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        seed=SEED,
        subset='training')

    val_images = image_datagen.flow_from_directory(
        data,
        target_size=img_size[0:2],
        class_mode=None,
        classes=['train'],
        batch_size=BATCH_SIZE,
        seed=SEED,
        subset='validation')

    val_masks = mask_datagen.flow_from_directory(
        data,
        target_size=img_size[0:2],
        class_mode=None,
        classes=['masks'],
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        seed=SEED,
        subset='validation')

    train_generator = zip(train_images, train_masks)
    val_generator = zip(val_images, val_masks)

    return train_generator, val_generator


def calculate_step_sizes(train, val, batch_size):
    """Calculates the number of steps per epoch for training and validation generators.

    Args:
      train: tf.keras.utils.Sequence, training data generator.
      val: tf.keras.utils.Sequence, validation data generator.
      batch_size: int, batch size used for training and validation.

    Returns:
      tuple(int, int), a tuple containing the number of steps per epoch for
        training and validation, respectively.
    """
    step_size_train = train.samples // batch_size
    step_size_valid = val.samples // batch_size

    return step_size_train, step_size_valid


def iou_coef(y_true, y_pred, smooth=1):
    """Calculates the Intersection over Union (IoU) coefficient between two sets of labels.

    Args:
        y_true: A tensor of shape (..., N) representing the ground truth labels.
          Values should be 0 or 1.
        y_pred: A tensor of shape (..., N) representing the predicted labels.
          Values should be 0 or 1.
        smooth: A small constant value for numerical stability, default to 1.

    Returns:
        iou: A scalar tensor representing the mean IoU coefficient across all classes.
    """
    intersection = keras.sum(keras.abs(y_true * y_pred), axis=[1, 2, 3])
    union = keras.sum(y_true, [1, 2, 3]) + keras.sum(y_pred, [1, 2, 3]) - intersection
    iou = keras.mean((intersection + smooth) / (union + smooth), axis=0)

    return iou


def dice_coef(y_true, y_pred):
    """Calculates the Dice coefficient between two sets of labels.

    Args:
        y_true: A tensor of shape (..., N) representing the ground truth labels.
          Values should be 0 or 1.
        y_pred: A tensor of shape (..., N) representing the predicted labels.
          Values should be 0 or 1.

    Returns:
        A scalar tensor representing the Dice coefficient.
    """
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

    return dice


def dice_coef_loss(y_true, y_pred):
    """Calculates the Dice coefficient loss.

    Args:
        y_true: A tensor of shape (..., N) representing the ground truth labels.
          Values should be 0 or 1.
        y_pred: A tensor of shape (..., N) representing the predicted labels.
          Values should be 0 or 1.

    Returns:
        A scalar tensor representing the Dice coefficient loss.
    """
    return 1 - dice_coef(y_true, y_pred)


def conv_block(input, num_filters):
    """Create a convolutional block consisting of two convolutional layers,
    each followed by Batch Normalization and ReLU activation.

    Args:
        input: The input tensor.
        num_filters: The number of filters for the convolutional layers.

    Returns:
        The output tensor of the convolutional block.
    """
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    """Create a decoder block consisting of a transposed convolution layer,
    concatenation with the corresponding skip features, and a convolutional block.

    Args:
        input: The input tensor.
        skip_features: The skip features from the encoder.
        num_filters: The number of filters for the convolutional layers.

    Returns:
        The output tensor of the decoder block.
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_model(input_shape):
    """Build an Inception-ResNetV2 U-Net model for image segmentation.

    Args:
        input_shape: The shape of the input image.

    Returns:
        The Inception-ResNetV2 U-Net model.
    """
    inputs = Input(input_shape)

    encoder = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)

    s1 = encoder.get_layer("input_1").output

    s2 = encoder.get_layer("activation").output
    s2 = ZeroPadding2D(((1, 0), (1, 0)))(s2)

    s3 = encoder.get_layer("activation_3").output
    s3 = ZeroPadding2D((1, 1))(s3)

    s4 = encoder.get_layer("activation_74").output
    s4 = ZeroPadding2D(((2, 1), (2, 1)))(s4)

    b1 = encoder.get_layer("activation_161").output
    b1 = ZeroPadding2D((1, 1))(b1)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    dropout = Dropout(0.3)(d4)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(dropout)

    unet = Model(inputs, outputs, name="InceptionResNetV2-UNet")

    return unet


def callback():
    """Creates a list of callbacks for model training.

    Args:
      None.

    Returns:
      list of tf.keras.callbacks: a list containing the created callbacks,
        ready to be passed to a model's `fit` method.
    """

    weight_path = 'model.h5'

    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_dice_coef',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)

    early = EarlyStopping(monitor='val_loss',
                          mode='min',
                          patience=4)

    callbacks = [checkpoint, early]

    return callbacks


def visualize_predictions(model, valid_generator):
    """Visualizes predictions on a sample of validation data.

    Args:
        model: A trained Keras model for making predictions.
        valid_generator: A data generator object representing the validation data.

    Returns:
        None.
    """
    random_val_samples = valid_generator.__next__()
    val_image_samples = random_val_samples[0]
    val_mask_samples = random_val_samples[1]

    predicted_masks = model.predict(val_image_samples)
    predicted_masks = (predicted_masks >= 0.5).astype(int)

    fig, ax = plt.subplots(15, 3, figsize=(20, 120))

    for i in range(0, 15):
        ax[i, 0].imshow(val_image_samples[i])
        ax[i, 0].title.set_text('Original Image')
        ax[i, 1].imshow(val_mask_samples[i])
        ax[i, 1].title.set_text('Actual Mask')
        ax[i, 2].imshow(predicted_masks[i])
        ax[i, 2].title.set_text('Predicted Mask')


def train_model(img_size, batch_size, epochs):
    model = build_model(input_shape=img_size)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=dice_coef_loss,
                  metrics=[dice_coef, iou_coef])
    print(model.summary())

    train, val = data_generator(DATA, IMG_SIZE)

    STEP_SIZE_TRAIN, STEP_SIZE_VALID = calculate_step_sizes(train, val, BATCH_SIZE)
    print('Total number of batches', STEP_SIZE_TRAIN, 'and', STEP_SIZE_VALID)

    callbacks_list = callback()

    history = model.fit(train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=callbacks_list)

    visualize_predictions(model, val)

    return history


def plot_hist_curves(history):
    """Plots training and validation loss and accuracy curves from a Keras history object.

    Args:
        history: A Keras history object containing training and validation loss and accuracy metrics.

    Returns:
        None.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    dice = history.history['dice_coef']
    val_dice = history.history['val_dice_coef']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='Train')
    plt.plot(epochs, val_loss, label='Val')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, dice, label='Train')
    plt.plot(epochs, val_dice, label='Val')
    plt.title('Dice Coef')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    hist = train_model(IMG_SIZE, BATCH_SIZE, EPOCHS)
    plot_hist_curves(hist)
