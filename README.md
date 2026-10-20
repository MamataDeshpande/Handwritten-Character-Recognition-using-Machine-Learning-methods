# Handwritten Digit Recognition using Machine Learning Methods

Project report: **23 May 2018 – 30 May 2018**.

This project recognizes handwritten digits from **0 to 9** using a **Convolutional Neural Network (CNN)** with the MNIST dataset and TensorFlow. The report describes 28 × 28 grayscale images, with **60,000 training images** and **10,000 test images**.

## Method

The model described in Chapter 5 uses:

- A convolutional layer with **32 filters of size 5**, followed by max pooling.
- A convolutional layer with **64 filters of size 3**, followed by max pooling.
- A fully connected layer with **1,024 neurons**, **0.3 dropout**, and **10 output classes**.
- Softmax cross-entropy loss, the Adam optimizer (learning rate **0.001**), and **10,000 mini-batch training iterations** with batches of 100 images.

The model learns image features and predicts the digit corresponding to a test image.

## Code

| File | Description |
|---|---|
| [train.py](code/train.py) | CNN model definition and MNIST training code from Chapter 5. |
| [predict.py](code/predict.py) | Checkpoint loading, image display, and digit prediction code from Chapter 5. |
| [report_code_as_printed.txt](code/report_code_as_printed.txt) | Text extracted directly from the printed code pages, retaining the original PDF layout. |
| [2.mlai.pdf](report/2.mlai.pdf) | Complete project report. |

The Python files transcribe the report code, with page-break indentation and split lines rejoined where needed. The **model architecture, layer parameters, training loop, and prediction method are retained**. They use **TensorFlow 1.x** APIs (`tf.Session`, `tf.layers`, `tf.contrib` and the older MNIST loader), which are not available in current TensorFlow releases. The two checkpoint paths also differ in the original report, and have not been silently changed.

These files have been checked for Python syntax; their execution has **not** been validated against a working historical TensorFlow environment.

## Figures and results from the report

The following images were extracted from the original PDF without redrawing or modifying them.

### Fig. 4.1 – Work Flow

![Fig. 4.1 Work Flow](images/fig_4_1_workflow.png)

### Fig. 4.2 – Architecture

![Fig. 4.2 Architecture](images/fig_4_2_architecture.png)

### Fig. 4.3 – CNN

![Fig. 4.3 CNN](images/fig_4_3_cnn.jpeg)

### Fig. 6.1 – Testing of Data

![Fig. 6.1 Testing of Data](images/fig_6_1_testing_data.jpeg)

### Fig. 6.2 – Output

![Fig. 6.2 Output](images/fig_6_2_output.jpeg)

### Fig. 6.3 – Recognized Digit

![Fig. 6.3 Recognized Digit](images/fig_6_3_recognized_digit.jpeg)

The report describes recognition accuracy above 96% in its abstract, but Chapter 6 provides screenshots rather than a detailed numerical test report. No new performance figures have been added to this repository.

## Project files

```text
handwritten-digit-recognition-ml-methods/
├── README.md
├── .gitignore
├── code/
│   ├── train.py
│   ├── predict.py
│   └── report_code_as_printed.txt
├── images/
│   ├── fig_4_1_workflow.png
│   ├── fig_4_2_architecture.png
│   ├── fig_4_3_cnn.jpeg
│   ├── fig_6_1_testing_data.jpeg
│   ├── fig_6_2_output.jpeg
│   └── fig_6_3_recognized_digit.jpeg
└── report/
    └── 2.mlai.pdf
```
