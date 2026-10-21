# Handwritten Digit Recognition using Machine Learning Methods


This project recognizes handwritten digits from **0 to 9** using a **Convolutional Neural Network (CNN)** with the MNIST dataset and TensorFlow. The report describes 28 × 28 grayscale images, with **60,000 training images** and **10,000 test images**.

## Method

The model uses:

- A convolutional layer with **32 filters of size 5**, followed by max pooling.
- A convolutional layer with **64 filters of size 3**, followed by max pooling.
- A fully connected layer with **1,024 neurons**, **0.3 dropout**, and **10 output classes**.
- Softmax cross-entropy loss, the Adam optimizer (learning rate **0.001**), and **10,000 mini-batch training iterations** with batches of 100 images.

The model learns image features and predicts the digit corresponding to a test image.

## Code

| File | Description |

| [train.py](code/train.py) | CNN model definition and MNIST training code 
| [predict.py](code/predict.py) | Checkpoint loading, image display, and digit prediction code 
| [report_code_as_printed.txt](code/report_code_as_printed.txt) 



## Figures and results 


### Fig. 4.1 – Work Flow

<img width="310" height="228" alt="image" src="https://github.com/user-attachments/assets/ae4439a8-3a07-435e-a893-014be333578b" />


### Fig. 4.2 – Architecture

<img width="276" height="238" alt="image" src="https://github.com/user-attachments/assets/0ef76df5-5a10-4ba5-997a-7f0c28c7a000" />


### Fig. 4.3 – CNN

<img width="317" height="93" alt="image" src="https://github.com/user-attachments/assets/8d967638-6d74-4093-8791-f6393b449633" />


### Fig. 6.1 – Testing of Data

<img width="244" height="160" alt="image" src="https://github.com/user-attachments/assets/4eba1ba7-407b-4bdd-b709-511acf1ea231" />


### Fig. 6.2 – Output

<img width="245" height="162" alt="image" src="https://github.com/user-attachments/assets/df12410d-cf85-4a0a-89f0-93f035f08084" />


### Fig. 6.3 – Recognized Digit

<img width="244" height="180" alt="image" src="https://github.com/user-attachments/assets/581b9082-7cfb-4553-b656-79fd7a89920d" />

The project describes recognition accuracy above 96% in its abstract.
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
