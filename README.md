# Make_Moons: Deep Learning Framework Comparison

## Introduction
In the process of comparing TensorFlow/Keras and PyTorch through a classification task using the make_moons dataset, notable differences in performance and user experience emerged. PyTorch impressed with its speed, completing the training process noticeably faster than TensorFlow/Keras. This advantage in speed can be particularly beneficial in scenarios where rapid prototyping and iterative testing are required. However, it was observed that the model trained with PyTorch exhibited slightly lower accuracy compared to its TensorFlow/Keras counterpart. Despite this, the ease of implementing dynamic computation graphs and the intuitive nature of PyTorch's programming model made it an enjoyable framework to work with. Ultimately, while TensorFlow/Keras might be preferred for projects where model accuracy is paramount, PyTorch stands out for its efficiency and developer-friendly approach, especially in research contexts where flexibility and speed are crucial.

## Installation

Before running the project, ensure that you have Python installed on your machine. Then, install the necessary dependencies using the following commands:

```
pip install tensorflow keras torch
```

## Usage

To run the TensorFlow/Keras model:

```
python tensorflow_keras_model.py
```

To run the PyTorch model:

```
python pytorch_model.py
```

The scripts will train the models on the `make_moons` dataset and output the performance metrics for comparison.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
