<span style="float:right;">[[source]](https://github.com/D3lt4lph4/pythondoc/tree/master/jpeg_deep/displayer/displayers.py#L95)</span>

## DisplayerObjects


```python
DisplayerObjects(classes: List=None, confidence_threshold: float=0.5)
```


Displayer detection tasks.

__Argument:__

- classes: A list of all the classes to be predicted, should be in the same order as the labels. If None, will use the Pascal VOC classes.
- confidence_threshold: The threshold under which objects will not be considered as detection.

***

<span style="float:right;">[[source]](https://github.com/D3lt4lph4/pythondoc/tree/master/jpeg_deep/displayer/displayers.py#L116)</span>

### display


```python
display(predictions: object, inputs: object)
```


Function to display the predictions on top of the input image.

__Arguments:__

- predictions: The predictions as returned by the detection network, should be an array of size (batch_size, n_boxes, 6), (prediction, confidence, x_min, y_min, x_max, y_max).
- inputs: The inputs images to the detector.

__Returns:__

Nothing, will display all the predictions.

***

<span style="float:right;">[[source]](https://github.com/D3lt4lph4/pythondoc/tree/master/jpeg_deep/displayer/displayers.py#L166)</span>

### display_with_gt


```python
display_with_gt(predictions: object, inputs: object, groundtruth: object)
```


Function to display the predictions on top of the input image. The ground truth will be added.

__Arguments:__

- predictions: The predictions as returned by the classifiers, should be an array of size (batch_size, n_boxes, 6), (prediction, confidence, x_min, y_min, x_max, y_max).
- inputs: The inputs images to the classifier.
- groundtruth: The real label of the predictions, should be an array of size (batch_size, n_boxes, 6), (prediction, confidence, x_min, y_min, x_max, y_max).

__Returns:__

Nothing, will display all the predictions alongside the groundtruths.

***

<span style="float:right;">[[source]](https://github.com/D3lt4lph4/pythondoc/tree/master/jpeg_deep/displayer/displayers.py#L9)</span>

## ImageNetDisplayer


```python
ImageNetDisplayer(index_file: str)
```


Displayer for an imagenet based classifier.

__Argument:__

- index_file: The json file containing the mapping of the index/classes (see details below).

__Details:__


Details of the content of the index file:
```json
    {
        "0": [
            "n01440764",
            "tench"],
        ...
    }
```


***

<span style="float:right;">[[source]](https://github.com/D3lt4lph4/pythondoc/tree/master/jpeg_deep/displayer/displayers.py#L37)</span>

### display


```python
display(predictions: object, inputs: object)
```


Function to display the predictions on top of the input image.

__Arguments:__

- predictions: The predictions as returned by the classifiers, should be an array of size (batch_size, 1000)
- inputs: The inputs images to the classifier.

__Returns:__

Nothing, will display all the predictions.

***

<span style="float:right;">[[source]](https://github.com/D3lt4lph4/pythondoc/tree/master/jpeg_deep/displayer/displayers.py#L64)</span>

### display_with_gt


```python
display_with_gt(prediction: object, inputs: object, groundtruth: object)
```


Function to display the predictions on top of the input image. The ground truth will be added.

__Arguments:__

- predictions: The predictions as returned by the classifiers, should be an array of size (batch_size, 1000).
- inputs: The inputs images to the classifier.
- groundtruth: The real label of the predictions, should be an array of size (batch_size, 1000).

__Returns:__

Nothing, will display all the predictions alongside the groundtruths.

