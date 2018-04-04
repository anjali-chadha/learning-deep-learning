Keras, Tensorflow, Caffe, PyTorch, Theano.... (Too many options!!?)

When to use what?

## Frameworks
* Microsoft CNTK
* Google TF
* Google Keras
* Facebook Caffe2
* Facebook PyTorch
* Amazon MXNetw

## Discussion
* Keras is high-level API built on Tensorflow (and can be used on top of Theano too).
* More user-friendly as compared to TF.
* Hmm.. then why will I ever use Tensorflow?
* Keras useful for rapid prototyping. Using keras, can build very simple or very complex nerual networks within a few minutes.
* Keras - pythonic, modularity
* Keras code is portable - can implement nerual network in Keras using Theano as a backend and then specidy the backend to run o TF, with no changes to code
* Tensorflow - use that to tinker with low-level details of neural network.
* TF offers more advanced operations as compared to Keras, and gives more control over network
* TF built around concept of Static Computational Graph. First, you define everything that is going to happen inside your framework, then you run it.
* PyTorch is Dynamic Computational Graph. Advantages -
  - Networks are modular. Can implement and debug each part separately.
  - Dynamic data structures inside network. Can define, change, execute nodes as you go.
  - In RNNs - input sequence length has to stay constant for static graphs. For instance, for a sentiment analysis model of English sentences
  we have to fix the sentence length to some maximum value and pad smaller sequences with zeros.

## Programming Details
* Both TF and Theano expects a 4-D tensor as input. There are some differences:
  - TF tensor format - (samples, rows, cols, **channels**)
  - Theano tensor format - (samples, **channels**, rows, cols)
  - To avoid ambiguity, while programming in Keras, we explicitly set the image dimension ordering to either 'tf' or 'th'
  
* Keras provide ability to **freeze** layers i.e we will not update the weights of the layer. This is useful when we are fine tuning a model. This is acheived by passing *trainable=False* parameter.
