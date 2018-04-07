
========================================================================
## Detection Algorithms (Week 3)
### Object Detection
* Classification with Localisation (Bounding Box) - generally one object
* Detection - Yolo does this - can have more objects in single image
* Image Classification - has one object generally
* Classification with localisation
* Self driving car classes
  * Pedestrian
  * Car
  * Motorcycle
  * Background
* Softmax of these classes
* If want bounding box, output layer will output four more numbers - bx, by, bw, bh
  * Midpoint - bx, by
  * Height - bh
  * Width - bh
  * Upper left (0,0) and Lower right (1, 1)
* Neural Network outputting - class labels (1-4) , bx, by, bw, bh
* Target label y -  [pc bx by bh bw c1 c2 c3] (8 elements in it)
* pc - if there is any object in the image
* In case no object in the image [0 DC.......] 
* DC - Don't care
* Loss function - L(y, yhat) calculate
* Two cases, if pc = 1 and if pc =0

### Landmark Detection -
* NN can output important points in the image - called landmarks
* Facial Recognition application - want the algorithm where is the corner of the eye (x, y)
* NN Output layer will output two numbers- representing the corner of the eye
* If we want all four corners of the eye, NN will output estimate positions of all four corners
* What if want multiple points along the circumference of the eye?
* Keypoints along the mouth - smiling or frowing
* Edges of the nose
* Landmarks - labelled training set
* Person's face input
* Goes through convnet
* Output - face or not? 
* Also output and multiple points - l1x, l1y, l2x, l2y, l64x, l64y 
* Total 129 output units
* L - landmarks
* Snapchat - augmented reality filters, computer graphics, 
* Labelled training set - manually labelling
* People Pose Detection - Key Position - chest, shoulder, elbow positions
* Specify key landmarks - 
* The identity of landmark1, 2,3,.4 should be consistent across all the input images
* Recognising face emotions

### Object Detection
* Car Detection example
* Labelled training set - with closely cropped images of car, x is only the car
* Conv Net with these type of images will just output 1 or 0, telling car or not
* Sliding Windows Detection 
* Take a window size - Input in the conv net - ONly that small rectangular region
* Next input - window slided - send through convnet
* Goes through every region of the image
* Passes small cropped image to the convnet
* Take these windows, these square boxes and slide them acorss the entire image and classify every square region with some stride as containing a car or not.
* Disadvantage of Sliding Windows Detection - computational cost.
* Because you are cropping out so many different square regions in the image and running each of them independently through a CNN
* If we use a big stride, a very big step size, that will reduce the number of windows you need to pass through the convnet but the granularity may hurt. performance.
* If very fine granularity, small stride, then huge number of all these little regions passing through convnet means high computationsl cost.
* Earlier era - People used hand engineered features along with simple classifier for object detection. These classifiers were relatively cheaper to compute. So SWD was working ok.
* But with convnet, SWD is expensive and slow

### Convolution implementation of Sliding Window Detection
* Turning FC layer into convolutional layer
* Output layer - 4 units - probablities of each class
* Let's say an input image - 4 windows.
* Lot of computation done by the convnets will be highly duplicative for these four windows
* Convolutional implemetation of sliding windows allows four passesin convet to share these computations.
* Disadvantage - Positions of bounding boxes not very accurate.

### Bounding Box Predictions -
* Perfect Bounding Box is infact not always a square, different aspect ration
* YOLO
* Place a grid on the image. say 3 by 3
* For each grid, run the algorithm of image classfication and localisation
* For each grid, give the labels - y --> [pc, bx, by bh bw c1 c2 c3]
* Output - 3 * 3* 8
* Advantage - Gives precise bounding boxes
* If multiple bounding boxes in the grid cells --> problem
* See the midpoint of the object. Assign the object to the grid cell, where the center lies
* Single convolutional implementation
* Very fast
* How to encode bounding boxes?
* Mid point given relative to the grid cell upper left corner and lower right (1, 1)
* Width, height of the object given relative to the grid cell - can be greater than 1
* Mid Point - always between 0 and 1


### Intersection over Union -  Measure of overlap between two bounding boxes.
* Evaluating object localization 
* Computes size of the the intersection divided by the size of the union
* Predicted and ground truth boxes
* If overlap exact -> IOU = 1
* Higher the IOU, more accurate the bounding box
* Convention - IOU is greater than 0.5, then accurate.
* Can use more stringent criteria of IOU.

### Non-Max Suppression - 
* Say grid size is 19*19
* Only one car say
* Running object detection on all the grids.
* Everyone will report that they found the car.
* Might get multiple detections of a single object
* Non-max suppression helps with this problem
* Looks at the probabilities associated with each detection.
* First takes the largest one, say 0.9, It says this is my most confident detection.
* Next, looks at the remaining rectanbles and all the ones with a high overlap or high IOU with this one will get suppressed.


### Anchor Boxes
* What if grid cells want to detect multiple objects?
* Overlapping objects - Car and human falling mid point in the same grid
* With anchor boxer, pre define two different shapes called anchor boxes or anchor box shapes.
* Can use more anchor boxes - 5 or more.
* Say two anchor boxes.
* Output label - instead of a single vector, repeat it twice
* Each object is assigned to grid cell that contains the object's midpoint and anchor box for the grid cell with highest IOU
* Ouptput y - 3*3*(2*8)
* What if only one object in the grid cell?
* Two objects with same anchor boxes?
* Three objects with two anchor boxes?
* In practise, this happens rarely.
* Allows learning algorithm to specialise better.
* Tall, skinny objects, wide objects.
* Anchor boxes choosen by hand - 5 or 10 than spans a variety of shapesthat seem to cover the types of objects that you seem to detect. 
* Later yolo papers, use K means algorithm to group togehter two types of object shapes.
* Helps in automatically choosing anchor boxes

### YOLO

* Three objects - pedestrian, car, motorcycle, trying to detect
* Two anchor boxes
* 3 grid cell
* Output size - 3*3*2*8
* Go through each grid cell and make the output vector.
* Practise - 19*19 grid cell
* 5 anchor boxes
* Training process
* Run non-max suppressed output separately for each class

### Region Proposal
* Remember 



===================================================
