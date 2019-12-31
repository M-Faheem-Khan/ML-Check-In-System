# Self Hosted AI-Powered Check-In
This open-source project was created to help events verify attendees, while having full control over their data.


## High Level View
Ideally, for this to work, you will need: 
```.env
A laptop with a camera and a dedicated computer running Ubuntu.

Thats it!
```
The logic behind this system is as follows:
```
Using the browser, when the user clicks the spacebar, a photo
is taken and is sent using a React PWA (again locally served, reasoning is if desired, you can make this a fully online system).
That server then compares that photo using ML and AI (look at the software stack below)
and (using a local MongoDB instance) updates that persons attendace status.
After updating their status, the browser will show a {user} checked in, and repeat
```

## Installation
We will attempt to install all needed packages and software, however if that fails, look at the source code to install the
packages

To start we will install:

- Node.js and NPM and React
- MongoDB Server
- 

### Pre-Made Components
To start, we will create the React App (i.e the "front-end" ), the MongoDB instance and the TensorFlow models. 

**It is Highly recommended to modify this system to better suit your needs**. All software in this repo is MIT, so go wild!

## Software Stack
```
pip install tensorflow
pip install matplotlib
pip install keras-vggface
pip install mtcnn
pip install numpy
```

The above software is needed to do the recognition and
tell us the results

## Hosting Environment
By using Ubuntu and MongoDB, we are able to self host this model,
as serve webpages that will act as the "front-end". Using the laptop camera, the user
will click the spacebar 


# For Muhammed
Okay so this is what I was thinking. 

From start to end:

React: We make a super simple app that allows the user to take a photo using the WebCam, and make a simple
request with the image. 

I was thinking a "central-brain" Node.js Express App, that accepts this image, and then
sends the image to a trained TF model which returns the most likely person it is. That person is then registered,
and then a "yes" is sent to the node.js app, which updates a MongoDB instance, 


