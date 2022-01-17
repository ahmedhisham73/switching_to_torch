# switching_to_torch
this repository is set to give strong hands on using PyTorch over tensorflow 
the repository structure is set to be as follow :

1-simple tensor for checking GPU if available and PyTorch is correctly installed with GPU support or not , in addition to perform simple basic tensor operations.


2-Neural networks basics which demonstrates the usage of the NN ( neural network ) module built in PyTorch. 


3-torch training which contains the demonstration of basic CNN and transfer learning with CPU and GPU training , in addition to saving and restoring model for inference.


4-torch CNN contains the CNN models that will be trained for Visual classification , detection and segmentation tasks. 


5-DCGAN training contains the reference implementation of DCGAN from official PyTorch documentation , the reference model was tuned 
to perform the generation on digital knee Xray dataset , with modifying its parameters.

6-APP directory contains pytorch model deployment using heroku , a simple handwritten MNIST digit classification model deployed in heroku platform ,where the directory also contains the required packages for installation(check for torch official documentation for the versions of Torch and Torchvision , since it might be Updated while you are reading this description, also make sure that in production avoid using Torch Cuda supported and use Torch CPU version) which contains the webAPP starting point ,gitignore file , run time file which contains the python version DENOTED by Heroku ( don't use any version except the published one on Heroku , otherwise your app will not git pushed onto heroku.
