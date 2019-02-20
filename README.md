# aphrodite-project

## About
This project is a deep learning model based on convolutional neural network that aim to calculate similarity between two faces in case of KTP selfie. The basic methodology use siamese network to compare two images.

This project is divided into two directory:

- embeddings: Contain pre-trained model to detect face and generate feature vector which is embeddings to compare face similarity

- siamese: Contain siamese convolutional neural network model and at&t face dataset. This model is experimental and not yet qualified to be used in production. You can train this model and play araound with the CNN architecure and code structure

You can find their README.md in each directory


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1. There are some dependecies you need to install into the environment. run requirement.sh to install all the dependencies or packages needed by this project

### Installing Dependencies

Install all dependecies as easy as run requirement.sh file in this repo.

```bash
sh requirement.sh
```

## Authors

* **Richard Adiguna**

## License

This project is licensed under the KoinWorks License.

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
