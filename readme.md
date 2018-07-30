# The Graves-ey Font

A simple implementation of [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850). We try to task of:

1) Unconditional Generation: Let the text generator just blabber random stuff. 

2) Conditional Generation: The text generator tries to write what you tell it to write.

3) Handwriting Recognition: Given a sequence of handwritting strokes, it predictes the charaters.

Experiment Docs: [Google Doc](https://docs.google.com/document/d/1HpNZtQtLDDAM9jQ106UIA4NGUZ2Xedkjni9cu1K8n6U/edit?usp=sharing)

## Training Details:

* 3 Variants of Architecture tried:

  *  Single GRU(900)-Linear(122)
  
  *  GRU(256)-GRU(512)-Linear(256)-Linear(122)
  
  *  GRU(400)-GRU(400)-GRU(400)-Linear(122)

* Details: 
  
  * All predict parameters of a Mixture of 20 Gaussians
  
  * (2) and (3) have skip connections as in paper. 

  * (1) Trained with no weight balancing for pen lift.

  * ADAM Optimizer

  * 5500 Training Sequences, 250 Test Sequences (Graves etal. use ~ 10K sequences for training)


## 1: Unconditional Generation:

### Our Results:

On validation set:

![val](misc/handwriting.png)

On Self-Generation:

![sg](misc/self_hw_1.png)

From Paper:

![sg](misc/self_hw_2.png)

## 2: Conditional Generation

