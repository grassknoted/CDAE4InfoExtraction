# CDAE4InfoExtraction
Cross-domain Auto-encoders for Information Extraction, using Capsule Networks


## Load the data and preprocess it into a 28 by 28 form

Use Tensorflow or other packages to reshape the images and preprocess the training data.

## Build a distribution using IWAE or VAE incorporating Capsule networks

Build on the CapsNet framework given in the other program. In addition to that, extract two vectors from
each capsule Mu and Sigma to generate 'n' distributions and build 'n' capsules. Use Reparametrization trick
for multidimensional data to obtain the distributions. Once the capsules are setup, we use KL Divergence in
addition to MSE to train the model on the Adam optimizer.

## Sample from the distribution and decode to identify 'important' elements

Use Importance Sampling to get 'n' samples from the correctly identified distribution and decode that using
a deep setup and generate the output (multi-hot output of the vocabulary). In case we aren't using IWAE for the 
initial prototypes, just sample from the top 1-3 standard deviations of the distribution and decode it. The learning
methodologies are a combination of Backpropogation and Hinton's novel learning method for CapsNets. 

## Testing

For testing, just feed a test instance to the CapsNets and the vectors in the vocabulary will identify the 'important'
parts of the data. This solves the objective.

## Reasons for using specific elements in the setup
1. IWAE - With the tight lower bound on the VAE, IWAE provides some freedom to generate accurate distributions.
2. CapsNets - Like the IWAE, CapsNets are improvements to the standard CNN setup to obtrain better results.
3. VAE - To perform Transfer Learning betweek domains.
4. Dataset - Due to unavailability of data in the form we are looking for, we decided to synthesize our own data.
