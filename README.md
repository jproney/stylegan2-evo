# Project Idea
Genetic algorithms are a theoretically powerful method for solving optimization problems when objectives 
are complex and search spaces are discrete. GAs have received less attention in the AI/ML 
communities over the past decade, mostly because these fields are primarily focused on continuous problems with 
differentiable objectives. However, discrete, non-differentiable optimization problems are becoming increasingly 
important. A few examples of such problems include drug design and neural architecture search. GAs seem like
they may be poised to tackle such problems, but there are a few roadblocks to their successful application. 
One issue with GAs is that, when the space of possible phenotypes is vast and fitness landscapes are rugged,
it becomes easy for populations to become stuck in local optima. If we look to nature for a solution to this 
problem, we see that biological organisms have a complex hierarchy of developmental processes that
map genotype space into phenotype space. These developmental mappings effectively restrict the phenotypic
space to patterns which are more likely to be successful, and allow for evolution to proceed faster. For
instance, due to the developmental modularity of biological organisms, a single mutation can cause an
animal to develop an extra limb. Rather than developing the limb cell by cell and traversing the low-fitness 
state of having a partial limb, a higher-order regulatory mutation can simply copy an existing limb blueprint
and leap over fitness valleys in a single generation.

The idea of building developmental encodings into GAs is not new. Ken Stanley's HyperNEAT is a good
example of previous work in this area. HyperNEAT uses Compositional Pattern Producing Networks to 
evolve a developmental plan for a neural network. While this work is definitely a step in the right 
direction, I think now might be the time to take it a step farther. While CPPNs are very expressive 
models and can subjectively replicate many key features of biological development, it is still unlikely
that they are able to effectively encode the types of functions that map genes into complex structures like 
the human brain. However, over the past few years we have seen rapid progress in deep generative models.
My basic idea is to use these types of deep generative models as development functions for GAs. For instance,
if we had a neural net that could embed and decode neural network architectures into a compact latent space, 
we could use vectors in that latent space as genomes in a genetic algorithm, while using the neural decoder function
as a developmental encoding. There are a lot of complicated issues with this idea, and it isn't fully developed yet. This repo is a simple proof-of-concept attempt to evolve faces to meet some objective using the StyleGAN2 latent representation
as a genome.

# The Basic Plan
## Evolution Procedure
* A finite population (size ~ 1000?) is initialized using samples from standard normal distribution,
which will be passed through StyleGAN's MLP and truncated to form the latent-space genomes.
*  At each iteration:
    - All genomes are generated into faces by StyleGAN, and each face is assigned a fitness based on similarity 
    measure to a target face. Similarity measure might start as MSE, but something more complicated will
    probably be needed eventually.
    - If N is the population size, repeat the following N times:
        - Sample 2 individuals from with population (with or without replacement?)
        - Add a random gaussian vector to each genome to represent mutation, with the gaussian
        truncated if it is below a certain value
        - Generate a a set of crossover points on the genome according to a Poisson process
        - Cross over the two parent genomes at the crossover points
        - Select one of the recombinant chromosomes as the child genome
        
# Initial Results
        
The following face evolved after 100 generations of rank selection with a population size of 128. Fitness was calculated with a VGG-16 perceputal loss. Left image is the target, right image is evolved.

![Results after 100 generations](https://github.com/jproney/stylegan2-evo/raw/master/target_gen.png)

Overall I think this result is pretty good given the small population size and training time. Further advances could probably be made by using a face-specific perceptual loss like the FaceNet embedding. It would also be interesting to use the affine-transformed styles as a genome, since these have been explicitly optimized to behave as independently-acting "genes."

Of course, what we've done here isn't super useful, especially since we can project images into the StyleGAN latent space using backpropagation. However, a similar technique could prove useful when training an agent on a reinforcement learning task where the objective is not differentiable.
