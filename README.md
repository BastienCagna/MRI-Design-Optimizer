# MRI-Design-Optimizer

This framework (will) provides a pipeline that compute the more
efficient design of a brain MRI experiment using randomly generated designs.

## Motivations
This project is developped to improve the results obtained with MRI
experimentation at the [INT (Institut des Neurosciences de la Timone)](http://www.int.univ-amu.fr/).
Algorithms used here are based on the works of K.J Frinston & al. (1999):
Stochastic Designs in Event-Related fMRI and the article of Rik Hanson
on Design Efficiency.

## Progress

The project is new and not finished at all. Experimentation on an audio
stimulation study is currently in progress.
Some new parts have been added to the project concerning fMRI data analysis (GLM directory).

## The Optimizer Pipeline
The process is divided in five different parts:

1. Create the parameters file
2. Generate a large set of designs
3. Compute the efficiency of each design for each desired contrast
4. Find the best design
5. Export the design to a CSV file

The package provides also some viewing functions. You can, for example
see the efficiencies distribution over all the designs for each contrast.

### Inputs
To do...

### Output
* designs.pck:
* efficiencies.pck:
* ...

## Design's parameters
(comming soon...)

## Example
Some examples that run the optimizer pipeline are available in the "examples" directory.
tHose script have been used to test the pipeline and to apply it to real experiment runned by the INT team BANCO.