# CS4100-Research-project


This was a project I conducted over the period of a semester for my Artificial Intelligence class at Northeastern University.


## References:
Kelley, D. R., et al. (2018). Sequential regulatory activity prediction across 
chromosomes with convolutional neural networks. *Genome Research*, 28(5), 739–750. 
https://doi.org/10.1101/gr.227819.117

(extra link (same): https://genome.cshlp.org/content/28/5/739)

## Results
Tested multiple times, across multiple models. Sometimes RNN outperformed the FNO and vice-versa, but overall had comparative results. The main 
metric measured was Pearson R, and the reason why the models created do not outperform the ones in Kelley, D. R. et al. is due to computation restraints, and less data used (their about 131000 of sequences used vs my 2048 sequences used here). This project was quite informal, however I am still in the process of learning how to do professional/high-level research and would like to continue working on this project- possibly measuring other biological methods to analyze genomic data.

| Model | Pearson R |

Basenji CNN Kelley, D. R et al. | .77 (for non-zero genes), .85 across all cell types

| FNO | 0.3657    |

| BiLSTM | 0.3681    |

| CNN | 0.1753    |

Our results match Kelley, D. R et al. results at their third quartile results, .3673 (for single cell types), which suggests that the models created in this project were able to outperform some of the CNN's trained on a larger dataset.


## Next Steps:

-Try and build models with Poisson Loss to better replicate the models in Kelley, D. R et al.

-Try to explore with GPU settings and PyTorch CUDA techniques to try and train/optimize a model with more compute

-Develop an actual design for research/potentially gather enough information to create a proper research paper

-Explain CAGE more in depth, and its applications on this readme file in the future- for now, just focusing on the development of the Neural Networks

