# keyword_spotting
Using turn on the light dataset and openWakeword repo's proposed model to compare with a more linear one.
The work shows that there is no need for a deep network in classification, as a PCA + Ridge Classification pipeline performs slightly better, with way less parameters and much faster.
This is probbably mostly due to their embeddings model, which does a remarkable job at extracting well-seperable features between the positve and negative classes.

This project was done as part of the Deep Learning assignments and was a team effort!
