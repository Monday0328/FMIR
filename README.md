# FMIR
**A Foundation Model-Based Framework for Robust Medical Image Registration**  
*Accepted at ISBI 2026*

This repository hosts the official PyTorch implementation of FMIR, a foundation model-based image registration framework presented in our paper: "FMIR: A Foundation Model-based Framework for Robust Medical Image Registration". FMIR leverages pre-trained 2D vision foundation models (e.g., DINOv3, SAM) as feature encoders to extract domain-invariant anatomical representations, combined with a general registration head and a novel channel regularization strategy. Trained on a single dataset, FMIR achieves state-of-the-art in-domain performance while maintaining strong generalization to out-of-domain images, offering a scalable and resource-efficient path toward building practical registration foundation models.


Highlights
Foundation Model-based Encoder: Utilizes pre-trained 2D vision models (DINOv3, SAM) to extract robust, domain-invariant features via a slice-wise 3D adaptation pipeline.

General Registration Head: A multi-scale pyramid architecture that estimates deformation fields from extracted features, compatible with various foundation model backbones without retraining.

Channel Regularization: A novel training strategy that randomly selects feature subsets to prevent overfitting to dataset-specific priors, significantly enhancing cross-domain generalization.

State-of-the-Art Performance: Outperforms existing registration methods on ACDC and Abdomen CT datasets, with superior in-domain accuracy and out-of-domain robustness.
[FMIR-0 (1).pdf](https://github.com/user-attachments/files/26258678/FMIR-0.1.pdf)
[channel_visualization.pdf](https://github.com/user-attachments/files/26258680/channel_visualization.pdf)

Datasets
We evaluate FMIR on two public benchmarks:

ACDC (Cardiac MR): Intra-subject registration between end-diastole (ED) and end-systole (ES) phases.

Abdomen CT (Learn2Reg 2020): Inter-subject registration with 13 annotated organs.

Usage
Run the following commands in the ./src folder to reproduce the results:

bash
# Train on ACDC (unsupervised)
python train.py -d acdc -m FMIR --unsupervised --epochs 200

# Train on Abdomen CT (weakly-supervised)
python train.py -d abdomen -m FMIR --weakly --epochs 200

# Test on ACDC with DINO encoder
python test.py -d acdc -m FMIR -e dino

# Test on Abdomen CT with SAM encoder (zero-shot)
python test.py -d abdomen -m FMIR -e sam
-d: Dataset (acdc or abdomen)

-m: Model name (FMIR)

-e: Foundation model encoder (dino or sam)

--unsupervised / --weakly: Training mode

--epochs: Number of training epochs

Model Checkpoints
Pre-trained FMIR models are available for download:

FMIR_ACDC_DINO.pth – Trained on ACDC with DINO encoder.

FMIR_Abdomen_DINO.pth – Trained on Abdomen CT with DINO encoder.

FMIR_Hybrid_DINO.pth – Trained on combined dataset.

Citation
If our work has influenced or contributed to your research, please kindly acknowledge it by citing:
@article{chen2026fmir,
  title={FMIR: A Foundation Model-based Framework for Robust Medical Image Registration},
  author={Chen, Lin and He, Yue and Zhang, Fengting and Wang, Yaonan and Lin, Fengming and Chen, Xiang and Liu, Min},
  journal={arXiv preprint arXiv:2601.19114},
  year={2026}
}



