# CosMx: data processing
The expression matrices, metadata, and FOV files for the four AppNL-G-F tissue sections were read in using squidpy’s read_nanostring() function. 
Transcripts associated with negative probes were removed from the count matrices. 
Cells were manually assigned regional annotation based on visual alignment with the Allen Brain Mouse Atlas reference (Figure S1C). 
Cells with a high or low number of genes or transcripts were removed. 

In order to identify likely mis-segmented doublets (i.e., two cells that were incorrectly identified as one cell during the segmentation process), 
scrublet61 was run on each of the hippocampal sections, separately. 
Cells that were assigned a high doublet probability were removed.

To account for potential non-specific binding of probes to amyloid plaques, we analyzed the expression level of each gene, including the negative probes, within a cell overlapping with a plaque and correlated it to the plaque overlap. 

We found that particularly two negative probes showed a strong correlation (R2 = 0.65 for NegPrb1 and R2 = 0.47 for NegPrb7), indicating non-specific binding of probes and arguably also genes, to plaques. We discarded all transcripts that had a higher correlation than the top two negative probes (for a total of 56 probes) from further analysis, as likely candidates of nonspecific binding. 

To remove cells that may have still been contaminated with, we subclustered the microglial cells, identified a cluster of cells characterized by higher counts of negative probe counts, and removed this cluster (141 cells) from subsequent analysis. A total of 37,840 high-quality cells across all tissue sections were retained for subsequent analysis.



# Amyloid segmentation
Amyloid-β plaques were annotated using the QuPath software through a pixel classification threshold with an additional size threshold of 65,000 px to discriminate the pathology from background and artificial staining. The segmentation was then manually checked to ensure segmented plaques were of high quality. The segmented plaques were then exported to ImageJ and converted into a binary mask for alignment to the transcriptome.
Using a previously described technique, the adjacent slide staining was aligned to the transcriptome of the Stereo-seq chip through the selection of corresponding landmarks on both the staining and the transcriptome.8 Using the Fiji “Landmark correspondences” plugin, the staining was then aligned to overlay with the transcriptome. The same method was followed for alignment of the Allen Brain Atlas with the transcriptome that allowed for assignment of brain regions to the transcriptome (Figures S1C and S1D). After alignment of adjacent slide stainings, segmented masks from either side of each chip were summed to obtain one binary amyloid segmentation mask per chip.