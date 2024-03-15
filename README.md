This research primarily focuses on employing deep learning segmentation approaches to detect deforestation and track the decline of forest cover. It specifically emphasizes forecasting parts of the forest that are extremely sensitive to deforestation. The objective of the project is to evaluate the effectiveness of semantic segmentation methods using the U-Net architecture, maybe with customized adjustments, by utilizing a recently curated dataset of satellite images.

**Research questions:**
-	RQ1: how to collect sequential satellite images?
-	RQ2: how to predict highly threatened areas of forest to deforestation?
-	RQ3: what are the modified U-Net architectures that will use in the study?
-	RQ4: how the model evaluation should do?
  
**Summary of research objectives:**
-	Evaluate the effectiveness of semantic segmentation methods, particularly utilizing the U-Net architecture, for detecting deforestation and tracking forest cover decline.
-	Explore the use of sequential satellite picture sequences to train the segmentation model and improve the prediction accuracy of changes in forest cover.
-	Develop a sophisticated deep-learning model capable of identifying areas where deforestation has occurred and differentiating between different land cover categories.
-	Investigate the most effective ways to utilize sequential satellite imagery to improve the predicted accuracy of the segmentation model.
-	Create a customized dataset of satellite images specifically tailored for identifying deforestation and tracking changes in forest cover.
-	Use attention mechanism to improve the feature learning of the u-net with the help of residual learning modules.
-	Reliable detection U-net architecture without utilizing auxiliary learning with multiple branching inside the model.
-	Conduct a meticulous examination of different versions of the U-Net architecture (different feature learning techniques), along with possible improvements or modifications, to determine the optimal method for precisely forecasting deforested regions and identifying instances of deforestation.
-	Enhance the progress of strong and accurate deep learning systems for detecting deforestation, providing important information to environmental stakeholders, conservationists, and policymakers.

 ![image](https://github.com/ashen007/forest-cover-shrinking/assets/48565752/360b2885-2875-40bc-9171-1b3ef318fe58)
General U-net architecture used in this study with the addition of feature learning modules showed in yellow box. Those modules will tested to found out which one will be the best match for optimal results.

![image](https://github.com/ashen007/forest-cover-shrinking/assets/48565752/6f4ae402-7d18-483a-b17a-3eda27035ae7)
Siam architecture, which use two separate contracting paths for each input image.

|Metric           |	Equation                        |
|-----------------|---------------------------------|
|Overall accuracy	| (TP + TN) / (TP + TN + FP + FN) |
|Precision	      | TP / (TP + FP)                  |
|Recall           |	TP / (TP + FN)                  |
|Dice             |	2∙TP / (2∙TP + FP + FN)         |
|Kappa            | (p_o  - P_e) / (1 - p_e)        |

metrics used in the study to compare and capture the performance of the models

### Improved Feature Learning U-net in Changed Detections

|Type	         | Network    	| Prec.	| Recall	| Dice   | Kappa  | F1   | Tot.  acc. |
|--------------|--------------|-------|---------|--------|--------|------|------------|
| Early fusion | Baseline 	  | 23.69	|58.19	  |32.36	 |26.97	  |33.68 |89.41       |
|	             | Res. block 	| 25.26	|63.22	  |32.94	 |28.39	  |36.09 |87.70       |
|	             | ResNext block|	26.35	|56.85	  |33.59	 |28.55	  |36.01 |90.17       |
|	             | ResNeSt block|	50.63	|32.23	  |36.49	 |29.98	  |39.39 |95.19       |
| Siam network | Baseline 	  | 25.22	|62.24	  |33.68	 |28.25	  |35.90 |87.41       |
|              | Res. block 	| 25.84	|64.56	  |33.65	 |29.25	  |36.91 |88.34       |
|              | ResNext block|	29.44	|60.18	  |34.64	 |30.76	  |39.54 |90.40       |
|              | ResNeSt block|	34.68	|46.97	  |37.55	 |33.43	  |39.90 |93.27       |

Test time performance of the models tested in phase one, best results are bold and highlighted

![image](https://github.com/ashen007/forest-cover-shrinking/assets/48565752/066ee0de-40ff-4f52-abd9-345709a2c130)

This shows the ground truth and each model predictions. Yellow area are the correct pixels model predicted and yellow areas are the model fails to predict but those areas has changes occurred. Red areas are where the model falsely predicted as changed areas. The left image shows the ResNeSt module used with early fusion architecture. Right image shows the ResNeSt module used in Siam architecture. As shown in table Siam network was predicting changes better than early fusion model aka less green areas in the right side image.

### Application of Attention Gates in U-nets

| Type  	     | Atten. type	| Network	      | Prec.	 | Recall	| Dice   | Kappa  | F1   | Tot.  acc.|
|--------------|--------------|---------------|--------|--------|--------|------- |------|-----------|
| Early fusion | Chan. (SE)	  | Res. block	  | 26.54	 | 61.83	|  34.50 |	28.78	|37.14 |  87.63    |
|		           |              | ResNext block	| 26.92	 | 57.15	|  33.48 |	28.43	|36.60 |  89.87    |
|		           |              | ResNeSt block	| 31.80	 | 54.18	|  37.02 |	32.27	|40.08 |  91.39    |
|	             | Chan. (FCA)	| Res. block	  | 25.79	 | 66.58	|  34.53 |	29.97	|37.18 |  88.07    |
|		           |              | ResNext block	| 19.36	 | 74.26	|  28.86 |	23.21	|30.71 |  81.65    |
|		           |              | ResNeSt block	| 40.66	 | 43.98	|  38.26 |	34.52	|42.26 |  94.47    |
|	             | Comb. (add.)	| Res. block	  | 27.04	 | 63.59	|  34.86 |	30.53	|37.94 |  88.45    |
|		           |              | ResNext block	| 26.81	 | 57.58	|  33.99 |	28.64	|36.59 |  88.49    |
|		           |              | ResNeSt block	| 25.75	 | 58.69	|  32.49 |	28.09	|35.79 |	88.45    |
|	             | Comb. (SP)	  | Res. block	  | 23.20	 | 59.03	|  29.87 |	24.88	|33.31 |	85.96    |
|		           |              | ResNext block	| 22.64	 | 61.05	|  30.63 |	24.68	|33.04 |	86.56    |
|		           |              | ResNeSt block	| 17.69	 | 30.22	|  20.46 |	14.87	|22.32 |	88.78    |
| Siam network | Chan. (SE)	  | Res. block	  | 23.03	 | 63.98	|  31.36 |	26.50	|33.87 |	87.38    |
|		           |              | ResNext block	| 25.69	 | 63.24	|  33.31 | 	28.83	|36.54 |	88.50    |
|		           |              | ResNeSt block	| 40.26	 | 45.77	|  39.03 |	35.13	|42.84 |	94.37    |
|	             | Chan. (FCA)	| Res. block	  | 26.47	 | 60.98	|  34.09 |	28.64	|36.92 |	89.04    |
|		           |              | ResNext block	| 26.24	 | 64.03	|  34.00 |	29.78	|37.23 |	89.16    |
|		           |              | ResNeSt block	| 35.32	 | 46.30	|  39.27 |	32.51	|40.07 |	93.12    |
|	             | Comb. (add.)	| Res. block	  | 27.61	 | 59.13	|  34.37 |	30.38	|37.64 |	90.30    |
|		           |              | ResNext block	| 29.27	 | 60.16	|  34.79 |	30.89	|39.38 |	89.74    |
|		           |              | ResNeSt block	| 28.03	 | 52.05	|  33.25 |	29.35	|36.44 |	89.93    |
|	             | Comb. (SP)	  | Res. block	  | 19.34	 | 63.42	|  29.66 |	22.91	|29.64 |	85.79    |
|		           |              | ResNext block	| 25.44	 | 59.53	|  32.94 |	28.72	|35.65 |	89.36    |
|		           |              | ResNeSt block	| 7.73	 | 62.48	|  12.97 |	5.18	|13.76 |	60.01    |

![image](https://github.com/ashen007/forest-cover-shrinking/assets/48565752/41c0229d-e779-4b00-a4a2-0e35a9685eb4)

Predictions of the models used attention gate inside the early fusion U-net architecture. (a), (b), (c) shows ResNeSt module with the SE as channel attention in early fusion, ResNeSt module with FCA as channel attention in early fusion and Residual module with additive attention in early fusion respectively. Unlike figure 4.1 these models were able to capture almost all the changed area presented in the ground truth label. aka very small or no green areas in the results.

![image](https://github.com/ashen007/forest-cover-shrinking/assets/48565752/30551352-5509-498a-ab60-158970ed5364)

Predictions of the models used attention gate inside the Siam U-net architecture. (d), (e), (f) shows ResNeSt modules with SE attention gates in Siam architecture, ResNeSt modules with FCA attention gates in Siam architecture and ResNeXt modules with additive attention gate in Siam architecture respectively. (d) shows the best results among these and (a), (b), (c) in the figure 4.2. See table 4.2 for more comprehensive results with the metrics we used.

### Onera satellite change detection dataset

| Type 	         | Network 	         | Prec. |	Recall |	Dice |	Kappa |	Tot.  acc.| 
|----------------|-------------------|-------|---------|-------|--------|-----------|
| Siam network   | FC-EF	           | 44.72 |	53.92	 | 48.89 |	-	    | 94.23     |
| 	             | FC-Siam-conc	     | 42.89 |	47.77	 | 45.20 |	-	    | 94.07     |
| 	             | FC-Siam-diff	     | 49.81 |	47.94	 | 48.86 |	-	    | 94.86     |
| 	             | FC-EF-Res	       | 52.27 |	68.24	 | 59.20 |	-	    | 95.34     |
| 	             | ResNeSt block (SE)| 53.32 |	53.99	 | 59.97 |	43.87	| 97.82     |

### High-resolution semantic change detection dataset

| Type 		     | Network           |	Prec. |	Recall |	Dice | Kappa | Tot.  acc.| 
|--------------|-------------------|--------|--------|-------|-------|-----------|
| Siam network | Str. 1	           | -      |	-	     | 5.56	 | 3.99	 | 86.07     |
|	             | Str. 2	           | -      |	-      | -	   | 21.54 | 98.30     |
|	             | Str. 3	           | -      |	-	     | 13.79 | 12.48 | 94.72     |
|	             | Str. 4.1	         | -      |	-	     | 20.23 | 19.13 | 96.87     |
|	             | Str. 4.2	         | -      |	-	     | 26.33 | 25.49 | 98.19     |
|	             | CNNF-O            | -      |	-	     | 2.43	 | 0.74	 | 64.54     |
|	             | CNNF-F	           | -      |	-	     | 4.84	 | 3.28	 | 88.66     |
|	             | PCA + KM	         | -      | -	     | 2.31	 | 0.67	 | 98.19     |
|	             | ResNeSt block (SE)|15.21	  | 39.18	 | 44.62 | 11.97 | 98.44     |
