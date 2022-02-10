# ShiKangdi-ShiKangdi-Image-Retrieval-via-Canonical-Correlation-Analysis-and-Binary-Hypothesis-Testing
Codes for proposed approach in "Image Retrieval via Canonical Correlation Analysis and Binary Hypothesis Testing

Training_Data_preparation.py: For feature extraction on training dataset, it saves each type of features as ''Max_Rshape_P5.npy', 'Ave_Rshape_P5.npy', 'Std_Rshape_P5.npy', and also save names for each training sample as 'NameList.npy'

Test_Data_preparation.py: For feature extraction on training dataset, it saves each type of features as ''Eva_Max_Rshape_P5.npy', 'Eva_Ave_Rshape_P5.npy', 'Eva_Std_Rshape_P5.npy', and also save names for each training sample as 'Eva_NameList.npy'.

CCA_preparation.py: For each type of features from training dataset, It respectively save matching and non-matching coefficients as "Max_Cm.py", "Ave_Cn.py", "Std_Cm.py", "Ave_Cn.py", "Ave_Cm.py", "Std_Cn.py", and canoncial vectors as "Max_MEV.py", "Ave_MEV.py", "Std_MEV.py".

LDA_preparation.py:For each type of features from training dataset, It respectively save between-class scatter matrix as "max_Sb.py","ave_Sb.py","std_Sb.py", and within-class scatter matrix as "max_Sw.py","ave_Sw.py","std_Sw.py".
