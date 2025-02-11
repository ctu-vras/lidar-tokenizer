# LiDAR Tokenizer
This is a repository containing code used in the Master's thesis [*LiDAR Tokenizer*](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://dspace.cvut.cz/bitstream/handle/10467/120372/F3-DP-2025-Herold-Adam-master-thesis-final.pdf&ved=2ahUKEwj74sXuqruLAxWFg_0HHbE6DysQFnoECBgQAQ&usg=AOvVaw3gErIWidQqKhFja6VfvqWG) by Adam Herold written in 09/2024-01/2025.
It contains two packages:
 - **LiDAR_Tokenizer** contains everything regarding the training and evaluation of two LiDAR tokenizer models.
 - **LiDAR_MaskGIT** contains a modified multimodal MaskGIT model for image-LiDAR point cloud synthesis.

LiDAR_Tokenizer works independently but LiDAR_MaskGIT requires LiDAR_Tokenizer to fully work. Originally, each package had its own repository and here they are together in their final cleaned-up versions just so that everything related to the thesis is at one place. Both of the packages contain their own README files with instruction on how to reproduce the experiments described in the thesis and with credits to other repositories I used to create them.
