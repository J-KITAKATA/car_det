# Vehicle Segmentation on Hailo-8

### The Starting Point
In Japan, traffic surveys are conducted manually. I decided to undertake this project based on two considerations: first, whether it would be possible to reduce the burden on surveyors in light of recent climate change; and second, whether automation might allow us to collect even more detailed data.

### Work Procedure
The work consists of two main parts: learning and conversion to make it work with Hailo-8.

##### Training
During the training process, data was collected and used for training based on the following factors.
* All the data was obtained from photographs.
* We prepared 14 classes and used approximately 5,000 images.
* The model used was the S size of RtmDet-Ins.

The library used was MMDetection, and I installed it without using MIM via the URL below.
https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip

Also, please read the Qiita article below for a detailed explanation.<br>
[Qiita (Japanese)](https://qiita.com/king334/items/edd1e7dcb5068f4f4b22)

##### Inference
After performing the reasoning, this is the result. First, here are the results for the unquantized state.

<img width="50%" alt="20250925_124123_task6" src="https://github.com/user-attachments/assets/a4eb7763-19e4-4795-a79b-bd8e9e049e55" />

