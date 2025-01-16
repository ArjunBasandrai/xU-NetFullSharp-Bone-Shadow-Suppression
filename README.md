<h1 align="center">xU-NetFullSharp Chest XRay Bone Shadow Suppression</h1>

<p align="center">
  PyTorch implementation of <a href="https://www.sciencedirect.com/science/article/abs/pii/S1746809424010413">"xU-NetFullSharp: The Novel Deep Learning Architecture for Chest X-ray Bone Shadow Suppression"</a> trained on the <a href="https://www.kaggle.com/datasets/hmchuong/xray-bone-shadow-supression">Augmented JSRT and BSE-JSRT</a> dataset
</p>

# Model Architecture

The model is built using the xU-NetFullSharp architecture described in the paper

<img src="https://github.com/ArjunBasandrai/xU-NetFullSharp-Bone-Shadow-Suppression/blob/main/images/xU-NetFS_EN.jpg"/>

## Training Data

The model is trained on the <a href="https://www.kaggle.com/datasets/hmchuong/xray-bone-shadow-supression">Augmented JSRT and BSE-JSRT</a> datatset available on Kaggle

## CXR Bone Shadow Suppression Results

![image](https://github.com/user-attachments/assets/5356e601-44cf-4265-8fb0-fb97596d38d7)
![image](https://github.com/user-attachments/assets/9a14ead5-38ba-432c-ba29-6a58bf674d7a)
![image](https://github.com/user-attachments/assets/fbbcc0f6-720e-4dd5-b685-14c39c9adf9d)
![image](https://github.com/user-attachments/assets/7a5c0494-400f-4b68-a0f5-29cf4f4d2eef)
![image](https://github.com/user-attachments/assets/dc884845-60fc-4e80-8c27-29b9455eab79)

The trained model is able to accurately reconstruct the finer details inside the lungs

![image](https://github.com/user-attachments/assets/ed1640ce-0d62-4791-ad9c-30d84630d517)
![image](https://github.com/user-attachments/assets/42821811-2009-4227-b175-1d825965b526)
![image](https://github.com/user-attachments/assets/8578440c-bbf0-4e8e-8168-c68b73b93f2e)

## Usage

1. Clone the repository using
```bash
git clone https://github.com/ArjunBasandrai/xU-NetFullSharp-Bone-Shadow-Suppression.git
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run `main.py`
```bash
python main.py --model_weights models/model.pth --input_image /path/to/input/image --output_image /path/to/output/image
python main.py --model_weights models/model.pth --input_image /path/to/input/image --output_image /path/to/output/image --use_cmap True
python main.py --model_weights models/model.pth --input_image /path/to/input/image --output_image /path/to/output/image --use_cmap True --no_limit_shape True
```

## References

1. Schiller, V., Burget, R., Genzor, S., Mizera, J., & Mezina, A. (2025). xU-NetFullSharp: The novel deep learning architecture for chest X-ray bone shadow suppression. Biomedical Signal Processing and Control, 100(Part B), Article 106983. https://doi.org/10.1016/j.bspc.2024.106983

2. Kligvasser, I., Shaham, T. R., & Michaeli, T. (2018). xUnit: Learning a spatial activation function for efficient image restoration. Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.48550/arXiv.1711.06445

3. Shiraishi, J., Katsuragawa, S., Ikezoe, J., Matsumoto, T., Kobayashi, T., Komatsu, K. I., Matsui, M., Fujita, H., Kodera, Y., & Doi, K. (2000). Development of a digital image database for chest radiographs with and without a lung nodule: Receiver operating characteristic analysis of radiologists’ detection of pulmonary nodules. American Journal of Roentgenology, 174(1), 71–74. https://doi.org/10.2214/ajr.174.1.1740071
