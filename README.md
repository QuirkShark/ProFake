# ProFake
Official code for 'ProFake: Detecting Deepfakes in the Wild against Quality Degradation with Progressive Quality-adaptive Learning'

## Dataset
The detailed code, dataset, and pre-trained model checkpoints used in this paper can be downloaded from: [Google Drive](https://drive.google.com/drive/folders/1AvVby3d9_Td8hCeESrLApN_JdLwIQAOB?usp=sharing)

### Dataset Structure
After downloading, organize your dataset directory to match the structure of the FF++ dataset as follows:
dataset/
├── real/ # Directory for real images
│ ├── 1/ # Subdirectory for real images with ID 1
│ │ ├── 0001.png # Real image with frame index 0001 (4-digit format)
│ ├── 2/ # Subdirectory for real images with ID 2
│ │ ├── 0001.png # Real image with frame index 0001 (4-digit format)
├── fake/ # Directory for fake images
│ ├── 01_23/ # Subdirectory for fake images with ID 01_23 (source_target)
│ │ ├── 0001.png # Fake image with frame index 0001 (4-digit format)



### Inference
To run inference on your images, follow these steps:

1. **Download the pre-trained model**: Get the model from the [Google Drive link](https://drive.google.com/drive/folders/1AvVby3d9_Td8hCeESrLApN_JdLwIQAOB?usp=sharing).
2. **Place the model checkpoint**: Move the downloaded model checkpoint to the `checkpoints/` directory.
3. **Run the inference script**: Execute the following command to run inference on your images:
    ```bash
    python inference.py --input_path path/to/your/image --checkpoint checkpoints/model.pth
    ```
#### Arguments
- `--input_path`: Path to input image or directory containing images
- `--checkpoint`: Path to the model checkpoint