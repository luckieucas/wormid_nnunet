# 📌 Installation

Run the following command to install all required packages:
```bash
pip install -r requirements.txt
```

---


## **Usage**

### **1. Generate heatmap**
Generate a heatmap from centroids by running:

```bash
python generate_heatmap_from_centroids.py --input_folder input_dir --output_folder output_dir
```
Replace input_dir with the path to your folder containing the h5 files, and output_dir with the folder where the generated heatmap files should be saved.

### **2. Split data into 5 folds**
This step splits your dataset into 5 folds for cross-validation. 

### **3. Train model**
Config your data directory in `configs/config.yaml`
```bash
python train.py -c configs/config.yaml -f 0
```
`-f` specifies which fold to train, can be `0-4` for 5-fold cross-validation.

### **4. Predict heatmap**
```bash
python test.py --input test_img_folder --output_dir heatmap_save_folder --model_path model_for_current_path
```

### **5. Extract centroid from heatmap**
```bash
python extract_centroid_from_heatmap.py --input_folder heatmaps_pred_path --output_folder centroid_saved_path --gt_folder [optional] /mmfs1/data/liupen/project/dataset/nuclei/wormID_data/centroids
```
## Model Download
Please download the trained model from the following link:
[Google Drive Link](https://drive.google.com/drive/folders/1sgzSQedq4XLkzIfjd0OI0Zc23YGo1-WA?usp=drive_link)