# 🎭 3D Facial Recognition System

This project provides a **deep learning-based facial recognition system** that combines **geometric facial features** (distances, areas, volumes) with **texture analysis** (color, LBP patterns) for robust identity verification using RGB-D images.

## 🧠 Core System Components
The system consists of three core components:

1. **Feature Extraction Pipeline**:  
   - Processes RGB-D images to extract 40 facial features  
   - Calculates normalized distances, areas, and volumes from 3D landmarks  
   - Analyzes texture patterns using LBP and color statistics  

2. **Deep Neural Network**:  
   - 3-layer MLP with BatchNorm and Dropout  
   - Input: 40-dimensional feature vector  
   - Output: 111 identity classes  

3. **Server Module**:  
   - TCP server for real-time recognition  
   - Processes incoming RGB-D frames from clients  
   - Returns predicted identity with confidence score  


## ✨ Key Features

| Component          | Technology Used              | Performance           |
|--------------------|------------------------------|-----------------------|
| Feature Extraction | Face Alignment, Depth Mapping | 40 optimized features |
| Neural Network     | PyTorch (MLP)                | ~95% test accuracy    |
| Server             | Python Socket API            | <50ms inference       |

## 🗂️ Database Structure

The system uses a **CSV-formatted database** containing facial features extracted from 3D RGB-D images. Each row represents a unique facial sample identified by subject and expression, and includes all the features used by the classifier.

### 📄 CSV Format

```csv
identity,expression,filename,dist_ensx_se,dist_ensx_prn,...,lbp_entropy
```

### ✅ Included Fields

| Category              | Description                                                                           |
| --------------------- | ------------------------------------------------------------------------------------- |
| `identity`            | Subject identity                                                                            |
| `expression`          | Facial expression (which is always **N** so neutral)                                               |
| `filename`            | Filename                                                    |
| **3D Distances**      | Euclidean distances between key landmarks (e.g., `dist_ensx_se`, `dist_prn_ls`, etc.) |
| **3D Areas**          | Triangular surface areas from facial landmarks (e.g., `area_prn_alL_alR`)             |
| **3D Volumes**        | Volumes of tetrahedrons formed by 3D landmarks (e.g., `volume_ensx_prn_alL_alR`)      |
| **RGB Texture Stats** | Color statistics: `mean_R/G/B`, `std`, `skewness`, `entropy`                          |
| **HSV Texture Stats** | Mean values for `H`, `S`, and `V` channels                                            |
| **LBP Features**      | Local texture patterns: `lbp_mean`, `lbp_std`, `lbp_entropy`                          |

### 🧾 Sample Row

```csv
bs000,N,bs000_N_N_0.lm3,0.7055,1.5689,0.9211,1.6894,...,0.9696
```

This row represents a sample for subject `bs000` with a neutral facial expression. It includes all **40 numeric features** that are fed into the neural network for identity recognition.


## 🛠️ Setup Guide

### 📋 Requirements

```bash
pip install torch face-alignment scikit-learn opencv-python matplotlib
````



### ⚡ Quick Start

1. **Run the system**:

```bash
python main.py
```

This will automatically:

* Load the pre-trained model (`best_model.pth`)
* Start the TCP server listening on port `5005`


### 🌐 Server Details

* **Address**: `0.0.0.0:5005`
* **Protocol**: TCP
* **Data Format**:

  1. 4 bytes indicating PNG size
  2. PNG image data (RGB)
  3. RAW depth map data (uncompressed)

> 🧠 **Note**: The model is already trained and ready to use.
> If you want to retrain it, simply uncomment `run_training()` in `main.py`.

## 🚀 How to Run the System

1. **Train the Model (Optional)**  
   If you want to train the classifier from scratch, uncomment `run_training()` and `run_test()` inside `main.py`.  
   The script uses features extracted from RGB-D images and supports both default and custom datasets.  
   👉 To skip training, simply use the pre-trained model provided at `models/best_model.pth` and move to step 2.

2. **Select the Input Mode**  
   Before running the server, choose your operating mode by editing the image flip setting in `server.py`:

   ```python
   # FLIP PNG - Comment/Uncomment based on your needs:
   ''' 
   png = Image.open(io.BytesIO(png_data))
   png_flip = png.transpose(Image.FLIP_TOP_BOTTOM)  # Vertical flip
   '''
    ````

* ✅ **Keep commented** for static mode
* 🔄 **Uncomment** for live RealSense camera input so for RealTime mode

3. **Start**
   Launch the server and load the model with:

   ```bash
   python main.py
   ```

4. **Open the Unity Scene**
   Copy the `Assets` folder into a new Unity project and open it.
   Press **Play** to run the scene. Unity communicates with the Python backend via **UDP**.

5. **Use the System via Unity Interface**
   From the Unity menu, choose your desired mode of interaction:

   * **🖼️ Manual Mode**
     Click **Connect**, select a previously acquired PNG and RAW depth pair, then press **Identify** to run prediction.

   * **📡 Real-Time Mode**
     Click **Connect** to link with the server. Once the RealSense stream is ready, press **Identify** to perform real-time recognition.
