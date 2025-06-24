# System analysis Workshops
<small>Juan David Buitrago Rodriguez - 20242020194</small>
<br>
<small>David Giovanni Aza Carvajal - 20241020137</small>

- [Workshop I](#workshop-i)
- [Workshop II](#workshop-ii)

---

# Workshop III 

## 💉 Blood Vessel Simulation - Cellular Automaton in Python

This project simulates the growth and behavior of blood vessels using a **cellular automaton** built with **Python** and **Tkinter**. It models arteries, healthy vessels, aneurysms, cell death, hypoxia, and vessel regeneration.

---

## 🧪 Simulation Rules

| Cell Type      | Code | Color         | Behavior                                                                                                                                   |
|----------------|------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **Artery**     | `A`  | Red           | - Constantly generates vessels in all cardinal directions (up, down, left, right).<br> - Always surrounded by live vessels.               |
| **Live Vessel**| `V`  | Blue (darker with more vessel neighbors, lighter with fewer) | - Propagates to adjacent empty cells (cardinal directions) with probability.<br> - Dies if isolated (no adjacent vessels).<br> - Becomes aneurysm if ≥ 5 vessel/artery neighbors (in 8 directions). |
| **Aneurysm**   | `X`  | Orange        | - Explodes if ≥ 2 aneurysm neighbors (8 directions).<br> - Explosion destroys vessels within a radius (`EXPLOSION_RADIUS`).<br> - Cures back to vessel if surrounded by < 4 vessels. |
| **Dead Vessel**| `D`  | Black         | - Can regenerate into a vessel if it has ≥ 1 live vessel/artery neighbor (cardinal directions).                                            |
| **Empty**      | `T`  | White         | - May become a vessel if near an artery or through vessel propagation.                                                                   |
| **Hypoxia**    | `H`  | Yellow        | - Visual marker; not yet used in simulation logic.                                                                                        |

---

## ⚙️ Global Parameters

The following parameters can be easily modified to tune the behavior:

```python
EXPLOSION_RADIUS = 2           # Radius of aneurysm destruction
PROPAGATION_PROB = 0.3         # Probability that a vessel creates a new vessel nearby
ANEURISM_THRESHOLD = 5         # Neighbor count (8 directions) to trigger aneurysm
CURE_THRESHOLD = 4             # If an aneurysm has fewer neighbors than this, it heals
DEAD_CURE_NEIGHBORS = 1        # Number of live neighbors needed to revive a dead vessel
```

---

## 🖱️ How to Use

1. When you launch the simulation, an empty grid will appear.
2. Use the buttons to select a cell type:
   - Artery (A)
   - Vessel (V)
   - Aneurysm (X)
   - Dead (D)
   - Empty (T)
   - Hypoxia (H)
3. Click on the grid to place cells.
4. Click **Start Simulation** to begin.
5. Use **Step** to advance the simulation manually.
6. Click **Reset** to clear the board and start over.

---

## 📦 Requirements

- Python 3.x
- Tkinter (usually pre-installed with Python)

---


# Workshop II 

📄 **[Read Full Report](./Workshop_2_Design/Workshop_II.pdf)**

Following the initial analysis phase, this workshop focuses on the implementation, evaluation, and iterative refinement of a deep learning model for histological image segmentation.

### 🧠 System Analysis Techniques

In this project, we used **system analysis** to better understand the problem and design the solution step by step. These are some of the techniques we applied:


- **Component Mapping:** We broke the project down into parts like: input data, preprocessing, model, output masks, and evaluation metrics.
- **Pipeline Visualization:** We created flowcharts and diagrams to see how the data moves through the model (from raw image to prediction).
- **Reference Analysis:** We studied successful public Kaggle notebooks and a key scientific paper that helped us understand how CNNs work in histological image segmentation.

---

### 🧬 CNN Implementation/Analysis Process

To build our Convolutional Neural Network (CNN) for classifying histological kidney images, we followed a structured process:

1. **Studied the Domain:** We analyzed histology slide structure and annotation formats, focusing on `.tif` images containing kidney tissue regions.

2. **Reviewed Scientific Literature:** We consulted key sources such as [Deep Learning for Semantic Segmentation in Histology](https://www.mdpi.com/2078-2489/16/3/195), which helped us understand histological image processing and neural network architectures relevant to biomedical tasks. Although the paper focuses on segmentation, we adapted its insights for a classification task.

3. **Implemented the Model:** We built a custom classifier using **ResNet50** enhanced with **CBAM (Convolutional Block Attention Module)**. Starting from ImageNet-pretrained weights, the model was fine-tuned to identify three classes in kidney histology tiles: *Glomerulus*, *Blood Vessel*, and *Unsure*. The addition of CBAM helps the network focus more effectively on meaningful spatial and channel-wise features, improving its ability to capture subtle patterns in the tissue.


4. **Prepared the Data:** Whole-slide `.tif` images were divided into 512×512 tiles. Each tile was preprocessed using resizing and normalization, then passed through the model for classification.

The model workflow is as follows:

- A `.tif` image is split into fixed-size tiles.
- Each tile is preprocessed and passed into the ResNet50_CBAM model.
- The model predicts class probabilities using a softmax layer.
- Predictions per tile are printed and optionally stored in a table for analysis.

This pipeline provides a fast and scalable method for classifying high-resolution histology data using deep learning and attention-enhanced CNNs.

5. **Applied the Concepts:** In the **[/code](./Workshop_2_Design/code)** folder, we tried to apply what we learned by creating a custom CNN Implementation. This involved testing out different architectural changes and making adjustments to improve the system.


📘 **[Return to Report](./Workshop_2_Design/Workshop_II_Report.pdf)**


---



# Workshop I 


## 🔍 Analysis 

📄 **[Read Full Report](./Workshop-I/Workshop_I_Report.pdf)**

The analysis conducted for this report was structured into four key phases, each with a specific goal:

---

###  Phases 

- **1. Data & Overview Analysis**
  - 📚 **Objective:** Understand the dataset and initial data structure.
  -  **Actions:**
    - Explored the `.tif` image files and metadata.
    - Reviewed competition goals and objectives.
    - The dataset for this competition is located in `/Workshop-I/data/`, exepting the folder `/test` and `/train` folders  that contain the `.tif` images  due to their size (>4GB). These files can be downloaded directly from [Kaggle’s competition page](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/data).

- **2. Thematic Research**
  - 📚 **Objective:** Investigate domain-specific knowledge.
  -  **Topics Covered:**
    - Histological tissues and blood vessel anatomy.
    - WSI (Whole Slide Imaging) concepts.
    - Role of masks and annotations in biomedical imaging.
    - How Kaggle competitions are typically structured.

- **3. Competition Notebooks Lookup**
  - 📚 **Objective:** Gain insights from public solutions.
  -  **Activities:**
    - Reviewed notebooks on Kaggle.
    - Analyzed preprocessing and visualization techniques.
    - Observed modeling strategies for mask handling.
    - A thorough review of public notebooks related to the competition was conducted, we made particular focus on the [notebook by Ahmed Maher El-Saeidy](https://www.kaggle.com/code/ahmedmaherelsaeidy/hubmap-hacking-the-human-vasculature-dataset). The code from this notebook was used as a reference and implemented under the purpose of understanding the dataflow and how the elements interacted among themselves. This implementation can be found in the folder `/Workshop-I/code`.

- **4. System Comprehension & Analysis**  
   After gathering all the information from the previous phases, we structured the system analysis by mapping out how components interact within the pipeline, summarizing key insights, and formulating visual overviews. All these findings were integrated into our final **[report](./Workshop-I/Workshop_I_Report.pdf)**.

---

📘 **[Return to Report](./Workshop-I/Workshop_I_Report.pdf)**


