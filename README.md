```markdown
# Text-to-Image Deep Learning Pipeline

This repository contains the complete pipeline for a Deep Learning project that generates images from textual descriptions—“text-to-image.” The project was developed and tested on Google Colab, using the Flickr8k dataset for training a dual‐branch model and demonstrating inference and deployment via Gradio. In cases where training could not be completed on free Colab due to hardware/time limitations, a separate notebook provides a Stable Diffusion–based simulation.

---

## 🚀 Project Overview

- **Objective:** Given a natural-language description (e.g. “a white cat wearing glasses”), generate or retrieve a corresponding image.
- **Pipeline Steps:**
  1. **Data Collection**  
     - Subset of [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) (2 000 image-text pairs).
  2. **Preprocessing**  
     - Resize images to 224×224, normalize pixel values.  
     - Tokenize captions with BERT tokenizer, extract `input_ids` and `attention_mask`.
  3. **Model Architecture**  
     - **Image branch:** Pretrained ResNet50 (last fully connected layer replaced with a 256-dimensional projection).  
     - **Text branch:** Pretrained BERT (pooler output projected into 256-dimensional embedding).  
     - Both embeddings are compared with a cosine‐based contrastive loss: matching pairs should be close in the 256-dimensional space.
  4. **Training**  
     - Run on Google Colab (Free Tier).  
     - Limitations: sessions often disconnect after 30–45 minutes, GPU memory is limited.  
     - We train for 1–3 epochs on a 2 000-example subset, saving checkpoints to Google Drive after each epoch.  
     - If a session disconnects, simply reload the last checkpoint and continue (checkpoint files are stored under `drive/MyDrive/text_to_image_project/`).
  5. **Simulation (Separate Notebook)**  
     - Because a full training run on free Colab is not always feasible, we provide a separate notebook using a pretrained Stable Diffusion pipeline to simulate “text-to-image” generation.  
     - That notebook shows how to call `StableDiffusionPipeline` from Hugging Face and deploy an interactive Gradio demo.
  6. **Deployment with Gradio**  
     - The “trained” dual‐branch model (or whichever checkpoint is available) is loaded.  
     - We precompute image embeddings for the 2 000-example subset.  
     - Given a new caption, compute its text embedding, then find the most similar image embedding (cosine similarity).  
     - A minimal Gradio interface lets users type a description and view the closest‐matching image.

---

## 📁 Repository Structure

```

.
├── README.md
├── notebooks
│   ├── training\_pipeline.ipynb
│   └── simulation\_stablediffusion.ipynb
├── data
│   └── flickr8k\_captions.csv          # Subset CSV of 2 000 image‐caption pairs
├── src
│   ├── dataset.py                     # FlickrDataset class definition
│   ├── model.py                       # DualBranchModel definition
│   ├── train.py                       # Training script for dual-branch model
│   └── inference\_gradio.py            # Gradio deployment code
└── requirements.txt

```

- **`notebooks/training_pipeline.ipynb`:** Jupyter Notebook that implements steps 1–4 and 6 (excluding Stable Diffusion simulation). Contains Markdown cells synchronized with the 7-minute presentation script.
- **`notebooks/simulation_stablediffusion.ipynb`:** Separate Notebook showing how to use a pretrained Stable Diffusion model for text-to-image simulation and Gradio deployment.
- **`data/flickr8k_captions.csv`:** A CSV containing 2 000 randomly sampled lines from the original Flickr8k token file. Each row has `image_filename` and `caption`.
- **`src/dataset.py`:** Contains the `FlickrDataset` class for PyTorch DataLoader.
- **`src/model.py`:** Defines `DualBranchModel` (ResNet50 + BERT).
- **`src/train.py`:** Standalone Python script (compatible with Google Colab) to train the dual-branch model and save checkpoint files to Google Drive.
- **`src/inference_gradio.py`:** Script that loads the latest checkpoint, precomputes all image embeddings, and launches a Gradio interface to retrieve the closest matching image for any input caption.
- **`requirements.txt`:** Lists Python package dependencies:
```

torch
torchvision
transformers
pandas
gradio
datasets
pillow

````

---

## 🔧 Setup & Installation

1. **Clone this repository**  
 ```bash
 git clone https://github.com/yourusername/text-to-image-pipeline.git
 cd text-to-image-pipeline
````

2. **Install dependencies** (ideally in a virtual environment)

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Flickr8k data**

   * Place the image folder (e.g., `Flickr8k_Dataset/Images/`) under `./data/`.
   * Place `Flickr8k_text/Flickr8k.token.txt` (captions file) under `./data/`.
   * Run the CSV‐extraction snippet below to create `data/flickr8k_captions.csv` (2 000 random samples).

   ```python
   import pandas as pd

   # Modifier le chemin si besoin
   df = pd.read_csv("data/Flickr8k_text/Flickr8k.token.txt", sep="\t", names=["image", "caption"])
   df_subset = df.sample(n=2000, random_state=42).reset_index(drop=True)
   df_subset.to_csv("data/flickr8k_captions.csv", index=False)
   ```

4. **(Optionnel) Mount Google Drive**

   * If you plan to run on Google Colab, add a cell in your notebook to mount Drive:

     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   * Ensure you have a folder like `drive/MyDrive/text_to_image_project/` for saving model checkpoints.

---

## ▶️ How to Run

### 1. Training Pipeline (On Colab or Local GPU)

* **Using Jupyter Notebook**

  1. Open `notebooks/training_pipeline.ipynb`.
  2. Exécutez les cellules une à une.

     * Montez Google Drive (si sur Colab).
     * Chargez `data/flickr8k_captions.csv` et le dossier d’images.
     * Prétraitez, créez `DataLoader`.
     * Définissez `DualBranchModel` et lancez l’entraînement.
     * Les checkpoints seront sauvegardés automatiquement dans votre Drive.

* **En Script Python (Local)**

  ```bash
  python src/train.py \
    --captions_file data/flickr8k_captions.csv \
    --image_dir data/Flickr8k_Dataset/Images \
    --save_dir path/to/save/checkpoints \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 1e-4
  ```

### 2. Inference & Deployment with Gradio

* **Jupyter Notebook**

  1. Ouvrez `notebooks/training_pipeline.ipynb` (ou `src/inference_gradio.py` dans un notebook).
  2. Chargez le dernier fichier checkpoint (`dual_model_epoch3.pth` par exemple).
  3. Exécutez la cellule de pré-calcul d’embeddings d’images.
  4. Lancez la cellule Gradio : un lien public s’affichera pour tester l’interface web.

* **Script Python**

  ```bash
  python src/inference_gradio.py \
    --checkpoint path/to/dual_model_epoch3.pth \
    --captions_file data/flickr8k_captions.csv \
    --image_dir data/Flickr8k_Dataset/Images \
    --batch_size 16 \
    --port 7860
  ```

  * Rendez-vous sur `http://localhost:7860` pour tester la génération.

### 3. Simulation avec Stable Diffusion

* Ouvrez `notebooks/simulation_stablediffusion.ipynb`.
* Exécutez les cellules pour installer `diffusers`, charger `StableDiffusionPipeline`, saisir un prompt textuel et générer une image.
* La démonstration Gradio y est incluse pour voir la génération en direct.

---

## 📚 Détails Techniques

* **FlickrDataset (src/dataset.py)**

  * Hérite de `torch.utils.data.Dataset`.
  * Lit les images depuis un dossier local, applique `transforms` pour redimensionner/normer.
  * Tokenise les légendes avec `BertTokenizer`.
  * Retourne un triplet `(image_tensor, input_ids, attention_mask)` pour chaque exemple.

* **DualBranchModel (src/model.py)**

  * `image_encoder` : ResNet50 préentraîné, modification de la dernière couche pour produire un vecteur `(batch_size, 256)`.
  * `text_encoder` : BERT préentraîné, utilisation du `pooler_output` (CLS) projeté en 256 dims.
  * Forward pass : renvoie `(image_embed, text_embed)`.

* **Loss et Optimizer**

  * Loss : `CosineEmbeddingLoss` avec `targets = +1` (les paires sont positives).
  * Optimizer : `Adam(model.parameters(), lr=1e-4)`.

* **Gestion des Checkpoints**

  * Après chaque epoch, on sauvegarde `model.state_dict()` dans `save_dir`.
  * Pour reprendre, on charge `model.load_state_dict(torch.load(checkpoint_path))`.

* **Inference & Gradio (src/inference\_gradio.py)**

  1. Charger le checkpoint.
  2. Pré-calculer tous les embeddings d’images (dictionary en mémoire).
  3. Faire forward d’un texte pour obtenir son embedding.
  4. Comparer à chaque `image_embedding` (similarité cosine).
  5. Renvoyer l’image au plus grand score.
  6. Gradio : simple interface “Textbox → Image”.

---

## 📝 Remarques et Conseils

* **Limites de Google Colab (Free Tier)**

  * Sessions instables : sauvegardez souvent.
  * GPU mémoire limitée : réduisez la taille du dataset ou la dimension de batch.
  * Pensez à utiliser Colab Pro si vous avez besoin d’entrainer plus longuement.

* **Extensibilité**

  * Vous pouvez facilement remplacer le ResNet50 par un autre vision‐encoder (ex. EfficientNet).
  * Remplacez BERT par un modèle de plus petite taille (ex. DistilBERT) pour accélérer.
  * Ajoutez un index de recherche (Faiss, Annoy) pour une recherche d’images à grande échelle.

* **Qualité de l’Alignement**

  * La loss contrastive simple ne garantit pas un alignement parfait : on peut explorer des pertes plus sophistiquées (Triplet‐loss, InfoNCE, CLIP‐style).

---

## 🖇️ Liens Utiles

* Flickr8k dataset : [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
* Hugging Face Diffusers (Stable Diffusion) : [https://huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
* Gradio : [https://gradio.app/](https://gradio.app/)
* PyTorch Documentation : [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
* Hugging Face Transformers : [https://huggingface.co/docs/transformers/index.html](https://huggingface.co/docs/transformers/index.html)

---

## 🤝 Contributions et Licences

* **Contributions** : Pull requests bienvenues. Veuillez forker le dépôt, créer une branche de fonctionnalité, puis proposer un pull request.
* **Licence** : Ce projet est distribué sous la licence MIT. Voir le fichier [LICENSE](./LICENSE) pour plus de détails.

---

Merci d’avoir consulté ce projet ! N’hésitez pas à ouvrir une “issue” pour poser des questions ou suggérer des améliorations.
