# 🚗 Vehicle_MEX_Model — Fine-Grained Vehicle Classification (YOLO-CLS)

<div align="center">
  <img src="assets/train_batch0.jpg" width="31%" alt="Vehicle_MEX training batch 0"/>
  <img src="assets/train_batch1.jpg" width="31%" alt="Vehicle_MEX training batch 1"/>
  <img src="assets/train_batch2.jpg" width="31%" alt="Vehicle_MEX training batch 2"/>
</div>

<div align="center">

[![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLO-111827)](https://github.com/ultralytics/ultralytics)
[![Task](https://img.shields.io/badge/Task-Image%20Classification-2563EB)](#-overview)
[![Input](https://img.shields.io/badge/Input-640x640-0EA5E9)](#-technical-specs)
[![Tuning](https://img.shields.io/badge/HPO-Ray%20Tune%20%2B%20Optuna-7C3AED)](#-hyperparameter-tuning-hpo)

</div>

## 📌 Overview
Este repositorio documenta el entrenamiento de un **modelo baseline** para **clasificación de vehículos** (fine-grained) usando **Ultralytics YOLO (modo classify)** sobre el dataset `Vehicle_MEX_Dataset`.  
Incluye scripts reproducibles, notebooks de aumento de datos, métricas, y visualizaciones (sin subir el dataset completo de imágenes).

## 📦 Dataset
- **Dataset original (descarga):** https://data.mendeley.com/datasets/gbk6gnv245/1
- **Estructura esperada para YOLO-CLS:**

```text
Vehicle_classification.v1/
├── train/
│   ├── B2/
│   ├── B3/
│   └── ...
├── valid/
│   ├── B2/
│   ├── B3/
│   └── ...
└── test/
    ├── B2/
    ├── B3/
    └── ...
```

## 🎯 Objectives
- ✅ Entrenar un **baseline reproducible** para clasificación de vehículos.
- ✅ Identificar errores por clase usando **matriz de confusión**.
- ✅ Proponer y ejecutar mejoras con **regularización** y **aumento de datos**.
- ✅ Preparar un flujo opcional de **búsqueda de hiperparámetros (HPO)** para acelerar la convergencia y mejorar precisión.

## 🧠 Method (Training Pipeline)
**Etapas del flujo:**
1. **Preprocesamiento y balanceo** con `notebooks/Data_augmenter.ipynb`
2. Entrenamiento con **YOLO-CLS** (Ultralytics)
3. Validación y análisis (matrices de confusión + curvas de entrenamiento)
4. Ajustes manuales e iteración
5. (Opcional) **HPO con Ray Tune + Optuna** para encontrar mejores hiperparámetros

## 📈 Results (Baseline_model-3)
Resultados reportados en `results/Terminal.txt` y visualizados en `results/`.

- **Top-1 Accuracy:** 0.7246
- **Top-5 Accuracy:** 0.8245
- **Best epoch (early stopping):** 11

<div align="center">
  <img src="results/results.png" width="80%" alt="Training curves"/>
</div>

### 🔎 Confusion Matrix
<div align="center">
  <img src="results/confusion_matrix.png" width="48%" alt="Confusion matrix"/>
  <img src="results/confusion_matrix_normalized.png" width="48%" alt="Confusion matrix normalized"/>
</div>

## 🎞️ Demo (Training Batches)
<div align="center">
  <img src="assets/demo.gif" width="80%" alt="Training batches demo"/>
</div>

## 🧩 Project Structure
```text
Vehicle_MEX_Model/
├── assets/                      # Imágenes/GIF ligeros para README (no dataset completo)
├── configs/                     # Configuraciones usadas en entrenamientos
├── notebooks/                   # Notebooks de análisis/augmentación
├── results/                     # Curvas, matrices, logs, CSV de resultados
├── scripts/                     # Scripts de entrenamiento y HPO
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Installation
### 1) Clonar
```bash
git clone <TU_URL_DE_GITHUB>/Vehicle_MEX_Model.git
cd Vehicle_MEX_Model
```

### 2) Crear entorno (recomendado)
```bash
python -m venv .venv
```

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/Mac**
```bash
source .venv/bin/activate
```

### 3) Instalar dependencias
```bash
pip install -r requirements.txt
```

## 🚀 Training
Script principal:
- `scripts/1_Classification.py`

Ejemplo:
```bash
python scripts/1_Classification.py
```

## 🧪 Dataset Augmentation & Balancing
Notebook:
- `notebooks/Data_augmenter.ipynb`

Incluye:
- selector de carpetas (local)
- análisis automático de desbalance por clase
- previsualización de augmentations
- export a estructura YOLO-CLS con imágenes 640x640 (padding + Lanczos)

## 🧬 Hyperparameter Tuning (HPO)
Script:
- `scripts/1_Classification_Optuna.py`

Este flujo usa `model.tune(..., use_ray=True)` (Ray Tune) y permite búsqueda tipo Optuna.  
Requiere instalar dependencias adicionales:
```bash
pip install -U "ray[tune]" optuna
```

## ✨ Main Features
- 🧱 Reproducibilidad: logs + configs (`configs/`, `results/`)
- 🧰 Aumentos realistas orientados a vehículos (rotación leve, perspectiva, brillo/contraste, ruido)
- 🧯 Regularización agresiva (dropout, weight decay, warmup)
- 🧪 HPO opcional con Ray Tune + Optuna

## 🧾 Technical Specs
- **Framework:** Ultralytics YOLO (classification)
- **Input:** 640×640 (con padding para conservar proporciones)
- **Optimizer:** AdamW
- **Scheduler:** Cosine LR (si se habilita en los scripts)

## 🏙️ Applications
- 🚦 Clasificación automática de tipos de vehículos en escenarios urbanos y carreteros
- 🛣️ Analítica de tráfico por categoría
- 🧾 Inventario vehicular y monitoreo en infraestructura
- 🔍 Soporte a sistemas ITS (Intelligent Transportation Systems)

## 🧑‍🔬 Research Team
Formato inspirado en: https://github.com/JaGuzmanT/SaltSpot

<table align="center">
  <thead>
    <tr>
      <th align="center" width="120">Photo</th>
      <th align="left">Researcher</th>
      <th align="left">Affiliation</th>
      <th align="left">Contact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" width="120">
        <img src="https://raw.githubusercontent.com/JaGuzmanT/SaltSpot/main/Public/docs/Images/JAGT1.jpg" alt="Dr. José Alberto Guzmán Torres" width="96" height="96">
      </td>
      <td>
        <b>Dr. José Alberto Guzmán Torres</b> :mexico:
        <sub>Engineering Applications &amp; Artificial Intelligence</sub>
      </td>
      <td>
        <a href="http://www.siiia.com.mx"><img alt="Company: SIIIA MATH" src="https://img.shields.io/badge/%F0%9F%8F%A2%20Company-SIIIA%20MATH-0B1B3A"></a>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:jose.alberto.guzman@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
        <a href="https://orcid.org/0000-0002-9309-9390"><img alt="ORCID 0000-0002-9309-9390" src="https://img.shields.io/badge/ORCID-0000--0002--9309--9390-green"></a>
        <a href="https://www.researchgate.net/profile/Jose-Guzman-Torres"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="https://raw.githubusercontent.com/JaGuzmanT/SaltSpot/main/Public/docs/Images/FJDM1.jpg" alt="Dr. Francisco Javier Domínguez Mota" width="96" height="96">
      </td>
      <td>
        <b>Dr. Francisco Javier Domínguez Mota</b> :mexico:
        <sub>Applied Mathematics &amp; Finite Difference Methods</sub>
      </td>
      <td>
        <a href="http://www.siiia.com.mx"><img alt="Company: SIIIA MATH" src="https://img.shields.io/badge/%F0%9F%8F%A2%20Company-SIIIA%20MATH-0B1B3A"></a>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:francisco.mota@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
        <a href="https://orcid.org/0000-0001-6837-172X"><img alt="ORCID 0000-0001-6837-172X" src="https://img.shields.io/badge/ORCID-0000--0001--6837--172X-green"></a>
        <a href="https://www.researchgate.net/profile/Francisco-Dominguez-Mota"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="https://raw.githubusercontent.com/JaGuzmanT/SaltSpot/main/Public/docs/Images/EMAG.jpg" alt="Dra. Elia M. Alonso Guzmán" width="96" height="96">
      </td>
      <td>
        <b>Dra. Elia M. Alonso Guzmán</b> :mexico:
        <sub>Civil Engineering &amp; Materials Science</sub>
      </td>
      <td>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:elia.alonso@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
        <a href="https://orcid.org/0000-0002-8502-4313"><img alt="ORCID 0000-0002-8502-4313" src="https://img.shields.io/badge/ORCID-0000--0002--8502--4313-green"></a>
        <a href="#"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="https://raw.githubusercontent.com/JaGuzmanT/SaltSpot/main/Public/docs/Images/GTG.jpg" alt="Dr. Gerardo Tinoco Guerrero" width="96" height="96">
      </td>
      <td>
        <b>Dr. Gerardo Tinoco Guerrero</b> :mexico:
        <sub>Numerical Methods &amp; Computational Mathematics</sub>
      </td>
      <td>
        <a href="http://www.siiia.com.mx"><img alt="Company: SIIIA MATH" src="https://img.shields.io/badge/%F0%9F%8F%A2%20Company-SIIIA%20MATH-0B1B3A"></a>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:gerardo.tinoco@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
        <a href="https://orcid.org/0000-0003-3119-770X"><img alt="ORCID 0000-0003-3119-770X" src="https://img.shields.io/badge/ORCID-0000--0003--3119--770X-green"></a>
        <a href="https://www.researchgate.net/profile/Gerardo-Tinoco-Guerrero"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="https://raw.githubusercontent.com/JaGuzmanT/SaltSpot/main/Public/docs/Images/JGTR1.jpg" alt="Dr. José Gerardo Tinoco Ruíz" width="96" height="96">
      </td>
      <td>
        <b>Dr. José Gerardo Tinoco Ruíz</b> :mexico:
        <sub>Applied Mathematics &amp; Computational Modeling</sub>
      </td>
      <td>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:jose.gerardo.tinoco@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
        <a href="https://orcid.org/0000-0002-0866-4798"><img alt="ORCID 0000-0002-0866-4798" src="https://img.shields.io/badge/ORCID-0000--0002--0866--4798-green"></a>
        <a href="#"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="120">
        <img src="https://raw.githubusercontent.com/JaGuzmanT/SaltSpot/main/Public/docs/Images/Harias.webp" alt="Dr. Heriberto Árias Rojas" width="96" height="96">
      </td>
      <td>
        <b>Dr. Heriberto Árias Rojas</b> :mexico:
        <sub>Engineering Applications</sub>
      </td>
      <td>
        <a href="http://www.siiia.com.mx"><img alt="Company: SIIIA MATH" src="https://img.shields.io/badge/%F0%9F%8F%A2%20Company-SIIIA%20MATH-0B1B3A"></a>
        <a href="http://www.umich.mx"><img alt="University: UMSNH" src="https://img.shields.io/badge/%F0%9F%8E%93%20University-UMSNH-1A3A6B"></a>
      </td>
      <td>
        <a href="mailto:heriberto.arias@umich.mx"><img alt="Contact" src="https://img.shields.io/badge/%F0%9F%93%A7-Contact-blue"></a>
        <a href="https://orcid.org/0000-0002-7641-8310"><img alt="ORCID 0000-0002-7641-8310" src="https://img.shields.io/badge/ORCID-0000--0002--7641--8310-green"></a>
        <a href="https://www.researchgate.net/profile/Heriberto-Arias-Rojas"><img alt="ResearchGate Profile" src="https://img.shields.io/badge/ResearchGate-Profile-teal"></a>
      </td>
    </tr>
  </tbody>
</table>

## ⚖️ License
MIT License. See [LICENSE](LICENSE).

