################################################################################
#####🔥 This script runs in the YOLO-Torch-2.1_ultralytics environment  🔥#####
#####          and in the Vehicle_MEX_Dataset directory  #####
################################################################################

# ---- Importing the necessary libraries ---- #
from ultralytics import YOLO
import multiprocessing

# pip install -U ultralytics "ray[tune]" optuna

def main():
    print("======================================")
    print("🔬 Preparing Hyperparameter Tuning with Ray Tune & Optuna...")
    print("======================================")
    print("📦 Dataset: Vehicle_MEX_Dataset")
    
    # ---- Loding the model ---- #
    # Usar un modelo base o preentrenado ligero para hacer la búsqueda más rápida.
    # best.pt (de tus runs anteriores) o yolo26s-cls.pt
    model = YOLO("yolo26s-cls.pt")
    print("Model loaded successfully... 🙋")

    # ---- Definición del Espacio de Búsqueda (Search Space) ---- #
    # Si dejas 'space=None', Ultralytics usará su espacio de búsqueda por defecto.
    # Aquí definimos uno personalizado enfocado en clasificación y mitigación de overfitting.
    custom_space = {
        "lr0": (1e-5, 1e-2),          # Learning rate inicial (escala logarítmica)
        "weight_decay": (0.0001, 0.01), # Penalización para regularización L2
        "warmup_epochs": (0.0, 5.0),  # Épocas de calentamiento
        "dropout": (0.0, 0.5),        # Probabilidad de Dropout en la cabeza de clasificación
        "degrees": (0.0, 10.0),       # Rotación de la imagen en grados
        "perspective": (0.0, 0.001)   # Distorsión de perspectiva
    }

    # ---- Iniciar la sintonización (Tuning) ---- #
    print("🚀 Starting Tuning process...")
    # model.tune usa Ray Tune por defecto si está instalado
    results = model.tune(
        data="Vehicle_classification.v1",
        use_ray=True,        # Activar Ray Tune + Optuna
        epochs=30,           # Épocas por cada intento (Trial). Suficiente para ver la tendencia de convergencia.
        iterations=30,       # Número de combinaciones diferentes (Trials) a probar.
        imgsz=640,           # Resolución de entrada
        batch=-1,            # AutoBatch para aprovechar VRAM
        optimizer="AdamW",   # Fijar AdamW, ya que es el mejor para este caso
        workers=4,           # Hilos de CPU para cargar datos
        patience=15,         # Detener un intento si no mejora en 15 épocas
        space=custom_space,  # Espacio de búsqueda personalizado
        project="Vehicle_MEX_Dataset_Tune",
        name="Tune_Experiment",
        save=False,          # No guardar todos los pesos intermedios para ahorrar disco
    )

    print("======================================")
    print("🎉 Hyperparameter Tuning completed successfully!")
    print("Los mejores hiperparámetros se han guardado en:")
    print(r"runs/classify/Vehicle_MEX_Dataset_Tune/Tune_Experiment/best_hyperparameters.yaml")
    print("======================================")

if __name__ == "__main__":
    # Required for multiprocessing in Windows (Ray Tune lo requiere)
    multiprocessing.freeze_support() 
    main()
