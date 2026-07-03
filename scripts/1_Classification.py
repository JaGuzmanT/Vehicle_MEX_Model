################################################################################
#####🔥 This script runs in the YOLO-Torch-2.1_ultralytics environment  🔥#####
#####          and in the Vehicle_MEX_Dataset directory  #####
################################################################################

# ---- Importing the necesay libraries ---- #
from ultralytics import YOLO
import multiprocessing

# ---- Loding the model ---- #
model = YOLO("yolo26s-cls.pt")
print("Model loaded successfully... 🙋")

# ---- Defining the training function ---- #
def main():
    print("======================================")
    print("🔧 Preparing training...")
    print("======================================")
    print("📦 Dataset: Vehicle_MEX_Dataset")
    print("Training... We will train your model 🚀!!")
    
    # ---- Training the model ---- #
    results = model.train(
        data="Vehicle_classification.v1",
        imgsz=640,
        batch=-1, # Automatically adjust batch size to fit VRAM
        patience=15,
        lr0=0.0001,
        optimizer="AdamW",
        pretrained=True,
        project="Vehicle_MEX_Dataset",
        name="Improvement_1",
        workers=4,        # Carga paralela de datos
        cos_lr=True,      # Activar Learning Rate dinámico (curva coseno)
        epochs=200,
        warmup_epochs=5,  # Aumentar calentamiento para estabilizar los pesos en las primeras etapas
        weight_decay=0.005, # Regularización de pesos para evitar sobreajuste, muy útil cuando se usa AdamW y hay overfitting
        dropout=0.4,        # Dropout para evitar sobreajuste, muy útil cuando se usa AdamW y hay overfitting
        degrees=5,          # Rotación aleatoria entre -5 y 5 grados para aumentar la variabilidad del conjunto de datos
        perspective=0.001,  # Transformación de perspectiva aleatoria para aumentar la variabilidad del conjunto de datos
        save=True, )

    print("Training completed successfully... 🎉")
    print("📊 Training metrics:")
    print(results)
    
    # ---- Validating results ---- #
    results_val = model.val()
    print("📊 Validation metrics:")
    print(results_val)
    return results, results_val

# ---- Training and validating the model ---- #

if __name__ == "__main__":
    multiprocessing.freeze_support() # Required for Windows
    main()
