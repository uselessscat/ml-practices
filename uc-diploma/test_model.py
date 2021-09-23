import torch
from torch.nn import CrossEntropyLoss


def test_model(model, test_dl):
    total_correctas = 0.0
    total_muestras = 0.0

    for x, target in test_dl:          # Iteramos sobre nuestros datos
        # Inicio de la iteración
        model.eval()                        # Dejamos el modelo en modo evaluación
        with torch.no_grad():               # No se calculará información de gradientes
            # en el código de más abajo.
            x = x.cuda()
            target = target.cuda()          # Enviamos nuestros datos a GPU
            # Hacemos el forward de nuestros datos
            output = model(x)

            # El máximo valor es nuestra predicción
            preds = output.argmax(dim=1)
            # Acumulamos las correctas durante la época
            correctas = (preds == target).sum()
            total_correctas += correctas
            # Sumamos el tamaño del batch
            total_muestras += target.shape[0]

            accuracy = total_correctas/total_muestras  # Acc = correctas/total

            print(
                "\rCorrectas: {} Total: {} Accuracy: {:.2f}%"
                .format(
                    total_correctas,
                    total_muestras,
                    100*accuracy),
                end=""
            )
