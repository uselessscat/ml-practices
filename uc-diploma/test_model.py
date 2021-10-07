import torch
from torch.nn import CrossEntropyLoss
from tqdm.notebook import tqdm


def test_model(model, test_dl):
    total_correctas = 0.0
    total_muestras = 0.0

    for x, target in test_dl:
        model.eval()

        with torch.no_grad():
            x = x.cuda()
            target = target.cuda()

            output = model(x)
            preds = output.argmax(dim=1)

            correctas = (preds == target).sum()
            total_correctas += correctas

            total_muestras += target.shape[0]

            accuracy = 100 * (total_correctas / total_muestras)

            print(
                f'\rCorrectas: {total_correctas} '
                f'Total: {total_muestras} '
                f'Accuracy: {accuracy:.2f}%',
                end=''
            )
