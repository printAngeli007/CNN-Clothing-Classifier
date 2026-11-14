# CNN Fashion MNIST

Проєкт навчає згорткову нейронну мережу (CNN) 10 категорій одягу.

## 1. Встановлення

Рекомендовано Python 3.13.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Навчання моделі

```bat
python train_cnn.py
```

Під час першого запуску TensorFlow завантажить набір даних Fashion MNIST. Після навчання буде створено:
- `fashion_mnist_cnn.keras` — навчена модель;
- `results/accuracy_loss.png` — графіки Accuracy та Loss;
- `results/confusion_matrix.png` — матриця помилок;
- `results/classification_report.txt` — precision, recall та F1-score.

## 3. Перевірка власних зображень

Після завершення навчання запусти:

```bat
python predict_gui.py
```

Натисни **Choose image** і вибери JPG або PNG із зображенням одного предмета одягу. Програма покаже прогноз і три найімовірніші класи.

## Категорії Fashion MNIST

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Важливо

Fashion MNIST складається з маленьких чорно-білих зображень 28×28. Власні фотографії автоматично переводяться у відтінки сірого, обрізаються та масштабуються до 28×28. Звичайні фотографії можуть відрізнятися від навчальних даних, тому точність на них не гарантована. Для кращої демонстрації обирай одне чітке зображення предмета на простому фоні.
