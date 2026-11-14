import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageFilter
import tensorflow as tf

MODEL_PATH = Path(__file__).with_name("fashion_mnist_cnn.keras")
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def largest_component(mask):
    """Keep the largest connected foreground component; remove small logos/noise."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            component = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) > len(best):
                best = component
    result = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        result[y, x] = True
    return result

def make_variants(path):
    """Return two 28x28 views: grayscale object and clean silhouette."""
    original = Image.open(path).convert("RGB")
    gray = ImageOps.grayscale(original)
    gray = ImageOps.autocontrast(gray)

    arr = np.asarray(gray, dtype=np.uint8)
    # Estimate background brightness from the image borders.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg = float(np.median(border))

    # Segment foreground by difference from the border/background.
    if bg >= 127:
        mask = arr < (bg - 28)
    else:
        mask = arr > (bg + 28)

    # Ignore isolated specks; keep the main connected object.
    mask = largest_component(mask)
    ys, xs = np.where(mask)
    if len(xs) < 30:
        # Fallback: use the original image if segmentation is too aggressive.
        crop = gray
        mask_image = ImageOps.invert(gray) if bg >= 127 else gray
    else:
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        crop = gray.crop(box)
        mask_crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
        # Fashion-MNIST convention: dark background, brighter foreground.
        if bg >= 127:
            crop = ImageOps.invert(crop)
        # Create a clean foreground silhouette using the segmented mask.
        mask_image = Image.fromarray((mask_crop.astype(np.uint8) * 255), mode="L")

    def fit28(img):
        img = img.copy()
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(img, ((28-img.width)//2, (28-img.height)//2))
        return canvas

    # Variant A retains grayscale texture; Variant B removes background clutter.
    gray28 = fit28(crop)
    mask28 = fit28(mask_image)
    a = np.asarray(gray28, dtype="float32") / 255.0
    b = np.asarray(mask28, dtype="float32") / 255.0
    return gray28, mask28, a.reshape(1,28,28,1), b.reshape(1,28,28,1)

class FashionClassifierApp:
    def __init__(self, root):
        self.root = root
        root.title("Fashion MNIST — Image Classifier")
        root.geometry("720x720")
        root.minsize(620, 650)
        self.model = None
        self.previews = []

        tk.Label(root, text="Fashion MNIST Image Classifier",
                 font=("Segoe UI", 18, "bold")).pack(pady=(16, 4))
        tk.Label(root, text="Choose one clothing item (JPG or PNG).",
                 font=("Segoe UI", 10)).pack(pady=(0, 12))
        tk.Button(root, text="Choose image", command=self.choose_image,
                  font=("Segoe UI", 11), padx=18, pady=8).pack()

        images = tk.Frame(root)
        images.pack(pady=12)
        self.image_labels = []
        for title in ("Grayscale preprocessing", "Background removed"):
            panel = tk.Frame(images, padx=10)
            panel.pack(side="left")
            tk.Label(panel, text=title, font=("Segoe UI", 10, "bold")).pack()
            label = tk.Label(panel, text="No image", width=28, height=14,
                             relief="groove", bg="#f3f3f3")
            label.pack()
            self.image_labels.append(label)

        self.result_label = tk.Label(root, text="Prediction will appear here",
                                     font=("Segoe UI", 14, "bold"))
        self.result_label.pack(pady=8)
        self.top_label = tk.Label(root, text="", font=("Segoe UI", 11), justify="left")
        self.top_label.pack(pady=4)
        self.note = tk.Label(
            root,
            text=("Fashion MNIST was trained on small 28×28 grayscale images. "
                  "Real product photos differ from the training data, so predictions are not guaranteed. "
                  "The result combines two preprocessing variants."),
            wraplength=650, justify="center", fg="#555", font=("Segoe UI", 9)
        )
        self.note.pack(side="bottom", pady=14, padx=20)

    def choose_image(self):
        if not MODEL_PATH.exists():
            messagebox.showerror("Model not found",
                "First run train_cnn.py and wait until training finishes.")
            return
        path = filedialog.askopenfilename(
            title="Choose a clothing image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            if self.model is None:
                self.result_label.config(text="Loading model...")
                self.root.update_idletasks()
                self.model = tf.keras.models.load_model(MODEL_PATH)

            gray_img, mask_img, batch_a, batch_b = make_variants(path)
            self.previews = []
            for img, label in zip((gray_img, mask_img), self.image_labels):
                display = img.resize((224, 224), Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(display)
                self.previews.append(photo)
                label.config(image=photo, text="", width=224, height=224)

            pred_a = self.model.predict(batch_a, verbose=0)[0]
            pred_b = self.model.predict(batch_b, verbose=0)[0]
            probabilities = (pred_a + pred_b) / 2.0
            best = int(np.argmax(probabilities))
            order = np.argsort(probabilities)[::-1][:3]

            self.result_label.config(
                text=f"Prediction: {CLASS_NAMES[best]} ({probabilities[best] * 100:.1f}%)"
            )
            lines = ["Top 3 predictions (average of two image versions):"]
            for index in order:
                lines.append(f"• {CLASS_NAMES[int(index)]}: {probabilities[index] * 100:.1f}%")
            self.top_label.config(text="\n".join(lines))
        except Exception as exc:
            messagebox.showerror("Image processing error", str(exc))

if __name__ == "__main__":
    root = tk.Tk()
    app = FashionClassifierApp(root)
    root.mainloop()
