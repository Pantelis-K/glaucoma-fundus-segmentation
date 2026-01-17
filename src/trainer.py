
# Imports -----------------------------------------------------
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
# Custom modules
import metrics
from model import build_UNET

# Config -----------------------------------------------------
cup_or_disc = "cup"
IMAGE_SIZE   = 256           
N_CHANNELS   = 5            # R, G, B, CLAHE‑gray, Sobel
BATCH_SIZE   = 4
AUTOTUNE     = tf.data.AUTOTUNE
CUP_VALUE = 2 # pixel value for cup in mask

#Directories -----------------------------------------------------
# Resolve project root (without assuming cwd) 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# input directories
DATA_DIR  = PROJECT_ROOT / "data"
STACK_DIR = DATA_DIR / "stacks"
MASK_DIR  = DATA_DIR / "masks"
# output directories
RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)
run_name = f"{cup_or_disc}_{IMAGE_SIZE}_unet"
run_dir = RUNS_DIR / run_name
run_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir = run_dir / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)


# Load dataset paths ---------------------------------------------------
stack_paths = sorted(STACK_DIR.glob("*_stack.npy"))
mask_paths  = sorted(MASK_DIR.glob("*.png"))
assert len(stack_paths) == len(mask_paths), "Stacks / masks count mismatch!"
print(f"{len(stack_paths)} paired samples found")




# function defintions -----------------------------------------------------

def load_stack(path):
    st = np.load(path.numpy().decode()).astype(np.float32)
    st = st[..., :N_CHANNELS]                     # RGB + CLAHE + Sobel
    st = tf.image.resize(st, (IMAGE_SIZE, IMAGE_SIZE))
    return st / 127.5 - 1.0                  # [-1,1]

def load_mask(path):
    img_raw = tf.io.read_file(path)
    mask    = tf.io.decode_png(img_raw, channels=1)        # (H,W,1), uint8
    mask  = tf.image.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), method='nearest')
    # cast to binary (0/1)
    if cup_or_disc == "cup":
        mask    = tf.cast(mask == CUP_VALUE, tf.float32)               
    elif cup_or_disc == "disc":
        mask    = tf.cast(mask > 0, tf.float32)
    return mask

def tf_load(stack_path, mask_path):
    #Wrap the numpy loader so tf.data can call it
    stack = tf.py_function(load_stack, [stack_path], tf.float32)
    mask  = tf.py_function(load_mask,  [mask_path],  tf.float32)
    stack.set_shape([IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS])
    mask .set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    return stack, mask

def augment(img, mask):
    # Generate a deterministic 2‑element seed tensor
    seed = tf.random.uniform([2], maxval=2**31 - 1, dtype=tf.int32)

    # Left–right flip (same seed for image & mask)
    img  = tf.image.stateless_random_flip_left_right(img,  seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed)

    # Brightness jitter on the RGB channels only
    rgb  = tf.image.stateless_random_brightness(img[..., :3], max_delta=0.1, seed=seed)
    img  = tf.concat([rgb, img[..., 3:]], axis=-1)

    return img, mask


# build the pipeline -----------------------------------------------------
paths_ds = tf.data.Dataset.from_tensor_slices((stack_paths, mask_paths))
# 80/10/10 split
n_total  = len(stack_paths)
n_train  = int(0.8 * n_total)
n_val    = int(0.1 * n_total)

train_ds = paths_ds.take(n_train)
val_ds   = paths_ds.skip(n_train).take(n_val)
test_ds  = paths_ds.skip(n_train + n_val)

train_ds = (train_ds
            .shuffle(n_train)
            .map(tf_load,  num_parallel_calls=AUTOTUNE)
            .map(augment,  num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (val_ds
          .map(tf_load, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

test_ds = (test_ds
           .map(tf_load, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))


# Build the model -----------------------------------------------------
model = build_UNET(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS),
    n_classes=1,
    name=f"{cup_or_disc}_UNET_{IMAGE_SIZE}"
)
model.summary()



# compile config -----------------------------------------------------
LR = 1e-4
ALPHA, BETA = 0.5, 0.5
MONITOR = "val_dice_coef"  # from custom metrics
MODE = "max"

EPOCHS_TO_SAVE = [1, 5, 10, 20, 50, 100]
MAX_EPOCHS = max(EPOCHS_TO_SAVE)



# Compile the model -----------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=metrics.log_dice_loss,                  
    metrics=[metrics.dice_coef,
             metrics.boundary_loss,
             metrics.combined_dice_boundary_loss(alpha=ALPHA,beta=BETA),
             metrics.iou,
             tf.keras.metrics.BinaryAccuracy()] 
)
#callbacks -----------------------------------------------------
class SaveEpochs(tf.keras.callbacks.Callback):
    def __init__(self, epochs_to_save, out_dir, pattern):
        super().__init__()
        self.keep = set(epochs_to_save)
        self.out_dir = out_dir
        self.pattern = pattern

    def on_epoch_end(self, epoch, logs=None):
        epoch_1idx = epoch + 1
        if epoch_1idx in self.keep:
            fname = self.out_dir / self.pattern.format(epoch_1idx)
            self.model.save(fname)
            print(f"\n✔ Saved {fname}")

save_epochs_cb = SaveEpochs(
    epochs_to_save=EPOCHS_TO_SAVE,
    out_dir=ckpt_dir,
    pattern="epoch{:03d}.h5",
)

best_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(ckpt_dir / "best.h5"),
    monitor=MONITOR,
    mode=MODE,
    save_best_only=True,
    verbose=1,
)

lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1,
)

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor=MONITOR,
    patience=10,
    mode=MODE,
    restore_best_weights=False,
    verbose=1,
)

callbacks = [save_epochs_cb, best_ckpt_cb, lr_cb, early_stop_cb]

# Train the model -----------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=callbacks,
)


# Post-run summary -----------------------------------------------------
best_epoch = int(np.argmax(history.history[MONITOR])) + 1
best_score = history.history[MONITOR][best_epoch - 1]
print(f"Best {MONITOR} at epoch {best_epoch}: {best_score:.4f}")
print(f"Best checkpoint saved to: {ckpt_dir / 'best.h5'}")

# plot and save training curves -----------------------------------------------------
epochs_ran = range(1, len(history.history["loss"]) + 1)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs_ran, history.history["dice_coef"],     label="Dice")
plt.plot(epochs_ran, history.history["val_dice_coef"], label="Val Dice")
plt.xlabel("Epoch"); plt.ylabel("Dice coef"); plt.legend(); plt.grid(True)
plt.subplot(1,2,2)
plt.plot(epochs_ran, history.history["loss"],     label="Loss")
plt.plot(epochs_ran, history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
plt.suptitle("Training history"); plt.tight_layout();
plot_path = run_dir / "training_curves.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Training curves saved to: {plot_path}")
plt.show()


# save training history as CSV -----------------------------------------------------
hist_df = pd.DataFrame(history.history)
hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))   # 1‑based epoch column
out_path = run_dir / "training_history.csv"
hist_df.to_csv(out_path, index=False)
print(f"Training history saved to: {out_path}")