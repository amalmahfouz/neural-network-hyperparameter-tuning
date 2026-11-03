Task 3 — Epoch-Based Learning Curve Exploration

Objective:
Comparison in Moodle across the three learning scenarios

____________________________________________________________________________________

Code :
import matplotlib.pyplot as plt

model_20 = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
model_20.compile(
    optimizer="adam", # كيف يعدل ال W
    loss=  "sparse_categorical_crossentropy", # كيف يحسب ال loss
    metrics =["accuracy"] # شو المقياس
)
    

# EarlyStopping
#callback = tf.keras.callbacks.EarlyStopping(patience=3,monitor="val_loss",restore_best_weights=True)
history_20 = model_20.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val), verbose=0)

    
#اتبع النتائج نظري
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
ax1.plot(history_20.history['loss'], label='Train Loss')
ax1.plot(history_20.history['val_loss'], label='Val Loss')
ax1.set_title(f'Loss 20 Epochs')
ax1.legend()
    
ax2.plot(history_20.history['accuracy'], label='Train Acc')
ax2.plot(history_20.history['val_accuracy'], label='Val Acc')
ax2.set_title(f'Accuracy 20 Epochs')
ax2.legend()
plt.show()




_________________________________________________________________________________
Results:
النتائج رح افقها بالملف
result / loss_curves/Task3_5_epochs
result / loss_curves/Task3_10_epochs
result / prediction/Task3_20_epochs

_________________________________________________________________________________
Short Analysis:
I trained three models with identical specifications, but the number of train loss values ​​varied.

When I had 5 epochs, I noticed it was still in the learning phase and needed further development. This was considered underfitting. The result was good, but not optimal.

When I had 10 epochs, I noticed significant improvement. However, when I had 20 epochs, the model was considered overfitting, as the Val_Loss started to increase while the Train_Loss decreased.

