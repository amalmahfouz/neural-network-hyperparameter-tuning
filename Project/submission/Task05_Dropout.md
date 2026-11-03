Task 5 | Dropout Ablation Study  

Objective:
how the dropout help to became the model more stable and further from overfitting 

____________________________________________________________________________________

Code :

#Task5
model_3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(64, activation="relu"),
    # نفس الكود مرتين هما مرة 0.1 و مرة 0.3
    keras.layers.Dropout(0.3), 
    keras.layers.Dense(10, activation="softmax")
])


model_3.compile(
    optimizer="adam", # كيف يعدل ال W
    loss=  "sparse_categorical_crossentropy", # كيف يحسب ال loss
    metrics =["accuracy"] # شو المقياس
)

# EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(patience= 3 ,monitor="val_loss",restore_best_weights=True)
history = model_3.fit(x_train,y_train ,epochs= 20, validation_data= (x_val,y_val),callbacks=callback )



#اتبع النتائج نظري
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
_________________________________________________________________________________
Results:
النتائج رح افقها بالملف
results\loss_curves\Task5_dropout0.1
results\loss_curves\Task5_dropout0.3

_________________________________________________________________________________
Short Analysis:

The main idea here is that the use of Dropout which plays a clear role in overfitting.

This means that by observing the results of the training and validation phases, there was a difference in the result in Loss.

The results with 0.1 showed that the model still leaned towards the overfitting

While at 0.3, the curve converged more, and therefore became more stable and further from overfitting

