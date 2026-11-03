Task 7|  Optimizer Comparison

Objective:
How does the different type of optimization algorithm affect the improvement and learning of the model and achieving the best accuracy?

____________________________________________________________________________________

Code :

#Task7


model_adamW= keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3), 
    keras.layers.Dense(10, activation="softmax")
])


# هذا السطر تم تغييره على الاربعة  حسب كل مودل 
#    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
#    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
#    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#  optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004),



model_adamW.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004),# هذا السطر تم تغييره على الاربعة 
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(patience= 3 ,monitor="val_loss",restore_best_weights=True)
history = model_adamW.fit(x_train,y_train ,epochs= 20, validation_data= (x_val,y_val),callbacks=callback )



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
\results\loss_curves\Task7_adam
\results\loss_curves\Task7_adamW
\results\loss_curves\Task7_SDG_With_M
\results\loss_curves\Task7_SDG

_________________________________________________________________________________
Short Analysis:

By tracking the results: Firstly, in terms of speed, the first model(SDG) took 20 full epochs and was slower compared to the others. The second model(SDG with M) was actually faster than the first, but it performed fewer epochs. The third and fourth models(adam , adamW) clearly showed faster speeds and a significantly lower number of epochs.

Secondly, regarding error topography, the first model(SDG) moves slowly, so it is affected by the existing irregularities. The second model(SDG With M) has somewhat better performance and therefore overcomes irregularities more easily. The third model(Adam) is an adaptive model that adjusts itself; if the curve is irregular, it moves slowly, but if the curve is flat, it speeds up.

The idea is that Adam calculates for each neuron individually, sometimes increasing the value and sometimes decreasing the weights according to stability, unlike the first model (SDG), for example, which uses a fixed learning rate for the entire network.

