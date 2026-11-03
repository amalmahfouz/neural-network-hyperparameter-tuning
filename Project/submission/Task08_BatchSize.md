Task 8 | Batch Size & Gradient Noise Experiment

Objective:
Not all noise is harmful!

____________________________________________________________________________________

Code :

#Task8



batch_sizes = [8, 32, 128]
histories = {}

for batch in batch_sizes:
    
    
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3), 
            keras.layers.Dense(10, activation="softmax")
           ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # التدريب
    history = model.fit(x_train, y_train, epochs=20, batch_size=batch, 
                        validation_data=(x_val, y_val), verbose=0)
    histories[batch] = history.history['loss']


plt.figure(figsize=(10, 6))
for batch, loss_values in histories.items():
    plt.plot(loss_values, label=f'Batch Size {batch}')

plt.title('Effect of Batch Size on Gradient Noise (Loss Curves)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

_________________________________________________________________________________
Results:
النتائج رح افقها بالملف
\results\loss_curves\Task8_batchsize


_________________________________________________________________________________
Short Analysis:
From my observations of the results, a size of 32 is optimal, offering a faster initial drop and better stability at the end.

The batch size clearly affects the path.

In the case of gradient noise, we observe that 8 prevents falling into false bottoms when the trend isn't global, allowing for easy recovery and preventing the model from becoming stuck or became overfitting.

The idea behind 128 is to utilize the processor and change weights less frequently in each era, but it's not the best for generalization.

