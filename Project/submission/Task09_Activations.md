Task 9| Activation Function Swap (ReLU vs Tanh vs GELU)

Objective:


____________________________________________________________________________________

Code :

#Task9
activations = ['relu', 'tanh', 'gelu']
histories = {}

for act in activations:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(64, activation=act), 
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # التدريب
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, 
                        validation_data=(x_val, y_val), verbose=0)
    histories[act] = history.history['val_loss']


plt.figure(figsize=(10, 6))
for act, val_loss in histories.items():
    plt.plot(val_loss, label=f'Activation: {act}')

plt.title('Activation Function Comparison (Validation Loss)')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()

_________________________________________________________________________________
Results:
النتائج رح افقها بالملف
\results\loss_curves\Task9


_________________________________________________________________________________
Short Analysis:
Research shows that the Relu function affects gradient flow by preventing vanishing in the positive range, while the Tanh function integrates outputs from -1 to 1. 
Softsign is similar to Tanh but slower as it approaches zero.

The risk of gradient vanishing is present in Tanh because it is a derivative, approaching zero very closely with very large and very small values. This can halt model learning.

The Relu function is excellent in transformers because it weights the input based on its actual value.

Relu is better and simpler to calculate, and its efficiency is also good with medium and simple data, making it suitable for most tasks.