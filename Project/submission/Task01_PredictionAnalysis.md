## **Task 1 — Deep Prediction Analysis**

##### 

##### ***Objective:*** 



It is how three images were randomly selected from the data that could be worked on in the test and to **test the previous mode**l to see if its prediction was correct or not.

***\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_***



##### ***Code :*** 

###### \#Task 1 

###### num\_samples = 3 #لاختار 3 عينات

###### random\_indices = np.random.choice(len(x\_test), num\_samples, replace=False) # بشل عشوائي تمااما من 10000 صورة رح نختار الثلاثة

###### 

###### for i in random\_indices:

###### &#x20;   sample = x\_test\[i].reshape(1, 28, 28) # كل صورة رح نلتزم بمقاس التدريب الي تدرب عليه المودل

###### &#x20;   prediction = model.predict(sample, verbose=0) # المودل رح يتوقع هنا 

###### &#x20;   predicted\_label = np.argmax(prediction) # رح نختار اعلى قيمة توقع بين الكلاسات العشرة 

###### &#x20;   true\_label = y\_test\[i] # هنا فعليا القيمة الحقيقية 

###### 

###### 

###### \# هنا بس عشان اضمن انه كل شي صح بدي اشوف كنت صورة كل عينة 

###### &#x20;   plt.figure(figsize=(2, 2))

###### &#x20;   plt.imshow(x\_test\[i], cmap='gray')

###### &#x20;   plt.show()

###### &#x20;   # هنا طبعت الرقم للصورة الي تم اختيارها + التوقع الي من المودل + القيمة الفعلية للصورة 

###### &#x20;   print(f"Index: {i} | True: {true\_label} | Predicted: {predicted\_label}")

###### &#x20;   #هنا عشان اشوف توقعه مع كل كلاس للصورة الوحدة و اشوف انع فعليا ال 

###### &#x20;   #argmax  شغالة صح

###### &#x20;   print(f"Softmax Output Vector: {prediction\[0]}\\n")







***\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_***

##### ***Results:***

***النتائج رح افقها بالملف*** 

***result / prediction/ Task1\_p1***

***result / prediction/ Task1\_p2***

***result / prediction/ Task1\_p3***

***\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_***

##### ***Short Analysis:***



The idea here is:

The image is essentially converted to a 28x28px . Then it's extended in a single column \[Flatten]. In this process, the weights are multiplied by the nodes in the column \[input layer]. The model retrieves these weights and works its way from the inputs layer to the hidden layers, then to the output layers.

&#x20;

this process extracts the features present in the image.



Here, Relu plays a role by removing unnecessary elements from the features \[ set it 0]





Finally, at the output, a prediction is made for 10 classes, since the numbers range from 0 to 9. 

Here, SoftMax's role is clear: it takes the result of the predictions for a single image and sets the values ​​to a \[sum of one] = 1 .



The loss is then calculated, and if it's high, we get Backpropagation and the Adam optmizer  is what adjusts the weights and Learning Rates.



