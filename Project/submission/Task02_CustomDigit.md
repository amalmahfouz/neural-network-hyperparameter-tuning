## **Task 2 — Deep Prediction Analysis**

##### 

##### ***Objective:***

The goal  was to load new data that the model had not been trained on and test it to see if its prediction would be correct , specialize if the data did not match the parameters of what it had been trained on.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_



##### ***Code :***



###### \#Task 2

from PIL import Image, ImageOps, ImageEnhance

import numpy as np

import matplotlib.pyplot as plt





from google.colab import files

import cv2  # للتعديل على الصور





uploaded = files.upload() #ارفع الملف

file\_name = list(uploaded.keys())\[0]



def prepare\_and\_predict(image\_path):

&#x20;

&#x20;   img = Image.open(image\_path).convert('L') # نفتح الصورة و نحولها ل الرمادي

&#x20;

&#x20;   # من المحاولات احافظ على الصورة في المنتصف

&#x20;   width, height = img.size

&#x20;   new\_side = max(width, height)

&#x20;   #لخلفية بيضا

&#x20;   new\_img = Image.new('L', (new\_side, new\_side), 255)

&#x20;   new\_img.paste(img, ((new\_side - width)//2, (new\_side - height)//2))

&#x20;

&#x20;   #  الحجم ل  28\*28

&#x20;   img = new\_img.resize((28, 28), Image.Resampling.LANCZOS)

&#x20;

&#x20;   # هنا عكس الالوان ليكون الرقم ابيض ة الخلفية سوداء

&#x20;   img\_array = np.array(img)

&#x20;   img\_array = 255 - img\_array

&#x20;

&#x20;   #من المحاولات اوصل لافضل تحويل

&#x20;   # لكن كان كله بنفس الاداء :(

&#x20;   \_, binary\_img = cv2.threshold(img\_array, 100, 255, cv2.THRESH\_BINARY)

&#x20;

&#x20;   # Normalization

&#x20;   normalized\_img = binary\_img / 255.0

&#x20;   # بدي اشوف الصورة الي انا ادخلتها و كيف تحولت عنده

&#x20;   plt.figure(figsize=(6, 3))

&#x20;   plt.subplot(1, 2, 1)

&#x20;   plt.imshow(Image.open(image\_path), cmap='gray')

&#x20;   plt.title("Original")

&#x20;   plt.axis('off')

&#x20;

&#x20;   plt.subplot(1, 2, 2)

&#x20;   plt.imshow(normalized\_img, cmap='gray')

&#x20;   plt.title("Processed for Model")

&#x20;   plt.axis('off')

&#x20;   plt.show()

&#x20;

&#x20;   # المودل يختبر هنا ع لصورة الي دخلتها

&#x20;   sample = normalized\_img.reshape(1, 28, 28)

&#x20;   prediction = model.predict(sample, verbose=0)

&#x20;   predicted\_label = np.argmax(prediction)

&#x20;

&#x20;   print(f"Predicted Digit: {predicted\_label}")

&#x20;   print(f"Softmax Output Vector: {prediction\[0].round(3)}")

&#x20;

&#x20;   return predicted\_label, prediction





prepare\_and\_predict(file\_name)





***\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_***

##### ***Results:***

***النتائج رح افقها بالملف***

***result / prediction/ Task2***

***\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_***

##### ***Short Analysis:***



***Through experimentation in this section, the images had to be converted to the same image specifications as those the model practiced in   MNIST  Observations during the experimentation indicated that the image should be 28 x 28 with a black background and the number in white.***



***Many drawings were tested, and it clearly failed to predict many of the numbers.***



***I noticed its success with thicker font, centered numbers, and maintaining space on the sides, i.e., the image was centered.***



***It failed to predict many of the images, and the images that were worked on are attached in a folder ( upload\_image\_task\_2)***

