# README  
Project 3: Real-Time 2D Object Recognition
Spring 2024  
CS 5330 Northeastern  
Professor Bruce Maxwell, PhD
___

### Group Member Names:
* Joseph Nelson Farrell 
* Harshil Bhojwani

___

### Links/Urls:

This is a link the video demo (it is also in the report):
* https://drive.google.com/file/d/1T4pmPGcWS7J1XhQ85cRmawAxwepCQfyq/view?usp=drive_link

___

### Operating System & IDE:
* MacOS
* Visual Studio Code

___

### Time Travel Days:
* 3

___
## Executing the Program:

### Step 1: Run Executable

To execute the program all the user has to do is run the following executable and follow the prompts.
```bash
./object_rec
```
This requires your iPhone be connected to computer

### Step 2: Train-Classic Features (add objects to classic features database)
The program can add items to two different databases and has two different training modes.
Pressing: 
```
t
```
This will generate a classic feature vector and add an object to ```feature_vectors.csv``` (the classis features database)

The user will see the following prompt:
```
******************************************
Entering Training Mode: Classical Feature Vectors
******************************************

What is the label of this item? Please enter the name:
```
Enter the label of the object in the command and press ```enter```.

An image frame will inform the user that the object has been added to the database.

```press any key```

### Step 3: Train-DNN Embeddings (add objects to DNN database)
The program can add items to two different databases and has two different training modes.
Pressing: 
```
d
```
This will generate a DNN embedding feature vector and add an object to ```dnn_feature_vectors.csv``` (the DNN embeddings database)

The user will see the following prompt:
```
******************************************
Entering Training Mode: DNN
******************************************

What is the label of this item? Please enter the name:
```
Enter the label of the object in the command and press ```enter```.

An image frame will inform the user that the object has been added to the database.

```press any key```

### Step 4: Classification Mode-Classic Features
The program can calssify objects using two different databases and has two different classification modes.
Press: 
```
c
```
This will classify an object using classic features.

The user will see the following prompt:
```
What is the true label?
```
Here the user enters the true label of the object. This is needed because programs it's classification accuracy and produced a confusion matrix.

Enter the label of the object in the command and press ```enter```.

An image frame will inform the user of the classification result.

If the image is not in the database the user will be informed with text overlaying a frame to add the object to the database.

```press any key```

### Step 5: Classification Mode-DNN
The program can calssify objects using two different databases and has two different classification modes.
Press: 
```
p
```
This will classify an object using DNN embeddings.

The user will see the following prompt:
```
What is the true label?
```
Here the user enters the true label of the object. This is needed because programs it's classification accuracy and produced a confusion matrix.

Enter the label of the object in the command and press ```enter```.

An image frame will inform the user of the classification result.

If the image is not in the database the user will be informed with text overlaying a frame to add the object to the database.

```press any key```

### Step 6: Exiting
The user can run the program as many times as they wish and switch between modes. The program track all classifcation in a consfusion matrix.  

To exit press: 
```
q
```
The user will see the following prompt:
```
Do you want to display the confusion matrix?
```
This is the confusion matrix for the classic feature classification. If the user wants to display the results enter:
```
yes
```
Press ```enter```. (anything other than ```yes``` will be regarded as ```no```).

If yes is selected the user will see this: 

```
This is the order of the rows and cols: 

cap chess_peice coin eraser pen spoon thule_key 

0 0 0 0 0 0 0 
0 0 0 0 0 0 0 
0 0 1 0 0 0 0 
0 0 0 0 0 0 0 
0 0 0 0 0 0 0 
0 0 0 0 0 0 0 
0 0 0 0 0 0 1 
```

Next, the user will see the following prompt:
```
Do you want to display the dnn confusion matrix?
```
This is the confusion matrix for the DNN embedding classification. If the user wants to display the results enter:
```
yes
```
Press ```enter```. (anything other than ```yes``` will be regarded as ```no```).

If yes is selected the user will see this: 

```
This is the order of the rows and cols: 

chess_peice coin eraser spoon thule_key 

0 0 0 0 0 
0 2 0 0 0 
0 0 0 0 0 
0 0 0 0 0 
0 0 0 0 0 
```

Program terminates.
