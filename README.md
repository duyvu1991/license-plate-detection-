# license-plate-detection-

# LICENSE PLATE DETECTION AND RECOGNITION
![image alt](https://www.kurokesu.com/main/wp-content/uploads/2016/08/out2-1.gif)

## What about the project?
* Detect stolen vehicles and support to search them.

* Manage of vehicles entrances and exist parking.
* Control toll fee

## Procees

![](https://i.imgur.com/C08xJ3C.png)
 
## 2 Subtask:
**1.Detect licens plate**
For each image, we manually annotated the 4 corners of the LP in the picture 
![](https://i.imgur.com/ES9kFf7.png)

  The proposed Warped Planar Object Detection Network (WPOD-NET) searches for LPs and regresses one affine transformation per detection, allowing a rectification of the LP area to a rectangle resembling a frontal view.
  

![](https://i.imgur.com/YxPhM4Q.png)
     **Detailed WPOD -NET- ARCHITECTURE**

![image alt](http://www.programmersought.com/images/856/72b3f28505498c604201c262cf1754d8.JPEG)



**2.Licence plate recognition**

Preprocessing image with Open CV2:


![](https://i.imgur.com/qwxEaKI.png)

Using SVM model to predict :

![](https://i.imgur.com/EVYhB78.png)


