# Retinal_Image_Classifier
Three class retinal image classification: Normal, Drusen, and Exudate retinal fundus.
<h1>Project description
</h1>
<h2>What is the problem?
</h2>
<p>Diabetic Retinopathy (DR) is a significant concern due to its potential to cause preventable blindness, primarily characterized by damage to the blood vessels at the back of the retina, with three classifications based on retinal image features: drusen and hard exudate as yellowish white deposits, and soft exudate manifesting as cotton wool spots. Early detection and diagnosis are crucial in managing this condition, alongside heightened awareness of its link to Age-Related Macular Degeneration (ARMD).</p>
<h2>Methodology</h2>
<p><img width="1440" alt="image" src="https://github.com/MonaMohsen166/Retinal_Image_-Classifier/assets/73717585/426d1347-98bb-41a3-9f51-f1daafa633db">
</p>
<p>As depicted, two approaches were put forth: Bag of Visual Words (BoVW) and Convolutional Neural Networks (CNN).
  However, it was demonstrated that CNN emerged as the superior method.
  
Why CNN better than BoVW?
<ul>
<li>Detect patterns better</li>
<li>Can detect complex pattern</li>
<li>Reduce image weight</li>
</ul>
<br>
Steps of CNN Deep Retinal Classification:
<br>
 1-Train
<br>
 2-Feature Extraction
<br>
 3-Feature Classification
<br>
 4-Results Analysis using metrics like:AUC,F1 Score,Precision,Sensitivity,Specificity and Accuracy.
<br>
</p>
<h2>GUI Output:</h2>
<br>
 <p>
    ![image](https://github.com/MonaMohsen166/Retinal_Image_-Classifier/assets/73717585/616cc79b-e63c-40b4-9780-7d8e049cd881)

![image](https://github.com/MonaMohsen166/Retinal_Image_-Classifier/assets/73717585/8bff8731-0106-4b67-977b-5defcff521cf)

 </p>
<br>
<br>
<h2>Results:</h2>
<br>
<p>
AUC: 0.84
  <br>
F1 Score: Drusen=0.80, Exudate:0.83 and Normal:0.91
  <br>
Precision: Drusen=0.89, Exudate:0.77 and Normal:0.91
  <br>
Sensitivity: Drusen=0.73, Exudate:0.91 and Normal:0.91
  <br>
Specificity: Drusen=0.95, Exudate:0.86 and Normal:0.95
  <br>
Accuracy: 0.85
<br>
</p>
  
