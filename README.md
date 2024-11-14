# FoodType-NeuralNetwork
This Neural Network general code architecture was stripped and greatly modified 
from on older iteration of a Neural network I built in a currently (11/7/24) private repo

This dataset was very complex and already normalized, this likely could have ended with
a high accuracy at the end of a given model's training, but there were 65 features with
only about 900 samples. This did not prove to be enough to obtain well over 65% accuracy
for the 20 different classifcations.

I do believe if this data were to be learned again, I would suggest taking a CVT approach.
I would also suggest attempting the use of PCA to minimize dimentionality and overfitting.