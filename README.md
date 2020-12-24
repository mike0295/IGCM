# IGMC
### SNU Embedding Systems and Applications 2020 Fall

Please note that this implementation of IGMC is not completely time-efficient.
It has only been correctly been trained and tested on the MovieLens100K dataset, and scalability is not ensured. ~~After all, I'm just a noob undergrad.~~

GPU memory 16GB required for training. 


To train the model on MovieLens100K dataset:
<pre>pip install -r requirements.txt  
python main.py --train_igmc</pre>

Once trained (which takes 80 epochs), the model will be saved in the model_dict folder. 


