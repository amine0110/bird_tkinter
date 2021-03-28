import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import os
import cv2
import numpy as np

path_to_model = '/home/pycad/Documents/Project/code/model'
path_to_folders = '/home/pycad/Downloads/images/train'
model = tf.keras.models.load_model(path_to_model) 
global path
path = None



def get_birds_names(path_to_folders):
    birds_names = os.listdir(path_to_folders)
    birds_names.sort()
    return birds_names

def predict(path_of_new_image):

    birds_names = get_birds_names(path_to_folders)         
    

    image_new_size = 229                                    
    img_array = cv2.imread(path_of_new_image)               
    img = img_array.copy()
    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        

    img_array = cv2.resize(img_array, (image_new_size, image_new_size)) 
    img_array = img_array.astype("float32")                 
    
    img_array = img_array / 255.0                           
    np_image = np.expand_dims(img_array, axis=0)            

    predictions = model(np_image)                           
    predicted_class_idx = np.argmax(predictions)            

    probability = np.max(predictions)                       
    predicted_class = birds_names[predicted_class_idx]      

    return predicted_class, probability 


def choose_image():
    global path
    path = filedialog.askopenfilename()
    global image
    
    if path:
        image = Image.open(path)
        image = image.resize((400,400), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        print_image.config(image=image)



def predict_bird():
    if path:
        prediction = predict(path)
        text = prediction[0] + '\n' + 'With a proba of: ' + str(int(prediction[1]*100))+'%'
        result_zone.config(text=text)


bg = '#002929'                      
root = Tk()                         
root.geometry("500x700")            
root.title('WHAT IS YOUR BIRD!')    
root.configure(bg=bg)               
root.resizable(width=0, height=0) 

title = Label(root, text='Choose your bird', font='agencyFB 25 bold', bg=bg, fg='white')
title.pack()

image_zone = Frame(root, width=400, height=400, bg='#063b3b')
image_zone.pack(pady=(20,0))
image_zone.pack_propagate(0)
print_image = Label(image_zone, bg='#063b3b')
print_image.pack()

open_image = Button(root, text='OPEN', width=15, bg='white', fg=bg, font='none 15 bold', command=choose_image)
open_image.pack(pady=(20,0))

predict_image = Button(root, text='Predict', width=15, bg='white', fg=bg, font='none 15 bold', command=predict_bird)
predict_image.pack(pady=(20,0))

result_zone = Label(root, font='none 20 bold', fg='white', bg=bg)
result_zone.pack(pady=(30,0))


root.mainloop()