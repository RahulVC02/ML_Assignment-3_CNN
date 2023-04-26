import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.pipeline import make_pipeline
import random
import librosa
from keras.applications.vgg16 import VGG16
from matplotlib.animation import FuncAnimation
import cv2
import tensorflow as tf
import streamlit.components.v1 as components
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras import backend as K

tf.compat.v1.disable_eager_execution()

def generate_saliency_map(model, img):
    # Load the image and preprocess it for VGG16
    x = img
    # Create a dictionary mapping layer names to the corresponding layer
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    # Set up an array to hold the saliency maps for each layer
    saliency_maps = []
    # Loop over all layers and generate the corresponding saliency map
    for layer_name in layer_dict:
        # Get the output of the current layer and the input to the model
        layer_output = layer_dict[layer_name].output
        layer_input = model.input
        # Compute the gradients of the output with respect to the input
        grads = K.gradients(layer_output, layer_input)[0]
        # Normalize the gradients
        grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
        # Define a function to generate the saliency map for the given input image
        saliency_fn = K.function([layer_input], [grads])
        # Generate the saliency map for the current layer
        saliency_map = saliency_fn([x])[0][0]
        # Add the saliency map to the list of maps for all layers
        saliency_maps.append(saliency_map)

    return saliency_maps


my_classes = ['dog','chainsaw','crackling_fire','helicopter','rain','crying_baby', 'clock_tick','sneezing','rooster','sea_waves']
audio_paths = {'dog':['data/1-100032-A-0.wav','data/1-110389-A-0.wav','data/1-30226-A-0.wav'], 'chainsaw':['data/1-116765-A-41.wav','data/1-19898-A-41.wav','data/1-19898-B-41.wav'], 'crackling_fire':['data/1-17150-A-12.wav','data/1-17565-A-12.wav','data/1-17742-A-12.wav'], 'helicopter':['data/1-172649-A-40.wav','data/1-172649-B-40.wav','data/1-172649-C-40.wav'], 'rain':['data/1-17367-A-10.wav','data/1-21189-A-10.wav','data/1-26222-A-10.wav'], 'crying_baby':['data/1-187207-A-20.wav','data/1-211527-A-20.wav','data/1-211527-B-20.wav'], 'clock_tick':['data/1-21934-A-38.wav','data/1-21935-A-38.wav','data/1-35687-A-38.wav'], 'sneezing':['data/1-26143-A-21.wav','data/1-29680-A-21.wav','data/1-31748-A-21.wav'], 'rooster':['data/1-26806-A-1.wav','data/1-27724-A-1.wav','data/1-34119-B-1.wav'], 'sea_waves':['data/1-28135-A-11.wav','data/1-28135-B-11.wav','data/1-39901-A-11.wav']}
               
# Create a streamlit app
st.title("2D CNN for Audio Classification")
st.write(
    "This app shows a 2D CNN model trained on the ESC-10 dataset. It displays the spectrogram of the audio file and the predicted class. It also displays an animation of the saliency maps at each layer of the network. I have used the VGG16 architecture for this task. The model was trained on the ESC-10 dataset."
) 

st.write(
    "The drop-down menu allows you to select a class and a random audio file from that class is selected. The spectrogram of the audio file is displayed and the predicted class is shown. The animation shows the saliency maps at each layer of the network. The animation is paused by default. You can click on the play button to start the animation. The animation may take a small amount of time to load."
)
    
# The sidebar contains the sliders
with st.sidebar:
    #create a slider to select audio class and file
    class_name = st.selectbox('Select Class', my_classes)
    file_list = audio_paths[class_name]
    audio_file_path = random.choice(file_list)

audio, sr = librosa.load(audio_file_path, sr=44100)
mel_spec = librosa.feature.melspectrogram(y = audio, sr=22050, n_fft=2048, hop_length=1024, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
input_shape = (224, 224)
resized_mel_spec = cv2.resize(mel_spec_db, input_shape)

fig, ax = plt.subplots()
librosa.display.specshow(resized_mel_spec, y_axis='mel', fmax=8000, x_axis='time')
plt.title(f'Spectrogram For {class_name}')
plt.colorbar(format='%+2.0f dB')
st.pyplot(fig)


model = VGG16(weights='imagenet', include_top=True)
mel_spec = librosa.feature.melspectrogram(y = audio, sr=22050, n_fft=2048, hop_length=1024, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
input_shape = (224, 224)
resized_mel_spec = cv2.resize(mel_spec_db, input_shape)

resized_mel_spec_rgb = np.stack((resized_mel_spec,) * 3, axis=-1)
resized_mel_spec_rgb = resized_mel_spec_rgb - resized_mel_spec_rgb.min()
resized_mel_spec_rgb = resized_mel_spec_rgb / resized_mel_spec_rgb.max()
resized_mel_spec_rgb = resized_mel_spec_rgb * 255
resized_mel_spec_rgb = np.ceil(resized_mel_spec_rgb).astype(np.uint8)


###################################################################################

maps = generate_saliency_map(model, resized_mel_spec_rgb[np.newaxis,...])
new_maps=[]

for map in maps:
    map = map - map.min()
    map = map / map.max()
    map = map * 255
    map = np.ceil(map).astype(np.uint8)
    new_maps.append(map)

fig3, ax3 = plt.subplots()
# Create a gif of the saliency maps
def update(frame):
    ax3.clear()
    ax3.imshow(new_maps[frame], cmap='jet')
    ax3.set_title(f"Layer {frame}")

anim = FuncAnimation(fig3, update, frames = 21, interval = 1000)

with open("myvideo.html","w") as f:
  a = anim.to_html5_video()
  print(a, file=f)
  
HtmlFile = open("myvideo.html", "r")
source_code = HtmlFile.read() 
components.html(source_code, height = 480,width=900)

# Line between the model and the plot
st.write("I have used the viridis colormap for the animation. Blue shades represent lower values while yellow shades represent higher values.")
