import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import random
import librosa
from keras.applications.vgg16 import VGG16
from matplotlib.animation import FuncAnimation
import cv2
import tensorflow as tf
import streamlit.components.v1 as components
from keras import backend as K

tf.compat.v1.disable_eager_execution()

def generate_saliency_map(model, img):
    model_output = model(img)

    # Create a dictionary mapping layer names to the corresponding layer
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    saliency_maps = []
    for layer_name in layer_dict:
        if not isinstance(layer_dict[layer_name], tf.keras.layers.Conv2D):
            continue

        layer_output = layer_dict[layer_name].output
        layer_input = model.input
        
        # Compute the gradients of the output with respect to the input
        grads = K.gradients(layer_output, layer_input)[0]
        saliency_fn = K.function([layer_input], [grads])
        saliency_map = saliency_fn([img])[0][0]
        saliency_maps.append(saliency_map)

    return saliency_maps


my_classes = ['dog','chainsaw','crackling_fire','helicopter','rain','crying_baby', 'clock_tick','sneezing','rooster','sea_waves']
audio_paths = {'dog':['data/1-100032-A-0.wav','data/1-110389-A-0.wav','data/1-30226-A-0.wav'], 'chainsaw':['data/1-116765-A-41.wav','data/1-19898-A-41.wav','data/1-19898-B-41.wav'], 'crackling_fire':['data/1-17150-A-12.wav','data/1-17565-A-12.wav','data/1-17742-A-12.wav'], 'helicopter':['data/1-172649-A-40.wav','data/1-172649-B-40.wav','data/1-172649-C-40.wav'], 'rain':['data/1-17367-A-10.wav','data/1-21189-A-10.wav','data/1-26222-A-10.wav'], 'crying_baby':['data/1-187207-A-20.wav','data/1-211527-A-20.wav','data/1-211527-B-20.wav'], 'clock_tick':['data/1-21934-A-38.wav','data/1-21935-A-38.wav','data/1-35687-A-38.wav'], 'sneezing':['data/1-26143-A-21.wav','data/1-29680-A-21.wav','data/1-31748-A-21.wav'], 'rooster':['data/1-26806-A-1.wav','data/1-27724-A-1.wav','data/1-34119-B-1.wav'], 'sea_waves':['data/1-28135-A-11.wav','data/1-28135-B-11.wav','data/1-39901-A-11.wav']}
               
st.title("2D CNN for Audio Classification")
st.write(
    "This app shows a 2D CNN model trained on the ESC-10 dataset. It displays the spectrogram of the audio file and the predicted class. It also displays an animation of the saliency maps at each layer of the network. I have used the pre-trained VGG16 architecture for this task. The model was trained on the ESC-10 dataset."
)

st.write(
    "VGG16 has 13 convolutional layers and 3 fully connected layers. I've generated the saliency maps for the 13 convolutional layers. Saliency maps help identify pixels of the input image which are being focussed on by a given layer."
)

st.write(
    "The drop-down menu allows you to select a class and a random audio file from that class is selected. The spectrogram of the audio file is displayed and the predicted class is shown. The animation shows the saliency maps at each layer of the network. The animation is paused by default. You can click on the play button to start the animation. The animation may take a small amount of time to load."
)

st.write("---")
    
with st.sidebar:
    class_name = st.selectbox('Select Class', my_classes)
    file_list = audio_paths[class_name]
    audio_file_path = random.choice(file_list)

# Display the spectrogram of the audio file
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

st.write("---")


model = VGG16(weights='imagenet', include_top=True)
resized_mel_spec_rgb = np.stack((resized_mel_spec,) * 3, axis=-1)
resized_mel_spec_rgb = resized_mel_spec_rgb - resized_mel_spec_rgb.min()
resized_mel_spec_rgb = resized_mel_spec_rgb / resized_mel_spec_rgb.max()
resized_mel_spec_rgb = resized_mel_spec_rgb * 255
resized_mel_spec_rgb = np.ceil(resized_mel_spec_rgb).astype(np.uint8)

maps = generate_saliency_map(model, resized_mel_spec_rgb[np.newaxis,...])
new_maps=[]


for map in maps:
    map = np.abs(map)
    map = map / map.max()
    map = map * 255
    map = np.ceil(map).astype(np.uint8)
    new_maps.append(map)


fig3, ax3 = plt.subplots()
def update(frame):
    ax3.clear()
    x = np.array(new_maps[frame])
    ax3.imshow(x, cmap='viridis')
    ax3.set_title(f"Saliency Map - Convolutional Layer {frame+1}")
    ax3.invert_yaxis()

anim = FuncAnimation(fig3, update, frames = 13, interval = 1250)

with open("myvideo.html","w") as f:
  a = anim.to_html5_video()
  print(a, file=f)
  
HtmlFile = open("myvideo.html", "r")
source_code = HtmlFile.read() 
components.html(source_code, height = 480,width=900)

st.write("---")
st.write("Using the viridis color map, green-yellow indicates high activation and dark-blue indicates low activation.")
