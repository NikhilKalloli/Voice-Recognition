import streamlit as st
import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import load_model
from train_speech_id_model import *
import tempfile
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Voice Recognition App")



if os.path.isfile('speech-id-model-110/saved_model.pb'):
    model = load_model('speech-id-model-110')
else:
    print("Error: Speech identification model not found!")

target_rate = 48000

def get_audio_embedding(audio_file):
    cur_data = tfio.audio.AudioIOTensor(audio_file)
    audio_data = cur_data.to_tensor()[:, 0]
    if cur_data.rate != target_rate:
        audio_data = tfio.audio.resample(
            audio_data,
            tf.cast(cur_data.rate, tf.int64),
            tf.cast(target_rate, tf.int64),
        )
    embedding = model.predict(tf.expand_dims(audio_data, axis=0))[0]
    return embedding

def main():
    st.title("Voice Recognition App")

    # Create a temporary directory to store uploaded files
    temp_dir = tempfile.mkdtemp()
    demo_folder = os.path.join(temp_dir, 'demo')
    os.makedirs(demo_folder, exist_ok=True)

    uploaded_files = st.file_uploader("Upload two MP3 files", type="mp3", accept_multiple_files=True)

    if uploaded_files is not None and len(uploaded_files) == 2:
        st.write("Files uploaded successfully!")

        file1, file2 = uploaded_files

        # Save the uploaded files to the demo folder
        file1_path = os.path.join(demo_folder, file1.name)
        file2_path = os.path.join(demo_folder, file2.name)
        with open(file1_path, 'wb') as f1, open(file2_path, 'wb') as f2:
            f1.write(file1.getbuffer())
            f2.write(file2.getbuffer())

        st.audio(file1, format='audio/mp3')
        st.audio(file2, format='audio/mp3')

        # Get embeddings for both audio files
        embedding1 = get_audio_embedding(file1_path)
        embedding2 = get_audio_embedding(file2_path)

        # Calculate distance between embeddings
        distance = np.linalg.norm(embedding1 - embedding2)
        print(f"Distance between embeddings: {distance}")

        threshold = 0.83
        if distance < threshold:
            conclusion = 'Same person'
            st.success(f"Conclusion: {conclusion}")
        else:
            conclusion = 'Different people'
            st.error(f"Conclusion: {conclusion}")

    else:
        st.error("Please upload two MP3 files to compare.")


    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
