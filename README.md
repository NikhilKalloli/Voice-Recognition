# Voice Recognition 
This project uses a pre trained speech embedding model for speaker recognition using TensorFlow 2.4 and TensorFlow 2.5. The model achieves over 98% accuracy in identifying whether two speech samples belong to the same person without storing any personal voice samples.



https://github.com/NikhilKalloli/Voice-Recognition/assets/123582746/994f5b86-7cde-4c66-86ce-c6c2971d8526



## Getting Started

1. Clone the repository:

   ```
   git clone https://github.com/NikhilKalloli/Voice-Recognition.git
   ```
2. Navigate to the directory:
    ```
    cd Voice-Recognition
    ```

3. Create a Virtual Environment with python version 3.8
    ```
    py -3.8 -m venv venv
    ```
   ###### If you don't have python 3.8 installed, you can download it from [here](https://www.python.org/downloads/release/python-380/)

4. Install the required dependencies(in a virtual Environment):
    ```
    pip install -r requirements.txt
    ```

5. Run the Streamlit app:
    ``` 
    streamlit run streamlit_app.py
    ```

## Working
#### **Model Architecture:** 
This speech embedding model architecture utilizes a stack of two LSTM layers following the computation of the mel-frequency spectrogram from the input audio data in the time domain. The model architecture includes a Dense layer that outputs the final audio embeddings.The total number of parameters is 641k.

#### **Dataset:**
In this work, the [Mozilla Common Voice dataset](https://commonvoice.mozilla.org/en/datasets) was used. It contains a large amount of voice samples, including client_id, the audio file, the sentence that was spoken and some features about the speaker (age, gender, etc).

#### **Training:** 
Training involves preprocessing audio samples, utilizing triplet loss, and training on a large dataset.

#### **Applications:** 
This model enables identity verification from speech, offering applications in audio diarization, identity verification, and transfer learning.


## Contributing
Contributions are welcome! If you have any improvements or new features to suggest, please create a pull request.
If you have any questions or issues, feel free to [open an issue](https://github.com/NikhilKalloli/Voice-Recognition/issues).


