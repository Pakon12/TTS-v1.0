# TTS Installation and Usage Guide

Follow these steps to set up and use the TTS (Text-to-Speech) system:

1. Create a new Conda environment:
   ```
   conda create -n trainTTS python=3.10
   ```

2. Activate the environment:
   ```
   conda activate trainTTS
   ```

3. Clone the TTS repository:
   ```
   git clone https://github.com/coqui-ai/TTS.git
   ```

4. Install requirements:
   ```
   cd TTS
   pip install requirements.txt
   ```

5. Install TTS:
   ```
   pip install TTS
   ```

6. Navigate to the recipes folder in the TTS directory.

7. Create the following folder structure:
   ```
   mySpeech/
   └── glow_tts/
       ├── train_glowtts.py
       ├── thai_cleaners.py
       ├── metadata.csv
       └── wavs/
   ```

8. Download `train_glowtts.py` from https://github.com/Pakon12/TTS-v1.0

9. Download `thai_cleaners.py` from https://github.com/Pakon12/TTS-v1.0

10. Download `metadata.csv` from:
    - https://github.com/Pakon12/TTS-v1.0
    - https://drive.google.com/drive/folders/1lKuAS5vjU8n2VYtbgzGJx0koSfEJ35pJ?usp=sharing

11. Start training:
    ```
    python train_glowtts.py
    ```

12. To monitor training progress, open a new terminal and run:
    ```
    conda activate trainTTS
    tensorboard --logdir=<path of glow_tts>
    ```

13. After training is complete, generate speech:
    ```
    tts --text "สวัสดีครับ" --model_path <path_to_model.pth> --config_path <path_to_config.json> --out_path path/output.wav
    ```

Replace `<path_to_model.pth>` with the actual path to your trained model file, and `<path_to_config.json>` with the path to your configuration file.
