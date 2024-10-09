import os
import torch

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig 

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig,BaseAudioConfig, CharactersConfig 

from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from thai_cleaners import thai_cleaners

# we use the same path as this script as our training folder.
output_path = os.path.dirname(os.path.abspath(__file__))

# DEFINE DATASET CONFIG
# Set LJSpeech as our target dataset and define its path.
# You can also use a simple Dict to define the dataset and pass it to your custom formatter.
dataset_config = BaseDatasetConfig(
    formatter="vctk", 
    meta_file_train="metadata.csv", 
    language="th", 
    path="ที่อยู่ของ path metadata.csv",
)



audio_config = BaseAudioConfig(
    sample_rate=16000
)


characters_config = CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",  # แก้ไขจาก THS เป็น TTS
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    characters="!¡'(),-.:;¿?abcdefghijklmnopqrstuvwxyzµßàáâãäåæçèéêëìíîïñòóôõöùúûüąćęłńœśşźżƒабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґӧ «°±µ»$%&‘’‚“`”„กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤฦะัาำิีึืุูเแโใไๅํ็่้๊๋ฯฺๆ์ํ๎",
    punctuations="!¡'(),-.:;¿? ",
    phonemes=None,
)
# INITIALIZE THE TRAINING CONFIGURATION
# Configure the model. Every config class inherits the BaseTTSConfig.
config = GlowTTSConfig(
    run_name="GlowTTS-Thai",
    run_description='Thai TTS using GlowTTS',
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="thai_cleaners",  # เปลี่ยนเป็นสตริง
    use_phonemes=False,
    phoneme_language="th",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    audio=audio_config,
    characters=characters_config,
    test_sentences=[
        "สวัสดีครับ นี่คือการทดสอบระบบสังเคราะห์เสียงภาษาไทย",
        "ฉันกำลังเรียนรู้การสร้างระบบ text-to-speech สำหรับภาษาไทย"
    ],
)

from TTS.tts.utils.text import cleaners
cleaners.thai_cleaners = thai_cleaners
# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.


def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[1]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items


# load training samples
# train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=formatter,
)

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.

trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
if __name__ == '__main__':
    trainer.fit()