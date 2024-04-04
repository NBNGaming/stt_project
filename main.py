import queue
from io import BytesIO

import torch
import torchaudio as ta
from df.enhance import enhance, init_df
from faster_whisper import WhisperModel
from ffmpeg import FFmpeg
from telegram.ext import Updater, MessageHandler, Filters

import config
import utils


class Task:
    def __init__(self, chat_id, message_id, file_id=None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.file_id = file_id

    def download_file(self):
        tg_file = updater.bot.get_file(self.file_id)
        file = BytesIO()
        tg_file.download(out=file)
        return file.getvalue()


def on_audio(update, context):
    new_task = Task(update.effective_chat.id, update.message.message_id)
    if update.message.voice:
        new_task.file_id = update.message.voice.file_id
    elif update.message.audio:
        new_task.file_id = update.message.audio.file_id
    else:
        new_task.file_id = update.message.document.file_id
    task_queue.put(new_task)
    context.bot.send_message(
        chat_id=new_task.chat_id,
        reply_to_message_id=new_task.message_id,
        text=f'Файл принят на обработку! Ваше место в очереди: {task_queue.qsize()}'
    )


def info(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f'Это бот для распознавания речи. Пришлите голосовое сообщение или аудиофайл, чтобы начать работу.'
    )


def load_audio_bytes(b):
    file = BytesIO(b)
    audio, _ = ta.load(file)
    return audio.contiguous()


def save_audio_bytes(audio):
    file = BytesIO()
    audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        audio.unsqueeze_(0)
    if audio.dtype != torch.int16:
        audio = (audio * (1 << 15)).to(torch.int16)
    ta.save(file, audio, df_state.sr(), format='wav')
    file.seek(0)
    return file


updater = Updater(config.TG_BOT_TOKEN, use_context=True)
updater.dispatcher.add_handler(MessageHandler(
    Filters.voice | Filters.audio | Filters.document.category('audio/'),
    on_audio
))
updater.dispatcher.add_handler(MessageHandler(Filters.all, info))
task_queue = queue.Queue()
df_model, df_state, _ = init_df()
whisper = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE)
# Нормализуем громкость и меняем частоту дискретизации на нужную для DeepFilterNet
ffmpeg = (
    FFmpeg()
    .option('y')
    .input('pipe:0')
    .output(
        'pipe:1',
        {'codec:a': 'pcm_s16le', 'filter:a': 'loudnorm'},
        ar=df_state.sr(),
        f='wav'
    )
)

updater.start_polling()
while True:
    current_task = task_queue.get()
    try:
        audio_bytes = current_task.download_file()
        normalized = ffmpeg.execute(audio_bytes)

        audio_tensor = load_audio_bytes(normalized)
        cleaned_tensor = enhance(df_model, df_state, audio_tensor)
        cleaned = save_audio_bytes(cleaned_tensor)

        segments, info = whisper.transcribe(cleaned, vad_filter=True)
        segments = [segment.text for segment in segments]
        utils.remove_hallucinations(segments, info.language)
        text = ''.join(segments).strip()

        updater.bot.send_message(
            chat_id=current_task.chat_id,
            reply_to_message_id=current_task.message_id,
            text=f'✅ Распознанная речь:\n{text}'
        )
    except Exception:
        updater.bot.send_message(
            chat_id=current_task.chat_id,
            reply_to_message_id=current_task.message_id,
            text='❌ Не удалось выполнить обработку файла'
        )
