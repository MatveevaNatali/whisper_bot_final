import whisper

from pydub import AudioSegment

from pyannote.core import Segment, notebook
from pyannote.audio import Audio
from IPython.display import Audio as IPythonAudio

from huggingface_hub import notebook_login
notebook_login()

from pyannote.audio import Pipeline
import torch
from pyannote.core import Annotation

from huggingface_hub import HfApi
import re

from openai import OpenAI
import tiktoken

import json
  


from telegram.ext import Updater, Filters, MessageHandler, CommandHandler
from telegram import ReplyKeyboardMarkup


updater = Updater(token='7043855919:AAGP1KiM75ss-a-9u7i8L5BHzYytc0W4KIc')

URL = 'https://api.thecatapi.com/v1/images/search'


'''
def get_new_image():
    response = requests.get(URL).json()
    random_cat = response[0].get('url')
    return random_cat

def new_cat(update, context):
    chat = update.effective_chat
    context.bot.send_photo(chat.id, get_new_image())

def wake_up(update, context):
    chat = update.effective_chat
    name = update.message.chat.first_name
    # За счёт параметра resize_keyboard=True сделаем кнопки поменьше
    button = ReplyKeyboardMarkup([['/newcat']], resize_keyboard=True)

    context.bot.send_message(
        chat_id=chat.id,
        text='Привет, {}. Посмотри, какого котика я тебе нашёл'.format(name),
        reply_markup=button
    )

    context.bot.send_photo(chat.id, get_new_image())

'''
def convert_ogg(file_path):
    audio = AudioSegment.from_file(file_path, format="ogg")
    new_file_path = f'output-{file_path}-new.wav'
    audio.export(new_file_path, format="wav") 
    print('сохранено в формате wav')
    return new_file_path

def wake_up(update, context):
    chat = update.effective_chat
    name = update.message.chat.first_name
    context.bot.send_message(chat_id=chat.id, text='Привет, {}, я Voice to text bot!'.format(name))

def get_audio_voice(update, context):
    folder = 'voice'
    chat = update.effective_chat
    context.bot.send_message(chat_id=chat.id, text='Была загружена голосовуха')

    new_file = context.bot.get_file(update.message.voice)
    extension = '.wav' # default extension
    body_name = (new_file.file_path).split("/")[-1] # доработать нормально
    file_name = folder + "-" + body_name 
    new_file.download(file_name)
    context.bot.send_message(chat_id=chat.id, text="Downloaded file: {}".format(file_name))

    reference = convert_ogg(file_name) # это новая ссылка

# я конечно сейчас вставлю этот кусок, но его нужно переписать
    
    model_name = 'large'
    model = whisper.load_model(model_name)
    result = model.transcribe(reference)
    print(result["text"])
    context.bot.send_message(chat_id=chat.id, text=result["text"])



def get_audio(update, context):
    folder = 'folder'
    chat = update.effective_chat
    context.bot.send_message(chat_id=chat.id, text='Была загружена аудиозапись')
    #context.bot.get_file(update.message.document).download()


    new_file = context.bot.get_file(update.message.document)
    extension = (new_file.file_path).split(".")[-1] # get file extension of received file
    body_name = (new_file.file_path).split("/")[-1] # доработать нормально
    file_name = folder + "-" + body_name + "." + extension
    new_file.download(file_name)
    context.bot.send_message(chat_id=chat.id, text="Downloaded file: {}".format(file_name))
    

    path = file_name #полный путь
            
    AUDIO_FILE = path
    pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',
    use_auth_token='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')

    available_pipelines = [p.modelId for p in HfApi().list_models(filter="pyannote-audio-pipeline")]
    list(filter(lambda p: p.startswith("pyannote/"), available_pipelines))
    dia = pipeline(AUDIO_FILE) 

    assert isinstance(dia, Annotation)
     
    text_inner = []
    for speech_turn, track, speaker in dia.itertracks(yield_label=True):
        text_inner.append(f"{speech_turn.start:4.1f} {speech_turn.end:4.1f} {track} {speaker}/")
    
    context.bot.send_message(chat_id=chat.id, text=text_inner)


    DEMO_FILE = {'uri': 'blabla', 'audio': AUDIO_FILE}
    dz = pipeline(DEMO_FILE)

    with open("diarization.txt", "w") as text_file:
        text_file.write(str(dz))


    print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")
    def millisec(timeStr):
                spl = timeStr.split(":")
                s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
                return s
            
    dzs = open('diarization.txt').read().splitlines()

    groups = []
    g = []
    lastend = 0

    for d in dzs:
        if g and (g[0].split()[-1] != d.split()[-1]):      #same speaker
            groups.append(g)
            g = []

        g.append(d)

        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
        end = millisec(end)
        if (lastend > end):       #segment engulfed by a previous segment
            groups.append(g)
            g = []
        else:
            lastend = end
    if g:
        groups.append(g)
    #print(*groups, sep='\n')

    audio = AudioSegment.from_wav(AUDIO_FILE)
    gidx = -1
    for g in groups:
        start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
        end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
        start = millisec(start) #- spacermilli
        end = millisec(end)  #- spacermilli
        gidx += 1
        audio[start:end].export(str(gidx) + '.wav', format='wav')
        #print(f"group {gidx}: {start}--{end}")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model('large-v3', device = device)

    import json


    transcribe_output = []
    for i in range(len(groups)):
        audiof = str(i) + '.wav'
        result = model.transcribe(audio=audiof, language='ru', word_timestamps=True)

        a = (groups[i][0][-11:])
        b = (result[list(result.keys())[0]])
        c = (f" {a}: {b}")
        #print(c)
        transcribe_output.append(c)

    with open("transcribation.txt", "w", encoding="utf-8") as text_file:
        text_file.write(str(transcribe_output))


    api_key = "sk-proj-XUAr5yzc2l2FrCRCTO9bT3BlbkFJ4jEcd3x2OGpf92O9QkPk"
    client = OpenAI(api_key = api_key)

    def sendToGpt(model, messages):
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens = 2500
        )
        return chat_completion.choices[0].message.content

    def processText(
        prompt=None,
        text_data=None,
        chat_model=
        # "gpt-4",
        "gpt-3.5-turbo",
        model_token_limit=8192,
        max_tokens=2500
    ):
        if not prompt:
            return "Error: Prompt is missing. Please provide a prompt."
        if not text_data:
            return "Error: Text data is missing. Please provide some text data."

        # Initialize the tokenizer
        tokenizer = tiktoken.encoding_for_model(chat_model)

        # Encode the text_data into token integers
        token_integers = tokenizer.encode(text_data)

        # Split the token integers into chunks based on max_tokens
        chunk_size = max_tokens - len(tokenizer.encode(prompt))
        chunks = [
            token_integers[i : i + chunk_size]
            for i in range(0, len(token_integers), chunk_size)
        ]

        # Decode token chunks back to strings
        chunks = [tokenizer.decode(chunk) for chunk in chunks]
        responses = []
        messages = [
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": "Чтобы пояснить контекст к запросу, я буду присылать текст частями. Когда я закончу, я напишу тебе 'ВСЕ ЧАСТИ ВЫСЛАНЫ'. Не отвечай пока не получишь все части.",
            },
        ]

        for chunk in chunks:
            messages.append({"role": "user", "content": chunk})
            # Check if total tokens exceed the model's limit and remove oldest chunks if necessary
            while (sum(len(tokenizer.encode(msg["content"])) for msg in messages) > model_token_limit):
                messages.pop(1)

        messages.append({"role": "user", "content": "ВСЕ ЧАСТИ ВЫСЛАНЫ"})

        response = sendToGpt(model=chat_model, messages=messages)
        final_response = response.strip()
        responses.append(final_response)

        return responses

    print_split_text = lambda s: [print(part) for part in s.split('\n')]

    raw_text = str(transcribe_output)

    promt_text = processText(prompt = "Выведи мне абзацы: 0. Тема диалога очень кратко, 1. суть текста, 2. очень краткую суть текста, 3. выжимку по каждому говорящему", text_data = raw_text)
    print_split_text(promt_text[0])

    #print(promt_text[0])

    context.bot.send_message(chat_id=chat.id, text=promt_text[0])
    

    



def echo(update, context) -> None:
    """Echo the user message."""
    chat = update.effective_chat
    update.message.reply_text(update.message.text)

# def get_audio_messages(message):
    
#     file_info = bot.get_file(message.voice.file_id)
#     downloaded_file = bot.download_file(file_info.file_path)
#     with open('user_voice.mp3', 'wb') as new_file:
#         new_file.write(downloaded_file)


updater.dispatcher.add_handler(CommandHandler('start', wake_up))

updater.dispatcher.add_handler(MessageHandler(Filters.text, echo)) # этот хендлер реагирует на текстовые сообщения
updater.dispatcher.add_handler(MessageHandler((Filters.document), get_audio)) # этот хендлер реагирует на аудио
updater.dispatcher.add_handler(MessageHandler(Filters.voice, get_audio_voice))
updater.start_polling()
updater.idle() 