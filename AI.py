import pyttsx3
import pickle
import requests
from bs4 import BeautifulSoup
import re
from transformers import BartTokenizer, BartForConditionalGeneration
import webbrowser
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
class song():
    def __init__(self):
        self.l = {"royalty":"https://www.youtube.com/watch?v=oOi3oJmfz4o&list=RDtBaEwlR8KDw&index=2",
             "another love":"https://www.youtube.com/watch?v=MwpMEbgC7DA&list=RDtBaEwlR8KDw&index=3",
             "rise":"https://www.youtube.com/watch?v=fB8TyLTD7EE&list=RDtBaEwlR8KDw&index=4",
             "andrew tate song":"https://www.youtube.com/watch?v=rBR1nwYIFck&list=RDtBaEwlR8KDw&index=6",
             "alor": "https://www.youtube.com/watch?v=tBaEwlR8KDw",
             "ava":"https://www.youtube.com/watch?v=LjY_AOtDMRg",
            "indila":"https://www.youtube.com/watch?v=vtNJMAyeP0s&list=RDtBaEwlR8KDw&index=13",
             "dynasty":"https://www.youtube.com/watch?v=5-ZiKXrnvog&list=RDtBaEwlR8KDw&index=18",
             "mary on cross":"https://www.youtube.com/watch?v=k5mX3NkA7jM",
             "ghost":"https://www.youtube.com/watch?v=8aKMHe_sRpc",
             "gangster paradise":"https://www.youtube.com/watch?v=LyeWUMJpTgw",
             "fracture":"https://www.youtube.com/watch?v=38750lf-u5w",
             "deadwood":"https://www.youtube.com/watch?v=T8BI2fKzdys",
             "lilly":"https://www.youtube.com/watch?v=ox4tmEV6-QU"}
    def get_list_song(self):
        return self.l.keys()
    def play_song(self,song):
        speak = pyttsx3.init()
        self.p =[]
        for x in self.l.keys():
            self.p.append(x)
        if song.lower() not in self.p:
            print("I do not have that song in my memory")
            speak.say("I do not have that song in my memory")
            speak.runAndWait()
        else:
            webbrowser.get().open(self.l[song.lower()])

def visual_documnet_anwer(question_for_asking,path_to_documnet):
    import os
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    import matplotlib.pyplot as plt

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",
                                               cache_dir=r"data/model_statedict")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",
                                                      cache_dir=r"data/model_statedict")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    image = plt.imread(path_to_documnet)
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    question = question_for_asking
    prompt = task_prompt.replace("{user_input}", question)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    print(processor.token2json(sequence))


def get_time():
    import datetime
    current_time = datetime.datetime.now().strftime("%H:%M")
    print("Current time:", current_time)
    return current_time

def BART_with_raw_data(data):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=r"data/model_statedict")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", cache_dir=r"data/model_statedict")
    text = data
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=12, max_length=100000, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    print(summary)
    return summary

def BART_summarize(path):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=r"data/model_statedict")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", cache_dir=r"data/model_statedict")

    with open(path) as f:
        text = f.read()
    input_text = text
    inputs = tokenizer(input_text, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=1000, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    print(summary)
    return summary


class searcher():
    def __init__(self):
        self.say_ = pyttsx3.init()


    def say(self,data):
        self.say_.say(data)
        self.say_.runAndWait()

    def check_keyword(self,x):
        with open(r"data/data/word_memmory.pickle", "rb") as f1:
            search_key_word = pickle.load(f1)

        self.serach_key = search_key_word
        word = word_tokenize(x)
        trigrams = list(ngrams(word, 3))
        bigram = list(ngrams(word, 2))
        quardgram = list(ngrams(word, 4))

        for w in word:
            if w in search_key_word:
                return True
        match_tri_gram = set(trigrams) & set(search_key_word)
        match_bi_gram = set(bigram) & set(search_key_word)
        match_quadragram = set(quardgram) & set(search_key_word)
        if match_bi_gram:
            return True
        elif match_quadragram:
            return True
        elif match_tri_gram:
            return True
        else:
            return False

    def remove_text(self,text):

        remove_text =['what','is','i','want','known','know','wondering','show','me','do','you','show','some','data','about']

        search_text = [word for word in text if word not in remove_text]
        print(text)

        query_key = " ".join(search_text)

        print(query_key)

        return query_key

    def query_wikipedia(self,user_query):
        # Google

        URL = "https://www.google.co.in/search?q=" + user_query
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        page = requests.get(URL, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        soup = str(soup)
        any = r'https://en.wikipedia.org/wiki/[^"\']+'
        x = re.findall(any,soup)
        print(x)
        w = int(input("type your index to get data : "))
        url =x[w]
        result = requests.get(url)
        doc = BeautifulSoup(result.text,"html.parser")
        paragraphs = []
        content_div = doc.find('div', {'id': 'mw-content-text'})

        for paragraph in content_div.find_all('p'):
            paragraphs.append(paragraph.text)
        paragraphs = paragraphs[1:]
        result  = " ".join(paragraphs)
        modified_data = re.sub(r'\[\d+\]', '', result)
        print(modified_data)
        return modified_data


def draw_image(prompt_t,save_image =False):
    from diffusers import StableDiffusionPipeline

    from keras.models import load_model
    from nltk.tokenize import word_tokenize
    from keras.utils import pad_sequences
    import tensorflow as tf
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32,cache_dir=r"data/model_statedict")
    prompt = prompt_t
    image = pipe(prompt).images[0]
    plt.imshow(image)
    plt.show()
    if save_image:
        image.save(f"{prompt}.png")
    else:
        with open("data/word/word_to_index_yes_no.pkl", "rb") as f1:
            word_to_index = pickle.load(f1)
        x = input("do you want to save image ?")
        x = x.lower()
        x_input = word_tokenize(x)
        input_list = []
        for i in x_input:
            input_list.append(word_to_index[i])
        print(input_list)
        model = load_model(filepath="data/model_statedict/yes_no_model.h5")

        input_list = tf.expand_dims(input_list, axis=0)
        input_list = pad_sequences(input_list, maxlen=40, padding="post")
        m = model.predict(input_list)
        y = m.argmax()
        if y==0:
            image.save(f"{prompt}.png")
        else:
            print("keep that in mind your image does not store this time")
    return image

text_speech = pyttsx3.init()
local_path = (
    "data/model_statedict/ggml-gpt4all-l13b-snoozy.bin"  # replace with your desired local file path
)
from langchain import ConversationChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import SystemMessage,HumanMessage
import tensorflow as tf
from nltk.tokenize import word_tokenize
from keras.utils import pad_sequences
from nltk.util import ngrams
from Balial_agents import searcher

loaded_model = tf.keras.models.load_model(r"model/text_classifier_beta1.3.h5")
with open("data/word/word_to_index.pkl", "rb") as f1:
    word_to_index = pickle.load(f1)
print(word_to_index)

callbacks = [StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True,backend="gptj")
searcher = searcher()
conversation = ConversationChain(llm=llm,verbose=True)
text_talk = pyttsx3.init()
text_talk.say("what do you want me to help you today")
text_talk.runAndWait()
while True:
    x = input("\ntype Input here : ===============================\n")
    from Bot_update_pb2 import correct_sentence

    seach_output = searcher.check_keyword(x)
    y = correct_sentence(x)
    z = y.lower()
    x_tokenize = word_tokenize(z)
    input_model = []

    if seach_output:
        x = searcher.remove_text(x_tokenize)
        searcher.query_wikipedia(x)
        continue
    try:
        for i in x_tokenize:
            input_model.append(word_to_index[i])
        input_model = tf.convert_to_tensor(input_model)
        input_model = tf.expand_dims(input_model, axis=0)
        input_model = pad_sequences(input_model, padding="post", maxlen=40)
        y_predic = loaded_model.predict(input_model)
        s = tf.squeeze(tf.round(y_predic.argmax()))
        print(s)


        if  s == 1:


            so = input("what is the song do you like me to play :(if this song do not exist in my memory you can type i for online searching :) :")

            songs = song()
            if so =="give me some plasylist" or so=="playlist":
                print(s.get_list_song())
                so =input("input your song : ")
                songs.play_song(so)
                continue
            elif so=="i" or "internet":
                internet_search = input("type your song : ")
                searcher.query_youtube(internet_search)

            else:
                songs.play_song(so)
                continue
        elif s==2:
            import stock_fish as sf
        #     pass
        elif s==3:

            x = get_time()
            text_talk.say(f"Current time is {x}")
            text_talk.runAndWait()
        elif s==4:
            import tkinter as tk
            from tkinter import filedialog
            question_for_visual_data = input("enter your question")
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()
            print("Selected file:", file_path)
            root.destroy()
            visual_documnet_anwer(question_for_asking=question_for_visual_data,path_to_documnet=file_path)
        elif s==5 or s==0:

            import pretrained_Vit as pv
            import tkinter as tk
            from tkinter import filedialog
            print("please choose your file : ")
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()
            print("Selected file:", file_path)
            root.destroy()
            x26 =plt.imread(file_path)
            x27 = plt.imshow(x26)
            plt.show()
            text_talk.say("this your file right?,if not your file you can type n or exit to get out and restart the program")
            text_talk.runAndWait()
            x = input("that your file ? (y/n)")
            if x.lower()=="n" or x.lower()=="no":
                continue
            out_put_1 = pv.predict_with_vit_base(file_path)
            out_put_2 = pv.predict_with_resnet_50(file_path)
            print("model_1 predict :" ,out_put_1)
            print("model_2 predict :",out_put_2)
            from Bot_update_pb2 import Visual_processing
            from Bot_update_pb2 import Visual_Questioner
            v_q = Visual_processing()
            v_q_output,image_path = v_q.predict_step([file_path])
            print(v_q_output)
            v_q_image_anser =input("if you have question please type here")
            system = SystemMessage(content = f"AI just process and tell the user about image of {out_put_1}", additional_kwargs={}, example=False)
            conversation.memory.chat_memory.messages.append(system)

            if v_q_image_anser.lower() =="exit" or v_q_image_anser.lower() =="quit" or v_q_image_anser.lower() =="no":
                continue
            else:
                answer = Visual_Questioner(question=v_q_image_anser,path=file_path)
                print(answer)
                continue
        elif s==6:
            import tkinter as tk
            from tkinter import filedialog
            from Bot_update_pb2 import BART_summarize
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()
            print("Selected file:", file_path)
            root.destroy()
            x = BART_summarize(file_path)
            system = SystemMessage(content =f"AI have summarized the text to the user here summarized text {x}", additional_kwargs={}, example=False)
            conversation.memory.chat_memory.messages.append(system)
            continue
        elif s==7:
            d_i = input("type something you wanna draw : ")
            image_from_AI = draw_image(prompt_t=d_i)
            system_message = SystemMessage(content =f"AI have drawed image to user , Image name : {d_i}", additional_kwargs={}, example=False)
            conversation.memory.chat_memory.messages.append(system_message)
            continue

    # elif x.lower =="chat":
    #     x = input("type me some text to talk to me : ")

    #
    except(KeyError):
        text_speech.say(conversation.run(x))
        text_speech.runAndWait()