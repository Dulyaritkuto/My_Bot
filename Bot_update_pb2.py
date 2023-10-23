import pickle
from langchain.schema import SystemMessage

import pyttsx3
import webbrowser
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,ViltProcessor, ViltForQuestionAnswering,MarianMTModel, MarianTokenizer,BartTokenizer, BartForConditionalGeneration
import torch
from PIL import Image
import language_tool_python


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import speech_recognition as sr

def get_profile():
    with open(r"profile_bot/profile_balial.txt") as f:
        q = f.read()
    print(q)
    system_message = SystemMessage(content =q, additional_kwargs={}, example=False)
    with open('data/old_conversation.pickle', 'rb') as f21:
        data = pickle.load(f21)


    data.chat_memory.messages.append(system_message)

    updated_content = pickle.dumps(data)
    with open('data/old_conversation.pickle', 'wb') as f22:
        f22.write(updated_content)

def add_knowledge():
    with open(r"profile_bot/profile_balial.txt") as f:
        q = f.read()
    with open(r"profile_bot/electricity bill.txt") as f1:
        elec = f1.read()
    system_message = SystemMessage(content = elec, additional_kwargs={}, example=False)
    with open('data/old_conversation.pickle','rb') as f222:
        old_chat = f222.read()
    data = pickle.loads(old_chat)
    data.chat_memory.messages.append(system_message)
    update = pickle.dumps(data)
    with open('data/old_conversation.pickle','wb') as f3:
        f3.write(update)

def get_date():
    from datetime import datetime
    x = datetime.now()
    day_of_week = x.weekday()
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    current_day = weekdays[day_of_week]

    d = x.day
    m = x.month
    y = x.year
    print(current_day)
    with open('data/old_conversation.pickle', 'rb') as f21:
        content = f21.read()

    data = pickle.loads(content)
    new_system_message = SystemMessage(content=f"today  is {current_day} {d} {months[m - 1]} {y}",
                                       additional_kwargs={}, example=False)
    data.chat_memory.messages.append(new_system_message)

    updated_content = pickle.dumps(data)
    with open('data/old_conversation.pickle', 'wb') as f22:
        f22.write(updated_content)
def new_day():
    from datetime import datetime
    x = datetime.now()
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    d = x.day
    m = x.month
    y = x.year
    with open('data/old_conversation.pickle', 'rb') as f21:
        content = f21.read()
    data = pickle.loads(content)
    new_system_message = SystemMessage(content=f'Today Date :  {d} {months[m - 1]} {y}',
                                       additional_kwargs={}, example=False)
    data.chat_memory.messages.pop()
    data.chat_memory.messages.append(new_system_message)
    updated_content = pickle.dumps(data)
    with open('data/old_conversation.pickle', 'wb') as f22:
        f22.write(updated_content)
def delete_old_day():
    with open('data/old_conversation.pickle', 'rb') as f21:
        content = f21.read()
    data = pickle.loads(content)
    data.chat_memory.messages.pop()
    updated_content = pickle.dumps(data)
    with open('data/old_conversation.pickle', 'wb') as f22:
        f22.write(updated_content)
def get_time():
    import datetime
    current_time = datetime.datetime.now().strftime("%H:%M")
    print("Current time:", current_time)
    return current_time

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

class Book():
    def __init__(self,file_path):
        self.filepath = file_path

    def getfile_path(self):
        return self.filepath
    def read_by_AI(self):
        import pyttsx3
        self.x = pyttsx3.init()
        self.x.say()
        self.x.runAndWait()
    def conclude_that_page(self,page):
        y = BART_summarize(page)
        print(y)
        return y
    def read_from_conclude(self,page):
        y = self.conclude_that_page(page)
        self.x.say(y)
        self.x.runAndWait()
    def go_to_pdf_to_text_website(self):
        import webbrowser
        webbrowser.get().open("https://pdftotext.com/")



def clean_string(input_string):
    characters_to_remove = ['$','#','!','?']
    for char in characters_to_remove:
        input_string = input_string.replace(char, '')
    return input_string

# Usage example

def visual_documnet_anwer(question_for_asking,path_to_documnet):
    import os
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import re
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from datasets import load_dataset
    import torch
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

class Visual_processing():
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning",cache_dir=r"data/model_statedict")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning",cache_dir=r"data/model_statedict")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning",cache_dir=r"data/model_statedict")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = 16
        self.num_beams = 4
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

    def predict_step(self,image_paths):
        images = []
        path  = image_paths
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)
        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds,image_paths

def Visual_Questioner(question,path):
    url = path
    image = Image.open(url)
    text = question
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa",cache_dir=r"data/model_statedict")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa",cache_dir=r"data/model_statedict")
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")
    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
    return model.config.id2label[idx]


class ThaiSpeechrecognition():
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000

    def thai_speech_to_text(self):
        with sr.Microphone() as source:
            print("Speak now...")
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio, language="th")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand your speech.")
        except sr.RequestError:
            print("Sorry, there was an issue with the speech recognition service.")
class MarianTranslate():
    def __init__(self):
        self.model_name = "Helsinki-NLP/opus-mt-th-en"
        self.model = MarianMTModel.from_pretrained(self.model_name,cache_dir=r"data/model_statedict")
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name,cache_dir=r"data/model_statedict")
    def make_predict(self,x):
        input_ids = self.tokenizer.encode(x, return_tensors="pt")
        translated = self.model.generate(input_ids)
        translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text

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

def BART_with_raw_data(data):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=r"data/model_statedict")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", cache_dir=r"data/model_statedict")
    text = data
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=12, max_length=100000, early_stopping=True)
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    print(summary)
    return summary





def correct_sentence(sentence):
    tool = language_tool_python.LanguageTool('en-US', )  # Specify the language
    matches = tool.check(sentence)
    q = tool.correct(sentence)
    return q
# Future_AI
def draw_image(prompt_t,save_image =False):
    from diffusers import StableDiffusionPipeline
    import matplotlib.pyplot as plt
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


def website_131():
    webbrowser.get().open("https://sites.google.com/view/siriraj131official/archives/sophomore/228229")


if __name__ =="__main__":
    get_profile()
    from datasets import load_dataset
    # s = song()
    # print(s.get_list_song())
    # s.play_song("another love")
    # vq = Visual_processing()
    # x = vq.predict_step(['2bird.jpg'])
    # t = ThaiSpeechrecognition()
    # q = t.thai_speech_to_text()
    # print(q)
    # m = MarianTranslate()
    # tran = m.make_predict(q)
    # print(tran)
    # draw_image("the picture of a man with his axe chop the wood")
    # b = Book("data")
    # b.go_to_pdf_to_text_website()
    # print(x)
    # Visual_Questioner(question="what is the color of this animal",path="snake3.jpg")
    # visual_documnet_anwer(question_for_asking="what is the full code of CNN?",path_to_documnet=r"D:\Balial\data\ilovepdf_pages-to-jpg (1)\AI_and_Machine_Learning_for_Coders_A_Programmers_Guide_to_Artificial_Intelligence_by_Laurence_Moroney_z-lib_org_page-0062.jpg")


from langchain.document_loaders import TextLoader
# import torch
# from timeout_decorator import timeout
# from transformers import GenerationConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model_id = "mrm8488/falcoder-7b"
# cache_directory = r"data\falcoder"
#
#
# tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_directory,trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_id,cache_dir=cache_directory,trust_remote_code=True)
#
#
#
# def generate(
#         instruction,
#         max_new_tokens=128,
#         temperature=0.1,
#         top_p=0.75,
#         top_k=40,
#         num_beams=4,
#         **kwargs
# ):
#     prompt = instruction + "\n### Solution:\n"
#     print(prompt)
#     inputs = tokenizer(prompt, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]
#     generation_config = GenerationConfig(
#         temperature=temperature,
#         top_p=top_p,
#         top_k=top_k,
#         num_beams=num_beams,
#         **kwargs,
#     )
#     with torch.no_grad():
#         generation_output = model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             generation_config=generation_config,
#             return_dict_in_generate=True,
#             output_scores=True,
#             max_new_tokens=max_new_tokens,
#             early_stopping=True
#         )
#     s = generation_output.sequences[0]
#     output = tokenizer.decode(s)
#     return output.split("### Solution:")[1].lstrip("\n")
#
# instruction = "Design a class for representing a person in Python."
# print(generate(instruction))

# from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
# from PIL import Image
# import requests
#
#
# processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# import torch
# import torchvision.transforms.functional as F
# from PIL import Image
# from torchvision.models.segmentation import deeplabv3_resnet50
#
# image = F.resize(image, (512, 512))  # Resize the image to desired dimensions
# image_tensor = F.to_tensor(image)  # Convert the image to a tensor
#
# # Load the segmentation model
# model = deeplabv3_resnet50(pretrained=True)
#
# # Set the model to evaluation mode
# model.eval()
#
# # Perform inference on the image
# output = model(torch.unsqueeze(image_tensor, 0))
#
# # Access the predicted segmentation mask
# segmentation_mask = output["out"].argmax(1)
#
# # inputs = feature_extractor(images=image, return_tensors="pt")
# outputs = model(**segmentation_mask)
# logits = outputs.logits
