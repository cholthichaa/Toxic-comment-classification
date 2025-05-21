from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime
import uvicorn
import json
from pythainlp.tokenize import word_tokenize
import re
import os
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_token_id = 1  #เริ่มต้นประโยค
pad_token_id = 0  #pad_id ตอนเทรนจะเติมเข้าไปกรณีบางประโยคสั้นกว่าประโยคอื่น
unk_token_id=2 # กรณีไม่พบคำศัพท์จะแทนด้วย 2 นี้



model_dir = 'model'
model = onnxruntime.InferenceSession(os.path.join(model_dir,"sentiment_model.onnx"))

def read_json(fname):
    with open(fname) as f:
        data = json.load(f)
        return data

token_to_id = read_json(os.path.join(model_dir,'token2idx.json'))
ids_to_labs  = read_json(os.path.join(model_dir,'idx2lab.json'))

def tokens_to_ids(tokens):
    out_id = [init_token_id]
    for w in tokens:
        if w in token_to_id.keys():
            out_id.append(token_to_id[w])
        else:
            out_id.append(unk_token_id) #unk word
    if len(out_id)==0:
        return [0]
    return out_id



def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def thai_clean_text(text):
    st = ""
    #สามารถเพิ่มโค้ด สำหรับคลีน ข้อความ เช่น ลบ emoji ออก เป็นต้น
    text = deEmojify(text)
    text = text.replace("\n"," ")
    for w in word_tokenize(text):
        st = st + w + " "

    return  re.sub(' +', ' ', st).strip()


class PredictionInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: PredictionInput):


    #convert text to id
    clean_text = thai_clean_text(input.text)
    token_ids = tokens_to_ids(clean_text.split(' '))
    input_data = np.array([token_ids])

    print('input :',input_data)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: input_data.astype(np.int64)})

    lab_index = np.argmax(result[0],axis=1)
    label = ids_to_labs[str(lab_index[0])]
    print('result :', result,label)
    return {"prediction": label}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=3000, reload=True)