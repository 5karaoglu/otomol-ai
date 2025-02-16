from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import random
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import speech_recognition as sr
from gtts import gTTS
import os
import base64
import tempfile
import logging
import subprocess
import argparse
import uvicorn

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM model ve tokenizer yükleme
MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# GPU kullanılabilirse modeli GPU'ya taşı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Text generation pipeline oluştur
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm originlere izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU kontrolü
print("GPU kullanılabilir mi:", torch.cuda.is_available())
print("Kullanılabilir GPU sayısı:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Kullanılan GPU:", torch.cuda.get_device_name(0))

# Sabit değerler
BOT_NAME = "OtomolAi"
USER_NAME = "Osman Bey"
MARKALAR = ["Mercedes", "BMW", "Audi", "Volkswagen"]
URETIM_YERLERI = ["İstanbul", "Ankara", "İzmir", "Bursa"]

# Veritabanı
DATABASE = {}

# Başlangıçta veritabanını yükle
try:
    with open("example_database.json", "r") as f:
        DATABASE.update(json.load(f))
    print("Veritabanı başarıyla yüklendi")
except Exception as e:
    print(f"Veritabanı yükleme hatası: {str(e)}")

def generate_context():
    context = ""
    for ay in DATABASE:
        for gun in DATABASE[ay]:
            for kayit in DATABASE[ay][gun]:
                context += f"{ay} ayı {gun}. günü {kayit['marka']} markası {kayit['uretim_yeri']} üretim yerinde {kayit['yukleme_adedi']} adet yükleme yapmıştır. "
    
    # Ek bilgiler
    context += f"Desteklenen markalar: {', '.join(MARKALAR)}. "
    context += f"Üretim yerleri: {', '.join(URETIM_YERLERI)}. "
    return context

@app.post("/upload-database")
async def upload_database(file: UploadFile = File(...)):
    try:
        content = await file.read()
        DATABASE.update(json.loads(content.decode()))
        return {"message": "Veritabanı başarıyla yüklendi"}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Geçersiz JSON formatı")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("Yeni WebSocket bağlantı denemesi")
    try:
        await websocket.accept()
        logger.info("WebSocket bağlantısı başarıyla kabul edildi")
        
        while True:
            try:
                # Ses verisini al
                data = await websocket.receive_text()
                logger.info("WebSocket üzerinden veri alındı")
                
                # Hoşgeldin mesajı kontrolü
                try:
                    json_data = json.loads(data)
                    if json_data.get('type') == 'welcome':
                        logger.info("Hoşgeldin mesajı alındı")
                        welcome_message = f"Merhaba {USER_NAME}! Ben {BOT_NAME}. Size nasıl yardımcı olabilirim?"
                        # Metni sese çevir
                        tts = gTTS(text=welcome_message, lang='tr')
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_voice:
                            tts.save(temp_voice.name)
                            with open(temp_voice.name, "rb") as audio_file:
                                audio_base64 = base64.b64encode(audio_file.read()).decode()
                        
                        # Cevabı gönder
                        await websocket.send_json({
                            "text": welcome_message,
                            "audio": f"data:audio/mp3;base64,{audio_base64}"
                        })
                        logger.info("Hoşgeldin mesajı gönderildi")
                        
                        # Geçici dosyayı temizle
                        os.unlink(temp_voice.name)
                        continue

                except json.JSONDecodeError as e:
                    logger.error(f"JSON ayrıştırma hatası: {str(e)}")
                    pass

                try:
                    # Base64'ten ses verisini çöz
                    audio_data = data.split(",")[1]
                    audio_bytes = base64.b64decode(audio_data)
                    logger.info("Ses verisi başarıyla decode edildi")
                    
                    # Geçici WebM dosyası oluştur
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
                        temp_webm.write(audio_bytes)
                        temp_webm_path = temp_webm.name

                    # WebM'den WAV'a dönüştür
                    temp_wav_path = temp_webm_path.replace(".webm", ".wav")
                    subprocess.run(['ffmpeg', '-i', temp_webm_path, '-acodec', 'pcm_s16le', '-ar', '44100', temp_wav_path])
                    logger.info("Ses dosyası WAV formatına dönüştürüldü")
                    
                    # Ses tanıma
                    recognizer = sr.Recognizer()
                    try:
                        with sr.AudioFile(temp_wav_path) as source:
                            audio = recognizer.record(source)
                            soru = recognizer.recognize_google(audio, language="tr-TR")
                            logger.info(f"Tanınan ses: {soru}")
                            
                            # Soruyu analiz et ve cevap oluştur
                            cevap = await process_query(soru)
                            logger.info(f"Oluşturulan cevap: {cevap}")
                            
                            # Metni sese çevir
                            tts = gTTS(text=cevap, lang='tr')
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_voice:
                                tts.save(temp_voice.name)
                                with open(temp_voice.name, "rb") as audio_file:
                                    audio_base64 = base64.b64encode(audio_file.read()).decode()
                            
                            # Cevabı gönder
                            await websocket.send_json({
                                "text": cevap,
                                "audio": f"data:audio/mp3;base64,{audio_base64}",
                                "recognized_text": soru
                            })
                            logger.info("Cevap başarıyla gönderildi")
                            
                            # Geçici dosyaları temizle
                            os.unlink(temp_voice.name)
                            os.unlink(temp_webm_path)
                            os.unlink(temp_wav_path)

                    except sr.UnknownValueError:
                        error_msg = "Üzgünüm, söylediklerinizi anlayamadım. Lütfen tekrar deneyin."
                        logger.error(error_msg)
                        await websocket.send_json({
                            "text": error_msg,
                            "error": True
                        })
                    except sr.RequestError as e:
                        error_msg = f"Ses tanıma servisi hatası: {str(e)}"
                        logger.error(error_msg)
                        await websocket.send_json({
                            "text": error_msg,
                            "error": True
                        })
                    
                except Exception as e:
                    logger.error(f"Ses işleme hatası: {str(e)}")
                    await websocket.send_text(f"Hata: {str(e)}")
                
            except Exception as e:
                logger.error(f"WebSocket veri alım hatası: {str(e)}")
                break
    except Exception as e:
        logger.error(f"WebSocket bağlantı hatası: {str(e)}")
        raise

async def process_query(query: str):
    query = query.lower()
    logger.info(f"İşlenen soru: {query}")
    
    try:
        # Veritabanı bilgilerini metin haline getir
        context = ""
        for ay in DATABASE:
            for gun in DATABASE[ay]:
                for kayit in DATABASE[ay][gun]:
                    context += f"{ay} ayı {gun}. günü {kayit['marka']} markası {kayit['uretim_yeri']} üretim yerinde {kayit['yukleme_adedi']} adet yükleme yapmıştır. "
        
        logger.info(f"Oluşturulan context: {context}")
        
        # Prompt oluştur
        prompt = f"""<s>[INST] <<SYS>> You are a helpful AI assistant that speaks Turkish. Your name is {BOT_NAME} and you're talking to {USER_NAME}.
You can engage in general conversation and answer questions about the database information.
Always respond in Turkish and be friendly. <</SYS>>

Here is the database information:
{context}

User's question: {query}

If the question is about the database (production, loading, brands), use only the database information to answer.
If it's general chat or greeting, respond politely and helpfully.
Always respond in Turkish and be friendly. [/INST]"""
        
        logger.info(f"Oluşturulan prompt: {prompt}")
        
        # LLM ile yanıt üret
        result = llm_pipeline(
            prompt,
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )[0]
        
        # Yanıtı ayıkla
        answer = result['generated_text'].split("[/INST]")[-1].strip()
        logger.info(f"LLM yanıtı: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"İşleme hatası: {str(e)}")
        return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."

# Root endpoint
@app.get("/")
async def root():
    return {"message": "OtomolAI Backend API"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="debug"
    ) 