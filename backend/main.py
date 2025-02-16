from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import json
import random
from datetime import datetime
import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
import speech_recognition as sr # type: ignore
from gtts import gTTS # type: ignore
import os
import base64
import tempfile
import logging
import subprocess
import argparse
import uvicorn # type: ignore

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM model ve tokenizer yükleme
TURKISH_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
GENERATION_MODEL_NAME = "ai-forever/mGPT"  # Çok dilli GPT modeli

# GPU bellek optimizasyonları
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Türkçe BERT model ve tokenizer yükleme
try:
    logger.info("Türkçe BERT model ve tokenizer yükleniyor...")
    turkish_tokenizer = AutoTokenizer.from_pretrained(TURKISH_MODEL_NAME)
    turkish_model = AutoModelForCausalLM.from_pretrained(
        TURKISH_MODEL_NAME,
        torch_dtype=torch_dtype
    ).to(device)
    logger.info("Türkçe BERT model ve tokenizer başarıyla yüklendi")
except Exception as e:
    logger.error(f"Türkçe model yükleme hatası: {str(e)}")
    turkish_model = None
    turkish_tokenizer = None

# Üretim modeli yükleme
try:
    logger.info("Üretim modeli yükleniyor...")
    generation_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    generation_model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch_dtype,
        load_in_8bit=True
    )
    logger.info("Üretim modeli başarıyla yüklendi")
except Exception as e:
    logger.error(f"Üretim modeli yükleme hatası: {str(e)}")
    generation_model = None
    generation_tokenizer = None

# Text generation pipeline oluştur
generation_pipeline = pipeline(
    "text-generation",
    model=generation_model,
    tokenizer=generation_tokenizer,
    device_map="auto",
    torch_dtype=torch_dtype,
    max_length=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.2
) if generation_model else None

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
    print("Kullanılabilir GPU belleği:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    print("Kullanılan GPU belleği:", torch.cuda.memory_allocated(0) / 1024**3, "GB")

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

async def process_query(query: str) -> str:
    query = query.lower()
    logger.info(f"İşlenen soru: {query}")
    
    try:
        # 1. Bağlam Oluşturma
        context = generate_context()
        logger.info(f"Oluşturulan context: {context}")
        
        if not turkish_model or not generation_model:
            return "Üzgünüm, modeller yüklenemediği için şu anda hizmet veremiyorum."
        
        # 2. BERT ile anlama
        bert_prompt = f"""Soru: {query}
Bağlam: {context}"""
        
        # BERT tokenization ve çıktı
        bert_inputs = turkish_tokenizer(
            bert_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(turkish_model.device)
        
        bert_outputs = turkish_model(**bert_inputs)
        bert_embeddings = bert_outputs.last_hidden_state
        
        # 3. GPT ile yanıt üretme
        gpt_prompt = f"""Sen OtomolAI adında bir yapay zeka asistanısın. 
Konuştuğun kişinin adı Osman Bey. 
Aşağıdaki soruyu bağlam bilgilerini kullanarak detaylı bir şekilde yanıtla.

Bağlam Bilgisi:
{context}

Soru: {query}

Yanıt:"""
        
        gpt_inputs = generation_tokenizer(
            gpt_prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to(generation_model.device)
        
        # GPT çıktısı üret
        outputs = generation_model.generate(
            gpt_inputs.input_ids,
            max_length=4096,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.2,
            num_return_sequences=1,
            min_length=50,
            pad_token_id=generation_tokenizer.eos_token_id
        )
        
        # Yanıtı ayıkla
        response = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Yanıt:")[-1].strip()
        
        # Türkçe son işlemler
        answer_tokens = turkish_tokenizer.tokenize(answer)
        answer = turkish_tokenizer.convert_tokens_to_string(answer_tokens)
        
        # Yanıt kontrolü
        if not answer or len(answer) < 10:
            answer = "Üzgünüm, sorunuzu tam olarak anlayamadım. Lütfen sorunuzu daha açık bir şekilde sorar mısınız?"
        
        logger.info(f"Final yanıt: {answer}")
        return answer
        
    except Exception as e:
        logger.error(f"Soru işleme hatası: {str(e)}", exc_info=True)
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