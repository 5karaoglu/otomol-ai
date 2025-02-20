from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import json
import random
from datetime import datetime
import torch # type: ignore
from transformers import LlamaForCausalLM, LlamaTokenizer # type: ignore
from googletrans import Translator # type: ignore
import os
import base64
import tempfile
import logging
import subprocess
import argparse
import uvicorn # type: ignore
import shutil
from pathlib import Path
from typing import List, Dict
from audio_utils import AudioProcessor

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('otomol_ai.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Model isimleri
GENERATION_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")  # Token'ı environment variable'dan al

# Sistem promptu
SYSTEM_PROMPT = """Sen OtomolAI adında, otomotiv satış verileri konusunda uzmanlaşmış bir veri analistisin.

VERİTABANI KULLANIMI:
- Sadece veritabanındaki bilgileri kullanarak yanıt ver
- Veritabanı dışındaki konularda "Üzgünüm, bu konu hakkında veritabanımda bilgi bulunmuyor." yanıtını ver
- Tahmin yürütme, veritabanında olmayan bilgileri kullanma
- Her zaman sayısal verileri Türk formatında sun (örn: 1.234.567,89)
- Yanıtlarını kısa ve öz tut

YANITLAMA KURALLARI:
- Sadece sorulan veriyi yanıtla
- Gereksiz açıklamalar yapma
- Eğer veri bulunamazsa "Üzgünüm, bu konuyla ilgili veritabanında bilgi bulamadım." de
- Selamlaşma veya sohbet girişimlerinde "Merhaba, size otomotiv satış verileri konusunda yardımcı olabilirim." yanıtını ver"""

# Çevirmen başlat
translator = Translator()

def translate_to_english(text: str) -> str:
    """
    Türkçe metni İngilizce'ye çevirir.
    
    Args:
        text (str): Çevrilecek Türkçe metin
    
    Returns:
        str: Çevrilmiş İngilizce metin
    
    Raises:
        Exception: Çeviri sırasında bir hata oluşursa
    """
    try:
        translation = translator.translate(text, src='tr', dest='en')
        return translation.text
    except Exception as e:
        logger.error(f"Çeviri hatası: {str(e)}")
        return text

def count_tokens(text: str) -> int:
    """
    Verilen metnin token sayısını hesaplar.
    
    Args:
        text (str): Token sayısı hesaplanacak metin
    
    Returns:
        int: Toplam token sayısı
    """
    return len(llama_tokenizer.encode(text))

def format_prompt(query: str, context: str) -> str:
    """
    LLaMA-2-chat formatında prompt oluşturur.
    
    Args:
        query (str): Kullanıcı sorgusu
        context (str): İlgili bağlam bilgisi
    
    Returns:
        str: Formatlanmış prompt
    """
    basic_greetings = ["merhaba", "selam", "gunaydin", "iyi gunler", "iyi aksamlar", "nasilsin", "naber"]
    if any(greeting in query.lower() for greeting in basic_greetings):
        return f"""<s>[SYSTEM]
{SYSTEM_PROMPT}
[/SYSTEM]

[USER]
{query}
[/USER]

[ASSISTANT]
Merhaba, size otomotiv satış verileri konusunda yardımcı olabilirim.
"""
    
    return f"""<s>[SYSTEM]
{SYSTEM_PROMPT}

ÖNEMLİ: Sadece aşağıdaki bağlam bilgisini kullanarak yanıt ver. 
Bağlam dışındaki bilgileri ASLA kullanma.
[/SYSTEM]

[USER]
<CONTEXT>
{context}
</CONTEXT>

{query}
[/USER]

[ASSISTANT]
"""

# GPU bellek optimizasyonları
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32

# Türkçe BERT model ve tokenizer yükleme
try:
    logger.info("Türkçe BERT model yükleniyor...")
    turkish_tokenizer = AutoTokenizer.from_pretrained(TURKISH_MODEL_NAME)
    turkish_model = AutoModel.from_pretrained(
        TURKISH_MODEL_NAME,
        torch_dtype=torch_dtype
    ).to(device)
    logger.info("Türkçe BERT model ve tokenizer başarıyla yüklendi")
except Exception as e:
    logger.error(f"Türkçe model yükleme hatası: {str(e)}")
    turkish_model = None
    turkish_tokenizer = None

# LLaMA modeli yükleme
try:
    logger.info("LLaMA modeli yükleniyor...")
    llama_tokenizer = LlamaTokenizer.from_pretrained(
        GENERATION_MODEL_NAME,
        token=HF_TOKEN
    )
    llama_model = LlamaForCausalLM.from_pretrained(
        GENERATION_MODEL_NAME,
    torch_dtype=torch_dtype,
        token=HF_TOKEN,
        device_map="auto",
        load_in_8bit=True
    )
    logger.info("LLaMA modeli başarıyla yüklendi")
except Exception as e:
    logger.error(f"LLaMA model yükleme hatası: {str(e)}")
    llama_model = None
    llama_tokenizer = None

def calculate_similarity(query_embedding: torch.Tensor, context_embedding: torch.Tensor) -> float:
    """İki embedding arasındaki benzerliği hesapla"""
    similarity = torch.nn.functional.cosine_similarity(
        query_embedding.mean(dim=1),
        context_embedding.mean(dim=1)
    )
    return similarity.item()

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
MARKALAR = ["BMW", "MERCEDES BENZ", "AUDI", "VOLVO", "VOLKSWAGEN", "TESLA", "SEAT", "SKODA"]
SUBELER = ["MERTER", "MASLAK", "ATAŞEHİR", "İZMİR", "BODRUM", "ÇANKAYA", "ÇAYYOLU", "ANTALYA"]

# Veritabanı
DATABASE = {}

# Başlangıçta veritabanını yükle
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "ocak_data.json")
    with open(db_path, "r", encoding='utf-8') as f:
        DATABASE = json.load(f)
    logger.info("Veritabanı başarıyla yüklendi")
    logger.info(f"Yüklenen kayıt sayısı: {len(DATABASE.get('Sheet1', []))}")
except Exception as e:
    logger.error(f"Veritabanı yükleme hatası: {str(e)}")

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
                        audio_base64, temp_file = AudioProcessor.text_to_speech(welcome_message)
                        
                        # Cevabı gönder
                        await websocket.send_json({
                            "text": welcome_message,
                            "audio": audio_base64
                        })
                        logger.info("Hoşgeldin mesajı gönderildi")
                        
                        # Geçici dosyayı temizle
                        AudioProcessor.cleanup_files([temp_file])
                        continue

                except json.JSONDecodeError as e:
                    logger.error(f"JSON ayrıştırma hatası: {str(e)}")
                    pass

                try:
                    # Base64'ten ses verisini çöz
                    audio_data = base64.b64decode(data.split(",")[1])
                    logger.info("Ses verisi başarıyla decode edildi")
                    
                    try:
                        # Ses tanıma işlemi
                        soru, temp_files = AudioProcessor.recognize_speech(audio_data)
                        
                        # Soruyu analiz et ve cevap oluştur
                        cevap = await process_query(soru)
                        logger.info(f"Oluşturulan cevap: {cevap}")
                        
                        # Metni sese çevir
                        audio_base64, temp_voice = AudioProcessor.text_to_speech(cevap)
                        temp_files.append(temp_voice)
                        
                        # Cevabı gönder
                        await websocket.send_json({
                            "text": cevap,
                            "audio": audio_base64,
                            "recognized_text": soru
                        })
                        logger.info("Cevap başarıyla gönderildi")
                        
                        # Geçici dosyaları temizle
                        AudioProcessor.cleanup_files(temp_files)

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
    """
    Kullanıcı sorgusunu işler ve yanıt üretir.
    
    Args:
        query (str): Kullanıcı sorgusu
    
    Returns:
        str: Üretilen yanıt
        
    Raises:
        Exception: Model yüklenemediğinde veya işlem sırasında hata oluştuğunda
    """
    query = query.lower().replace('i̇', 'i')
    logger.info(f"İşlenen soru: {query}")
    
    try:
        if not llama_model:
            return "Üzgünüm, model yüklenemediği için şu anda hizmet veremiyorum."
        
        basic_greetings = ["merhaba", "selam", "gunaydin", "iyi gunler", "iyi aksamlar", "nasilsin", "naber"]
        if any(greeting in query for greeting in basic_greetings):
            return "Merhaba, size otomotiv satış verileri konusunda yardımcı olabilirim."
        
        english_query = translate_to_english(query)
        logger.info(f"İngilizce'ye çevrilmiş soru: {english_query}")
        
        if not DATABASE or 'Sheet1' not in DATABASE:
            return "Üzgünüm, veritabanında hiç veri bulunamadı."
        
        # TODO: RAG sistemi implementasyonu burada yapılacak
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a professional automotive sales data analyst. Here is the relevant data for the query:

CONTEXT:
{' '.join(DATABASE['Sheet1'])}

Please follow these rules in your response:
1. ONLY use the data provided in the context above
2. If the answer cannot be found in the context, respond with "No information found in the database for this query."
3. Keep responses focused only on the data
4. Format numbers with commas for thousands (example: 1,234)
5. Respond in English
6. Be brief and precise
7. Do not add any explanations or pleasantries
8. For sales questions, include both the number of vehicles and revenue
9. For revenue questions, always include the TL symbol
10. For comparison questions, show the data in a clear format
11. For brand questions, list all brands with their numbers
12. For branch questions, include all relevant data from that branch"""
            },
            {
                "role": "user",
                "content": english_query
            }
        ]
        
        prompt = llama_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = llama_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        ).to(device)
        
        outputs = llama_model.generate(
            inputs.input_ids,
            max_length=2048,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.3,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
        
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in response:
            answer = response.split("<|assistant|>")[-1].strip()
        else:
            answer = response.strip()
        
        answer = " ".join(answer.split())
        
        logger.info(f"LLM'den gelen İngilizce yanıt: {answer}")
        
        try:
            turkish_answer = translator.translate(answer, src='en', dest='tr').text
            logger.info(f"Türkçe'ye çevrilmiş yanıt: {turkish_answer}")
            
            import re
            def format_number(match):
                number = match.group(0)
                try:
                    return "{:,.0f}".format(float(number)).replace(",", ".")
                except:
                    return number
            
            turkish_answer = re.sub(r'\d+(?:,\d+)?', format_number, turkish_answer)
            
            return turkish_answer
            
        except Exception as e:
            logger.error(f"Çeviri hatası: {str(e)}")
            return "Üzgünüm, yanıt üretilirken bir hata oluştu."
        
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