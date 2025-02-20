from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import json
import random
from datetime import datetime
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
from llm_utils import LLMProcessor

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

# LLM işlemcisini başlat
llm_processor = LLMProcessor()

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

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm originlere izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

                    except Exception as e:
                        error_msg = "Üzgünüm, söylediklerinizi anlayamadım. Lütfen tekrar deneyin."
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
        basic_greetings = ["merhaba", "selam", "gunaydin", "iyi gunler", "iyi aksamlar", "nasilsin", "naber"]
        if any(greeting in query for greeting in basic_greetings):
            return "Merhaba, size otomotiv satış verileri konusunda yardımcı olabilirim."
        
        if not DATABASE or 'Sheet1' not in DATABASE:
            return "Üzgünüm, veritabanında hiç veri bulunamadı."
        
        # TODO: RAG sistemi implementasyonu burada yapılacak
        context = ' '.join(DATABASE['Sheet1'])
        
        return await llm_processor.generate_response(query, context)
        
    except Exception as e:
        logger.error(f"Soru işleme hatası: {str(e)}", exc_info=True)
        return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."

# Root endpoint
@app.get("/")
async def root():
    return {"message": "OtomolAI Backend API"}

@app.get("/system-info")
async def get_system_info():
    """
    Sistem bilgilerini döndürür.
    """
    try:
        gpu_info = llm_processor.get_gpu_info()
        return {
            "status": "success",
            "gpu": gpu_info,
            "model": {
                "name": llm_processor.GENERATION_MODEL_NAME,
                "device": llm_processor.device,
                "loaded": llm_processor.model is not None
            }
        }
    except Exception as e:
        logger.error(f"Sistem bilgileri alınırken hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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