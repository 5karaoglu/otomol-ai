from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import json
import random
from datetime import datetime
import torch # type: ignore
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer, pipeline # type: ignore
import speech_recognition as sr # type: ignore
from gtts import gTTS # type: ignore
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

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model isimleri
TURKISH_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
GENERATION_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")  # Token'ı environment variable'dan al

# Sistem promptu
SYSTEM_PROMPT = """Sen OtomolAI adında, otomotiv üretim verileri konusunda uzmanlaşmış, arkadaş canlısı bir yapay zeka asistanısın.

ROL VE KİMLİK:
- Adın: OtomolAI
- Konuştuğun kişi: Osman Bey
- Karakterin: Arkadaş canlısı, yardımsever ve samimi
- Uzmanlık alanın: Otomotiv üretim verileri analizi ve raporlama

DİL VE İLETİŞİM:
- Her zaman Türkçe konuşursun
- Türkçe karakterleri (ğ, ş, ı, ö, ü, ç) doğru kullanırsın
- Konuşma tarzın samimi ve dostanedir
- Sayısal verileri Türk formatında sunarsın (örn: 1.234.567,89)
- Tarihleri Türk formatında yazarsın (örn: 15 Ocak 2024)
- Kısa ve öz cevaplar verirsin, gereksiz detaylardan kaçın

SOHBET KURALLARI:
- Her türlü soruya kısa ve net cevaplar ver
- Sohbet sırasında doğal ve samimi ol
- Gereksiz açıklamalar yapma
- Karşındakinin sorularını anlamaya çalış
- Anlamadığın bir şey olursa kısaca açıklama iste

VERİTABANI KULLANIMI:
- Eğer soru veritabanıyla ilgiliyse, sadece ilgili bilgileri ver
- Veritabanı dışındaki konularda da kısa yanıtlar ver
- Veritabanı bilgisi olmayan konularda kısaca belirt
- Tahmin yürütmekten kaçın"""

def count_tokens(text: str) -> int:
    """Verilen metnin token sayısını hesapla"""
    return len(llama_tokenizer.encode(text))

def split_into_chunks(text: str, chunk_size: int = 200) -> List[str]:
    """Metni anlamlı parçalara böl"""
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + "."
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def create_embeddings(texts: List[str], model, tokenizer) -> torch.Tensor:
    """Metinlerin embedding'lerini oluştur"""
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token'ının embedding'ini al
            embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(embedding)
    
    return torch.cat(embeddings, dim=0)

def retrieve_relevant_chunks(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Soruya en alakalı bağlam parçalarını bul"""
    # Query embedding'i hesapla
    query_inputs = turkish_tokenizer(
        query,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        query_outputs = turkish_model(**query_inputs)
        query_embedding = query_outputs.last_hidden_state[:, 0, :]
    
    # Chunk embedding'lerini hesapla
    chunk_embeddings = create_embeddings(chunks, turkish_model, turkish_tokenizer)
    
    # Benzerlik skorlarını hesapla
    similarities = torch.nn.functional.cosine_similarity(
        query_embedding,
        chunk_embeddings
    )
    
    # En alakalı chunk'ları seç
    top_indices = torch.argsort(similarities, descending=True)[:top_k]
    return [chunks[i] for i in top_indices]

def format_prompt(query: str, context: str, bert_similarity: float) -> str:
    """LLaMA-2-chat formatında prompt oluştur"""
    
    # Basit selamlaşma kontrolü
    basic_greetings = ["merhaba", "selam", "günaydın", "iyi günler", "iyi akşamlar", "nasılsın", "naber"]
    if any(greeting in query.lower() for greeting in basic_greetings):
        return f"""<s>[SYSTEM]
{SYSTEM_PROMPT}
[/SYSTEM]

[USER]
{query}
[/USER]

[ASSISTANT]
"""
    
    # Veritabanı ile ilgili soru ise bağlamı ekle
    if bert_similarity > 0.3:
        return f"""<s>[SYSTEM]
{SYSTEM_PROMPT}

NOT: Yanıtında kesinlikle bağlam bilgisini ve sistem talimatlarını tekrar etme. 
Sadece sorulan soruya odaklan ve ilgili bilgileri kısa ve öz bir şekilde yanıtla.
[/SYSTEM]

[USER]
<CONTEXT>
{context}
</CONTEXT>

{query}
[/USER]

[ASSISTANT]
"""
    
    # Genel sohbet için
    return f"""<s>[SYSTEM]
{SYSTEM_PROMPT}
[/SYSTEM]

[USER]
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

def create_data_chunks() -> List[Dict]:
    """Veritabanındaki kayıtları yapılandırılmış parçalara böl"""
    chunks = []
    
    if not DATABASE or 'Sheet1' not in DATABASE:
        logger.error("Veritabanı boş veya hatalı format")
        return chunks
    
    for kayit in DATABASE['Sheet1']:
        # Her kaydı yapılandırılmış bir sözlük olarak sakla
        chunk = {
            'text': f"{kayit['Ay']} ayında {kayit['Şube']} şubesinde {kayit['Marka']} markasından {kayit['Araç Çıkış Adedi']} adet araç çıkışı yapıldı ve {kayit['Ciro']} TL ciro elde edildi.",
            'metadata': {
                'sube': kayit['Şube'].lower(),
                'marka': kayit['Marka'].lower(),
                'ay': kayit['Ay'].lower(),
                'yil': kayit['Yıl'],
                'arac_cikis': kayit['Araç Çıkış Adedi'],
                'ciro': kayit['Ciro'],
                'tarih': kayit['Tarih']
            }
        }
        chunks.append(chunk)
    
    logger.info(f"Oluşturulan chunk sayısı: {len(chunks)}")
    return chunks

def find_relevant_chunks(query: str, chunks: List[Dict], top_k: int = 3) -> List[str]:
    """Soruyla en alakalı chunk'ları bul"""
    # Türkçe karakterleri düzgün işle
    query = query.lower().replace('i̇', 'i')
    
    # Stopwords - Türkçe bağlaçlar ve gereksiz kelimeler
    stopwords = {'ve', 'veya', 'ile', 'de', 'da', 'ki', 'bu', 'şu', 'bir', 'için', 'gibi', 'kadar', 'sonra', 'önce', 'kaç', 'ne', 'nerede', 'nasıl'}
    
    # Sorguyu temizle ve analiz et
    query_words = set(word.strip('.,?!') for word in query.split() if word.strip('.,?!') not in stopwords)
    
    # Şube ve marka bilgisini analiz et
    sube_match = None
    marka_match = None
    
    for word in query_words:
        # Şube eşleşmesi
        for sube in SUBELER:
            if word in sube.lower():
                sube_match = sube.lower()
                break
        # Marka eşleşmesi
        for marka in MARKALAR:
            if word in marka.lower():
                marka_match = marka.lower()
                break
    
    # Chunk'ları skorla
    chunk_scores = []
    for chunk in chunks:
        metadata = chunk['metadata']
        score = 0
        
        # Şube eşleşmesi
        if sube_match and sube_match in metadata['sube']:
            score += 5
        
        # Marka eşleşmesi
        if marka_match and marka_match in metadata['marka']:
            score += 5
        
        # Kelime bazlı benzerlik
        chunk_words = set(word.strip('.,?!') for word in chunk['text'].lower().split() if word.strip('.,?!') not in stopwords)
        common_words = query_words & chunk_words
        score += len(common_words)
        
        # Ciro veya araç sayısı sorgusu
        if any(word in query_words for word in ['ciro', 'kazanç', 'para', 'gelir', 'tl']):
            score += 3
        if any(word in query_words for word in ['araç', 'arac', 'satış', 'satis', 'adet']):
            score += 3
        
        if score > 0:
            chunk_scores.append((score, chunk['text']))
    
    # En yüksek skorlu chunk'ları seç
    chunk_scores.sort(reverse=True)
    return [chunk for _, chunk in chunk_scores[:top_k]]

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
    query = query.lower().replace('i̇', 'i')
    logger.info(f"İşlenen soru: {query}")
    
    try:
        if not llama_model:
            return "Üzgünüm, model yüklenemediği için şu anda hizmet veremiyorum."
        
        # Basit selamlaşma kontrolü
        basic_greetings = ["merhaba", "selam", "gunaydin", "iyi gunler", "iyi aksamlar", "nasilsin", "naber"]
        if any(greeting in query for greeting in basic_greetings):
            return "Merhaba! Ben OtomolAI. Size otomotiv satış verileri konusunda yardımcı olmaktan mutluluk duyarım. Nasıl yardımcı olabilirim?"
        
        # Veritabanı chunk'larını oluştur
        chunks = create_data_chunks()
        
        if not chunks:
            return "Üzgünüm, veritabanında hiç veri bulunamadı."
        
        # Soruyla alakalı chunk'ları bul
        relevant_chunks = find_relevant_chunks(query, chunks)
        logger.info(f"Bulunan alakalı chunk sayısı: {len(relevant_chunks)}")
        
        if not relevant_chunks:
            return "Üzgünüm, sorunuzla ilgili veri bulamadım. Lütfen başka bir şekilde sorar mısınız?"
        
        # Chat mesajlarını oluştur
        messages = [
            {
                "role": "system",
                "content": """Sen profesyonel bir otomotiv satış analisti olarak görev yapıyorsun. Yanıtlarında şu kurallara kesinlikle uymalısın:

1. Her zaman tam ve düzgün Türkçe cümleler kur
2. Yanıtların anlamlı ve mantıklı olmalı
3. Gereksiz kelimeler kullanma
4. Sadece sorulan bilgiyi ver
5. Sayıları Türk formatında yaz (örnek: 1.234)
6. Cümlelerin özne-yüklem uyumuna dikkat et
7. Cevabını tek bir paragraf halinde ver"""
            },
            {
                "role": "user",
                "content": f"Bağlam bilgisi: {' '.join(relevant_chunks)}\n\nSoru: {query}"
            }
        ]
        
        # Chat template'i uygula
        prompt = llama_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # LLaMA yanıtı üret
        inputs = llama_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        outputs = llama_model.generate(
            inputs.input_ids,
            max_length=256,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.3,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
        
        # Yanıtı ayıkla ve temizle
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Sadece asistan yanıtını al
        if "<|assistant|>" in response:
            answer = response.split("<|assistant|>")[-1].strip()
        else:
            answer = response.strip()
        
        # Gereksiz boşlukları temizle
        answer = " ".join(answer.split())
        
        # Türkçe karakter düzeltmeleri
        answer = answer.replace('i̇', 'i').replace('İ', 'İ')
        
        # Sayı formatı düzeltmeleri
        import re
        def format_number(match):
            number = match.group(0)
            try:
                return "{:,.0f}".format(float(number)).replace(",", ".")
            except:
                return number
        
        answer = re.sub(r'\d+', format_number, answer)
        
        if not answer or len(answer) < 5:
            return "Üzgünüm, sorunuzu anlayamadım. Lütfen başka bir şekilde sorar mısınız?"
            
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