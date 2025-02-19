from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import json
import random
from datetime import datetime
import torch # type: ignore
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer, pipeline # type: ignore
import speech_recognition as sr # type: ignore
from gtts import gTTS # type: ignore
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

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model isimleri
TURKISH_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
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
    """Türkçe metni İngilizce'ye çevir"""
    try:
        translation = translator.translate(text, src='tr', dest='en')
        return translation.text
    except Exception as e:
        logger.error(f"Çeviri hatası: {str(e)}")
        return text  # Hata durumunda orijinal metni döndür

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
            # CLS token'ınının embedding'ini al
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
Merhaba, size otomotiv satış verileri konusunda yardımcı olabilirim.
"""
    
    # Veritabanı ile ilgili soru ise bağlamı ekle
    if bert_similarity > 0.3:
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
    
    # Veritabanı dışı soru
    return f"""<s>[SYSTEM]
{SYSTEM_PROMPT}
[/SYSTEM]

[USER]
{query}
[/USER]

[ASSISTANT]
Üzgünüm, bu konu hakkında veritabanımda bilgi bulunmuyor.
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
    """Split database records into structured chunks"""
    chunks = []
    
    if not DATABASE or 'Sheet1' not in DATABASE:
        logger.error("Database is empty or has invalid format")
        return chunks
    
    for kayit in DATABASE['Sheet1']:
        # Store each record as a structured dictionary
        chunk = {
            'text': f"In {kayit['Ay']}, {kayit['Araç Çıkış Adedi']} vehicles of {kayit['Marka']} brand were delivered from {kayit['Şube']} branch and generated a revenue of {kayit['Ciro']} TL.",
            'metadata': {
                'branch': kayit['Şube'].lower(),
                'brand': kayit['Marka'].lower(),
                'month': kayit['Ay'].lower(),
                'year': kayit['Yıl'],
                'vehicle_count': kayit['Araç Çıkış Adedi'],
                'revenue': kayit['Ciro'],
                'date': kayit['Tarih']
            }
        }
        chunks.append(chunk)
    
    logger.info(f"Number of chunks created: {len(chunks)}")
    return chunks

def find_relevant_chunks(query: str, chunks: List[Dict], top_k: int = 3) -> List[str]:
    """Find chunks most relevant to the query"""
    # Handle Turkish characters properly
    query = query.lower().replace('i̇', 'i')
    
    # Stopwords - Turkish conjunctions and unnecessary words
    stopwords = {'ve', 'veya', 'ile', 'de', 'da', 'ki', 'bu', 'şu', 'bir', 'için', 'gibi', 'kadar', 'sonra', 'önce', 'kaç', 'ne', 'nerede', 'nasıl'}
    
    # Clean and analyze query
    query_words = set(word.strip('.,?!') for word in query.split() if word.strip('.,?!') not in stopwords)
    
    # Analyze branch and brand information
    branch_match = None
    brand_match = None
    
    for word in query_words:
        # Branch matching
        for branch in SUBELER:
            if word in branch.lower():
                branch_match = branch.lower()
                break
        # Brand matching
        for brand in MARKALAR:
            if word in brand.lower():
                brand_match = brand.lower()
                break
    
    # Score chunks
    chunk_scores = []
    for chunk in chunks:
        metadata = chunk['metadata']
        score = 0
        
        # Branch matching
        if branch_match and branch_match in metadata['branch']:
            score += 5
        
        # Brand matching
        if brand_match and brand_match in metadata['brand']:
            score += 5
        
        # Word-based similarity
        chunk_words = set(word.strip('.,?!') for word in chunk['text'].lower().split() if word.strip('.,?!') not in stopwords)
        common_words = query_words & chunk_words
        score += len(common_words)
        
        # Revenue or vehicle count query
        if any(word in query_words for word in ['ciro', 'kazanç', 'para', 'gelir', 'tl', 'revenue', 'income', 'money']):
            score += 3
        if any(word in query_words for word in ['araç', 'arac', 'satış', 'satis', 'adet', 'vehicle', 'car', 'sales', 'count']):
            score += 3
        
        if score > 0:
            chunk_scores.append((score, chunk['text']))
    
    # Select chunks with highest scores
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
            return "Merhaba, size otomotiv satış verileri konusunda yardımcı olabilirim."
        
        # Soruyu İngilizce'ye çevir
        english_query = translate_to_english(query)
        logger.info(f"İngilizce'ye çevrilmiş soru: {english_query}")
        
        # Veritabanı chunk'larını oluştur
        chunks = create_data_chunks()
        
        if not chunks:
            return "Üzgünüm, veritabanında hiç veri bulunamadı."
        
        # İngilizce soru ile alakalı chunk'ları bul
        relevant_chunks = find_relevant_chunks(english_query, chunks)
        logger.info(f"Bulunan alakalı chunk sayısı: {len(relevant_chunks)}")
        
        # Alakalı chunk'ları detaylı logla
        logger.info("Bulunan alakalı chunk'lar:")
        for i, chunk in enumerate(relevant_chunks, 1):
            logger.info(f"Chunk {i}:")
            logger.info(f"İçerik: {chunk}")
            logger.info("-" * 50)
        
        if not relevant_chunks:
            return "Üzgünüm, sorunuzla ilgili veri bulamadım. Lütfen başka bir şekilde sorar mısınız?"
        
        # Chat mesajlarını oluştur
        messages = [
            {
                "role": "system",
                "content": f"""You are a professional automotive sales data analyst. Here is the relevant data for the query:

CONTEXT:
{' '.join(relevant_chunks)}

Please follow these rules in your response:
1. ONLY use the data provided in the context above
2. If the answer cannot be found in the context, respond with "No information found in the database for this query."
3. Keep responses focused only on the data
4. Format numbers with commas for thousands (example: 1,234)
5. Respond in English
6. Be brief and precise
7. Do not add any explanations or pleasantries"""
            },
            {
                "role": "user",
                "content": english_query
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
        
        logger.info(f"LLM'den gelen İngilizce yanıt: {answer}")
        
        # İngilizce yanıtı Türkçe'ye çevir
        try:
            turkish_answer = translator.translate(answer, src='en', dest='tr').text
            logger.info(f"Türkçe'ye çevrilmiş yanıt: {turkish_answer}")
            
            # Sayı formatını Türk formatına çevir
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