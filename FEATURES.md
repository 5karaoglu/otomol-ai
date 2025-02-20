# OtomolAI Özellikleri ve Fonksiyonları

## 🎯 Genel Özellikler

### Ses Tanıma ve Yanıt Verme
- Kullanıcının sesli komutlarını algılama
- Türkçe ses tanıma desteği
- Sesli yanıt üretme
- Doğal dil işleme ile akıllı yanıtlar

### Veri Analizi
- Otomotiv satış verilerini analiz etme
- Şube bazlı performans analizi
- Marka bazlı satış analizi
- Tarih bazlı trend analizi

### Dil İşleme
- Türkçe-İngilizce çeviri desteği
- Doğal dil anlama
- Bağlama duyarlı yanıtlar
- Akıllı veri filtreleme

## 🔧 Teknik Özellikler

### Veritabanı İşlemleri
- JSON formatında veri depolama
- Dinamik veri güncelleme
- Veri doğrulama ve temizleme
- Güvenli veri erişimi

### API Entegrasyonları
- FastAPI tabanlı backend
- WebSocket gerçek zamanlı iletişim
- CORS desteği
- Güvenli dosya yükleme

## 📚 Fonksiyon Açıklamaları

### LLM İşleme Modülü (`backend/llm_utils.py`)

#### LLMProcessor Sınıfı
```python
class LLMProcessor
```
- **Açıklama**: LLM işlemlerini yöneten ana sınıf
- **Özellikler**:
  - Model yükleme ve yönetimi
  - Prompt oluşturma
  - Yanıt üretme
  - Çeviri işlemleri

##### Model Yönetimi
```python
def __init__(self)
```
- **Açıklama**: LLM modelini ve gerekli bileşenleri yükler
- **Kullanım**: Sınıf başlatıldığında otomatik çalışır

##### Çeviri İşlemleri
```python
def translate_to_english(self, text: str) -> str
```
- **Açıklama**: Türkçe metni İngilizce'ye çevirir
- **Kullanım**: Kullanıcı sorgularını model için çevirme

##### Prompt Oluşturma
```python
def format_prompt(self, query: str, context: str = "") -> str
```
- **Açıklama**: LLaMA-2-chat formatında prompt oluşturur
- **Kullanım**: Model girdisini yapılandırma

##### Yanıt Üretme
```python
async def generate_response(self, query: str, context: str = "") -> str
```
- **Açıklama**: Verilen sorgu için LLM yanıtı üretir
- **Kullanım**: Kullanıcı sorgularını yanıtlama

### Ses İşleme Modülü (`backend/audio_utils.py`)

#### AudioProcessor Sınıfı
```python
class AudioProcessor
```
- **Açıklama**: Ses işleme işlemlerini yöneten ana sınıf
- **Özellikler**:
  - WebM'den WAV'a dönüşüm
  - Ses tanıma
  - Metin-ses dönüşümü
  - Geçici dosya yönetimi

##### WebM'den WAV'a Dönüşüm
```python
@staticmethod
convert_webm_to_wav(webm_path: str) -> str
```
- **Açıklama**: WebM formatındaki ses dosyasını WAV formatına dönüştürür
- **Kullanım**: Ses tanıma için format dönüşümü

##### Ses Tanıma
```python
@staticmethod
recognize_speech(audio_data: bytes) -> tuple[str, list[str]]
```
- **Açıklama**: Ses verisini metne dönüştürür
- **Kullanım**: Kullanıcı ses girişini metne çevirme

##### Metin-Ses Dönüşümü
```python
@staticmethod
text_to_speech(text: str) -> tuple[str, str]
```
- **Açıklama**: Metni sese dönüştürür
- **Kullanım**: Sistem yanıtlarını sesli yanıta çevirme

##### Dosya Temizleme
```python
@staticmethod
cleanup_files(file_paths: list[str]) -> None
```
- **Açıklama**: Geçici dosyaları temizler
- **Kullanım**: İşlem sonrası geçici dosyaları silme

### Backend Fonksiyonları (`backend/main.py`)

#### Token İşleme
```python
count_tokens(text: str) -> int
```
- **Açıklama**: Verilen metnin token sayısını hesaplar
- **Kullanım**: Model girdi limitlerini kontrol etme

#### Prompt Oluşturma
```python
format_prompt(query: str, context: str) -> str
```
- **Açıklama**: LLaMA-2-chat formatında prompt oluşturur
- **Kullanım**: Model girdisini yapılandırma

#### Sorgu İşleme
```python
process_query(query: str) -> str
```
- **Açıklama**: Kullanıcı sorgusunu işler ve yanıt üretir
- **Kullanım**: Ana sorgu işleme fonksiyonu

### WebSocket Endpoint'leri

#### `/ws` Endpoint
- **Açıklama**: Gerçek zamanlı ses iletişimi sağlar
- **Özellikler**:
  - Ses verisi alma
  - Ses tanıma
  - Yanıt üretme
  - Sesli yanıt gönderme

#### `/upload-database` Endpoint
- **Açıklama**: Veritabanı güncelleme endpoint'i
- **Özellikler**:
  - JSON dosya yükleme
  - Veri doğrulama
  - Veritabanı güncelleme

## 🔄 Güncellemeler

### v2.3 (Güncel Sürüm)
- LLM işlemleri ayrı bir modüle taşındı (`llm_utils.py`)
- Kod modülerliği artırıldı
- LLM işlemleri için sınıf yapısı oluşturuldu
- Bellek yönetimi iyileştirildi

### v2.2
- Chunk işleme fonksiyonları kaldırıldı
- Modern RAG sistemi için hazırlık yapıldı
- Planlanan RAG özellikleri:
  - Akıllı metin bölümleme (RecursiveCharacterTextSplitter)
  - Vektör veritabanı entegrasyonu (Chroma/FAISS)
  - Gelişmiş embedding modeli
  - Hibrit arama sistemi

### v2.1
- BERT modeli ve ilgili fonksiyonlar kaldırıldı
- RAG sistemi için hazırlık yapıldı
- Kod temizliği ve optimizasyonu yapıldı

### v2.0
- Ses işleme modülü oluşturuldu (`audio_utils.py`)
- Kod modüler yapıya geçirilmeye başlandı
- Türkçe fonksiyon açıklamaları eklendi
- FEATURES.md dosyası oluşturuldu
- Cursor yapılandırması eklendi

### v1.0
- Temel ses tanıma özellikleri
- FastAPI backend
- WebSocket desteği
- Veritabanı entegrasyonu 