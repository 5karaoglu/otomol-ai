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

### Backend Fonksiyonları (`backend/main.py`)

#### Çeviri Fonksiyonları
```python
translate_to_english(text: str) -> str
```
- **Açıklama**: Türkçe metni İngilizce'ye çevirir
- **Kullanım**: Kullanıcı sorgularını model için İngilizce'ye çevirme

#### Token İşleme
```python
count_tokens(text: str) -> int
```
- **Açıklama**: Verilen metnin token sayısını hesaplar
- **Kullanım**: Model girdi limitlerini kontrol etme

#### Metin Parçalama
```python
split_into_chunks(text: str, chunk_size: int = 200) -> List[str]
```
- **Açıklama**: Metni anlamlı parçalara böler
- **Kullanım**: Uzun metinleri işlenebilir parçalara ayırma

#### Embedding İşlemleri
```python
create_embeddings(texts: List[str], model, tokenizer) -> torch.Tensor
```
- **Açıklama**: Metinlerin vektör temsillerini oluşturur
- **Kullanım**: Semantik arama ve benzerlik hesaplama

#### İlgili İçerik Bulma
```python
retrieve_relevant_chunks(query: str, chunks: List[str], top_k: int = 3) -> List[str]
```
- **Açıklama**: Soruya en alakalı bağlam parçalarını bulur
- **Kullanım**: Veritabanından ilgili bilgileri çıkarma

#### Prompt Oluşturma
```python
format_prompt(query: str, context: str, bert_similarity: float) -> str
```
- **Açıklama**: LLaMA-2-chat formatında prompt oluşturur
- **Kullanım**: Model girdisini yapılandırma

#### Veri İşleme
```python
create_data_chunks() -> List[Dict]
```
- **Açıklama**: Veritabanı kayıtlarını yapılandırılmış parçalara böler
- **Kullanım**: Veritabanı verilerini işlenebilir formata dönüştürme

#### İlgili Veri Bulma
```python
find_relevant_chunks(query: str, chunks: List[Dict], top_k: int = 5) -> List[str]
```
- **Açıklama**: Sorguyla en alakalı veri parçalarını bulur
- **Kullanım**: Kullanıcı sorgularına uygun verileri filtreleme

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

### v2 (Güncel Sürüm)
- Türkçe fonksiyon açıklamaları eklendi
- FEATURES.md dosyası oluşturuldu
- Cursor yapılandırması eklendi

### v1
- Temel ses tanıma özellikleri
- FastAPI backend
- WebSocket desteği
- Veritabanı entegrasyonu 