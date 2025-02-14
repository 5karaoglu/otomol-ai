# Türkçe Sesli RAG Sistemi

Bu proje, Türkçe ses tanıma ve yanıt verme özelliklerine sahip bir RAG (Retrieval-Augmented Generation) sistemidir.

## Özellikler

- FastAPI backend
- React frontend
- Türkçe ses tanıma ve yanıt verme
- GPU destekli LLM (dbmdz/turkish-macro-bert-base)
- WebSocket bağlantısı
- Veritabanı dosyası yükleme özelliği

## Kurulum

### Backend

1. Python bağımlılıklarını yükleyin:
```bash
cd backend
pip install -r requirements.txt
```

2. Backend'i başlatın:
```bash
python main.py
```

### Frontend

1. Node.js bağımlılıklarını yükleyin:
```bash
cd frontend
npm install
```

2. Frontend'i başlatın:
```bash
npm start
```

## Kullanım

1. Frontend uygulamasını tarayıcınızda açın (http://localhost:3000)
2. JSON formatındaki veritabanı dosyanızı yükleyin
3. Mikrofon butonuna tıklayarak konuşmaya başlayın
4. Konuşmayı durdurmak için tekrar butona tıklayın
5. Sistem sorunuzu analiz edip sesli yanıt verecektir

## Veritabanı Formatı

```json
{
  "ocak_2025": {
    "1": [
      {
        "marka": "zara man",
        "yukleme_adedi": 28500,
        "uretim_yeri": "çalık"
      },
      {
        "marka": "zara woman",
        "yukleme_adedi": 15700,
        "uretim_yeri": "smart"
      }
    ]
  }
}
```

## Desteklenen Markalar

- zara man
- zara woman
- bershka
- urban

## Üretim Yerleri

- çalık
- smart
- alex
- afrika

## Gereksinimler

- Python 3.8+
- Node.js 14+
- CUDA destekli GPU
- Mikrofon 