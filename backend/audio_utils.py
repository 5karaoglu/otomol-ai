import speech_recognition as sr
from gtts import gTTS
import tempfile
import base64
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Ses işleme işlemlerini yöneten sınıf.
    """
    
    @staticmethod
    def convert_webm_to_wav(webm_path: str) -> str:
        """
        WebM formatındaki ses dosyasını WAV formatına dönüştürür.
        
        Args:
            webm_path (str): WebM dosyasının yolu
            
        Returns:
            str: Oluşturulan WAV dosyasının yolu
        """
        wav_path = webm_path.replace(".webm", ".wav")
        subprocess.run(['ffmpeg', '-i', webm_path, '-acodec', 'pcm_s16le', '-ar', '44100', wav_path])
        logger.info("Ses dosyası WAV formatına dönüştürüldü")
        return wav_path
    
    @staticmethod
    def recognize_speech(audio_data: bytes) -> tuple[str, list[str]]:
        """
        Ses verisini metne dönüştürür.
        
        Args:
            audio_data (bytes): İşlenecek ses verisi
            
        Returns:
            tuple[str, list[str]]: Tanınan metin ve oluşturulan geçici dosyaların yolları
        
        Raises:
            sr.UnknownValueError: Ses tanınamadığında
            sr.RequestError: Google servisi hatası durumunda
        """
        temp_files = []
        
        # WebM dosyası oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
            temp_webm.write(audio_data)
            temp_webm_path = temp_webm.name
            temp_files.append(temp_webm_path)
        
        # WAV'a dönüştür
        wav_path = AudioProcessor.convert_webm_to_wav(temp_webm_path)
        temp_files.append(wav_path)
        
        # Ses tanıma
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio, language="tr-TR")
            logger.info(f"Tanınan ses: {recognized_text}")
        
        return recognized_text, temp_files
    
    @staticmethod
    def text_to_speech(text: str) -> tuple[str, str]:
        """
        Metni sese dönüştürür.
        
        Args:
            text (str): Sese dönüştürülecek metin
            
        Returns:
            tuple[str, str]: Base64 formatında ses verisi ve geçici dosya yolu
        """
        # Metni sese çevir
        tts = gTTS(text=text, lang='tr')
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_voice:
            tts.save(temp_voice.name)
            with open(temp_voice.name, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode()
        
        return f"data:audio/mp3;base64,{audio_base64}", temp_voice.name
    
    @staticmethod
    def cleanup_files(file_paths: list[str]) -> None:
        """
        Geçici dosyaları temizler.
        
        Args:
            file_paths (list[str]): Silinecek dosya yolları listesi
        """
        for file_path in file_paths:
            try:
                os.unlink(file_path)
            except Exception as e:
                logger.error(f"Dosya silme hatası ({file_path}): {str(e)}") 