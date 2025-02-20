import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator
import logging
import os
import re

logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    LLM işlemlerini yöneten sınıf.
    """
    
    def __init__(self):
        """
        LLM işlemcisini başlatır ve gerekli modelleri yükler.
        """
        # GPU kontrolü
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"GPU Kullanılıyor: {torch.cuda.get_device_name(0)}")
            logger.info(f"Kullanılabilir GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            logger.info(f"Kullanılan GPU Belleği: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            # GPU bellek optimizasyonları
            torch.cuda.empty_cache()
        else:
            logger.info("GPU bulunamadı, CPU kullanılıyor")
        
        self.GENERATION_MODEL_NAME = "deepseek-ai/deepseek-r1-distill-qwen-32b"
        self.HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.translator = Translator()
        
        # DeepSeek modeli yükleme
        try:
            logger.info("DeepSeek-R1-Distill-Qwen-32B modeli yükleniyor...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.GENERATION_MODEL_NAME,
                token=self.HF_TOKEN,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.GENERATION_MODEL_NAME,
                torch_dtype=self.torch_dtype,
                token=self.HF_TOKEN,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True
            )
            logger.info("DeepSeek-R1-Distill-Qwen-32B modeli başarıyla yüklendi")
        except Exception as e:
            logger.error(f"DeepSeek-R1-Distill-Qwen-32B model yükleme hatası: {str(e)}")
            self.model = None
            self.tokenizer = None

    def get_gpu_info(self) -> dict:
        """
        GPU durum bilgilerini döndürür.
        
        Returns:
            dict: GPU bilgileri
        """
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": None,
            "total_memory": None,
            "used_memory": None
        }
        
        if gpu_info["available"]:
            gpu_info.update({
                "device_name": torch.cuda.get_device_name(0),
                "total_memory": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "used_memory": torch.cuda.memory_allocated(0) / 1024**3
            })
        
        return gpu_info

    def translate_to_english(self, text: str) -> str:
        """
        Türkçe metni İngilizce'ye çevirir.
        
        Args:
            text (str): Çevrilecek Türkçe metin
        
        Returns:
            str: Çevrilmiş İngilizce metin
        """
        try:
            translation = self.translator.translate(text, src='tr', dest='en')
            return translation.text
        except Exception as e:
            logger.error(f"Çeviri hatası: {str(e)}")
            return text

    def count_tokens(self, text: str) -> int:
        """
        Verilen metnin token sayısını hesaplar.
        
        Args:
            text (str): Token sayısı hesaplanacak metin
        
        Returns:
            int: Toplam token sayısı
        """
        return len(self.tokenizer.encode(text))

    def format_prompt(self, query: str, context: str = "") -> str:
        """
        DeepSeek-R1-Distill-Qwen formatında prompt oluşturur.
        
        Args:
            query (str): Kullanıcı sorgusu
            context (str): İlgili bağlam bilgisi
        
        Returns:
            str: Formatlanmış prompt
        """
        system_prompt = """You are a professional automotive sales data analyst. Here is the relevant data for the query:

CONTEXT:
{context}

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

        prompt = f"""<|im_start|>system
{system_prompt.format(context=context)}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant"""

        return prompt

    async def generate_response(self, query: str, context: str = "") -> str:
        """
        Verilen sorgu için LLM yanıtı üretir.
        
        Args:
            query (str): Kullanıcı sorgusu
            context (str): Bağlam bilgisi
            
        Returns:
            str: Üretilen yanıt
        """
        if not self.model:
            return "Üzgünüm, model yüklenemediği için şu anda hizmet veremiyorum."
            
        try:
            # Prompt oluştur
            prompt = self.format_prompt(query, context)
            
            # Model girdisini hazırla
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=8192,  # Qwen modeli daha uzun sekans destekliyor
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Yanıt üret
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=8192,
                max_new_tokens=2048,  # Daha uzun yanıtlar için
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Yanıtı decode et
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Asistan yanıtını ayıkla
            if "<|im_start|>assistant" in response:
                answer = response.split("<|im_start|>assistant")[-1].strip()
                if "<|im_end|>" in answer:
                    answer = answer.split("<|im_end|>")[0].strip()
            else:
                answer = response.strip()
            
            answer = " ".join(answer.split())
            
            # Türkçe'ye çevir
            try:
                turkish_answer = self.translator.translate(answer, src='en', dest='tr').text
                
                # Sayıları formatla
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
            logger.error(f"Yanıt üretme hatası: {str(e)}")
            return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin." 