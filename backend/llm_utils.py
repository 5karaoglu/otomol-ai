import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
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
        self.GENERATION_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32
        self.translator = Translator()
        
        # GPU bellek optimizasyonları
        torch.cuda.empty_cache()
        
        # LLaMA modeli yükleme
        try:
            logger.info("LLaMA modeli yükleniyor...")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.GENERATION_MODEL_NAME,
                token=self.HF_TOKEN
            )
            self.model = LlamaForCausalLM.from_pretrained(
                self.GENERATION_MODEL_NAME,
                torch_dtype=self.torch_dtype,
                token=self.HF_TOKEN,
                device_map="auto",
                load_in_8bit=True
            )
            logger.info("LLaMA modeli başarıyla yüklendi")
        except Exception as e:
            logger.error(f"LLaMA model yükleme hatası: {str(e)}")
            self.model = None
            self.tokenizer = None

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
        LLaMA-2-chat formatında prompt oluşturur.
        
        Args:
            query (str): Kullanıcı sorgusu
            context (str): İlgili bağlam bilgisi
        
        Returns:
            str: Formatlanmış prompt
        """
        SYSTEM_PROMPT = """You are a professional automotive sales data analyst. Here is the relevant data for the query:

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

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context=context)
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

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
                max_length=2048,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Yanıt üret
            outputs = self.model.generate(
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
            
            # Yanıtı decode et
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "<|assistant|>" in response:
                answer = response.split("<|assistant|>")[-1].strip()
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