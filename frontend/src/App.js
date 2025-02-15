import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  Button,
  IconButton,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://213.181.123.11:54722';
const WS_URL = BACKEND_URL.replace('http', 'ws');

function App() {
  const [messages, setMessages] = useState([]);
  const [connection, setConnection] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const reconnectTimeout = useRef(null);
  const maxReconnectAttempts = 5;
  const reconnectAttempts = useRef(0);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(`${WS_URL}/ws`);
      setConnection(ws);

      ws.onopen = () => {
        console.log('WebSocket bağlantısı kuruldu');
        reconnectAttempts.current = 0;
        setMessages(prev => [...prev, { type: 'system', text: 'Bağlantı kuruldu' }]);
      };

      ws.onmessage = async (event) => {
        try {
          const response = JSON.parse(event.data);
          
          // Son mesaj "Dinleniyor..." ise onu kaldır
          setMessages(prev => {
            const newMessages = [...prev];
            if (newMessages.length > 0 && newMessages[newMessages.length - 1].text === 'Dinleniyor...') {
              newMessages.pop();
            }
            return newMessages;
          });

          // Eğer tanınan ses varsa, kullanıcı mesajı olarak ekle
          if (response.recognized_text) {
            setMessages(prev => [...prev, { type: 'user', text: response.recognized_text }]);
          }
          
          // Bot yanıtını ekle
          setMessages(prev => [...prev, { type: 'bot', text: response.text }]);
          
          if (response.audio) {
            const audio = new Audio(response.audio);
            await audio.play().catch(console.error);
            startRecording();
          }
        } catch (error) {
          console.error('WebSocket mesaj hatası:', error);
          if (typeof event.data === 'string') {
            setMessages(prev => [...prev, { type: 'error', text: event.data }]);
          }
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket hatası:', error);
        setMessages(prev => [...prev, { type: 'error', text: 'Bağlantı hatası oluştu' }]);
      };

      ws.onclose = (event) => {
        console.log('WebSocket bağlantısı kapandı:', event.code);
        setConnection(null);
        
        // Planlı kapanma değilse yeniden bağlan
        if (event.code !== 1000 && event.code !== 1001) {
          if (reconnectAttempts.current < maxReconnectAttempts) {
            reconnectAttempts.current += 1;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
            setMessages(prev => [...prev, { type: 'system', text: `Yeniden bağlanılıyor (${reconnectAttempts.current}/${maxReconnectAttempts})...` }]);
            
            reconnectTimeout.current = setTimeout(() => {
              connectWebSocket();
            }, delay);
          } else {
            setMessages(prev => [...prev, { type: 'error', text: 'Bağlantı kurulamadı. Lütfen sayfayı yenileyin.' }]);
          }
        }
      };

      return ws;
    } catch (error) {
      console.error('WebSocket bağlantı hatası:', error);
      setMessages(prev => [...prev, { type: 'error', text: 'Bağlantı hatası: ' + error.message }]);
      return null;
    }
  };

  const startRecording = async () => {
    if (isRecording) return;
    
    // WebSocket bağlantısını kontrol et
    if (!connection || connection.readyState !== WebSocket.OPEN) {
      console.log('WebSocket bağlantısı yok, yeniden bağlanılıyor...');
      await new Promise(resolve => setTimeout(resolve, 1000)); // 1 saniye bekle
      const newWs = connectWebSocket();
      if (!newWs) {
        setMessages(prev => [...prev, { type: 'error', text: 'Bağlantı kurulamadı, tekrar deneniyor...' }]);
        setTimeout(startRecording, 2000);
        return;
      }
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      audioChunks.current = [];
      
      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      mediaRecorder.current.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm;codecs=opus' });
        const reader = new FileReader();
        
        reader.onload = async () => {
          if (connection && connection.readyState === WebSocket.OPEN) {
            setMessages(prev => [...prev, { type: 'system', text: 'Dinleniyor...' }]);
            connection.send(reader.result);
          } else {
            console.log('Ses gönderilemiyor, bağlantı kapalı');
            setMessages(prev => [...prev, { type: 'error', text: 'Bağlantı hatası, yeniden bağlanılıyor...' }]);
            connectWebSocket();
          }
        };
        
        reader.readAsDataURL(audioBlob);
      };

      mediaRecorder.current.start();
      setIsRecording(true);

    } catch (error) {
      console.error('Mikrofon erişim hatası:', error);
      setMessages(prev => [...prev, { type: 'error', text: 'Mikrofon hatası: ' + error.message }]);
      setIsRecording(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${BACKEND_URL}/upload-database`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setMessages(prev => [...prev, { type: 'system', text: 'Veritabanı başarıyla yüklendi' }]);
      } else {
        const errorText = await response.text();
        setMessages(prev => [...prev, { type: 'error', text: 'Veritabanı yükleme hatası: ' + errorText }]);
      }
    } catch (error) {
      console.error('Error uploading database:', error);
      setMessages(prev => [...prev, { type: 'error', text: 'Veritabanı yükleme hatası: ' + error.message }]);
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      // Kaydı durdur
      if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
        mediaRecorder.current.stop();
        mediaRecorder.current.stream.getTracks().forEach(track => track.stop());
      }
      setIsRecording(false);
    } else {
      // Yeni kayıt başlat
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder.current = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=opus'
        });
        audioChunks.current = [];
        
        mediaRecorder.current.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.current.push(event.data);
          }
        };

        mediaRecorder.current.onstop = async () => {
          const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm;codecs=opus' });
          const reader = new FileReader();
          
          reader.onload = async () => {
            if (connection && connection.readyState === WebSocket.OPEN) {
              setMessages(prev => [...prev, { type: 'system', text: 'Dinleniyor...' }]);
              connection.send(reader.result);
            } else {
              setMessages(prev => [...prev, { type: 'error', text: 'Bağlantı hatası, yeniden bağlanılıyor...' }]);
              connectWebSocket();
            }
          };
          
          reader.readAsDataURL(audioBlob);
        };

        mediaRecorder.current.start();
        setIsRecording(true);

      } catch (error) {
        console.error('Mikrofon erişim hatası:', error);
        setMessages(prev => [...prev, { type: 'error', text: 'Mikrofon hatası: ' + error.message }]);
        setIsRecording(false);
      }
    }
  };

  useEffect(() => {
    let mainWs = null;
    let welcomeWs = null;

    const initializeConnection = async () => {
      // Ana WebSocket bağlantısını kur
      mainWs = connectWebSocket();
      
      // Hoşgeldin mesajı için yeni WebSocket bağlantısı
      welcomeWs = new WebSocket('ws://localhost:8000/ws');
      welcomeWs.onopen = () => {
        welcomeWs.send(JSON.stringify({ type: 'welcome' }));
      };

      welcomeWs.onmessage = async (event) => {
        try {
          const response = JSON.parse(event.data);
          setMessages([{ type: 'bot', text: response.text }]);
          
          if (response.audio) {
            const audio = new Audio(response.audio);
            await audio.play().catch(console.error);
            welcomeWs.close();
          }
        } catch (error) {
          console.error('Hoşgeldin mesajı hatası:', error);
          welcomeWs.close();
        }
      };

      welcomeWs.onerror = () => {
        console.error('Hoşgeldin bağlantısı hatası');
        welcomeWs.close();
      };
    };

    initializeConnection();

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (mainWs) {
        mainWs.close(1000, 'Component unmounting');
      }
      if (welcomeWs) {
        welcomeWs.close();
      }
      if (mediaRecorder.current) {
        mediaRecorder.current.stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const getMessageColor = (type) => {
    switch (type) {
      case 'bot':
        return 'primary.main';
      case 'user':
        return 'success.main';
      case 'error':
        return 'error.main';
      case 'system':
        return 'text.secondary';
      default:
        return 'text.primary';
    }
  };

  return (
    <Container maxWidth="sm">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Türkçe Sesli Asistan
        </Typography>
        
        <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <input
              accept=".json"
              style={{ display: 'none' }}
              id="database-file"
              type="file"
              onChange={handleFileUpload}
            />
            <label htmlFor="database-file" style={{ flexGrow: 1 }}>
              <Button
                variant="contained"
                component="span"
                fullWidth
                startIcon={<UploadFileIcon />}
              >
                Veritabanı Yükle
              </Button>
            </label>
            <IconButton 
              color={isRecording ? "error" : "primary"}
              onClick={toggleRecording}
              sx={{ width: 56, height: 56 }}
            >
              {isRecording ? <StopIcon /> : <MicIcon />}
            </IconButton>
          </Box>
        </Paper>

        <Paper elevation={3} sx={{ p: 2, mb: 2, minHeight: '300px', maxHeight: '500px', overflow: 'auto' }}>
          <List>
            {messages.map((message, index) => (
              <ListItem key={index}>
                <ListItemText
                  primary={message.text}
                  secondary={
                    message.type === 'bot' ? 'OtomolAi' : 
                    message.type === 'user' ? 'Osman Bey' :
                    message.type === 'system' ? 'Sistem' : 
                    'Hata'
                  }
                  sx={{ 
                    color: getMessageColor(message.type),
                    textAlign: message.type === 'user' ? 'right' : 'left'
                  }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      </Box>
    </Container>
  );
}

export default App; 