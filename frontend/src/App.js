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
  CircularProgress,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://213.181.123.87:43103';
const WS_URL = BACKEND_URL.replace('http', 'ws');
console.log('Backend URL:', BACKEND_URL);
console.log('WebSocket URL:', WS_URL);

function App() {
  const [messages, setMessages] = useState([]);
  const [connection, setConnection] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentAudio, setCurrentAudio] = useState(null);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const reconnectTimeout = useRef(null);
  const maxReconnectAttempts = 5;
  const reconnectAttempts = useRef(0);

  const connectWebSocket = () => {
    try {
      console.log('WebSocket bağlantısı başlatılıyor:', `${WS_URL}/ws`);
      const ws = new WebSocket(`${WS_URL}/ws`, [], {
        rejectUnauthorized: false
      });
      setConnection(ws);

      ws.onopen = () => {
        console.log('WebSocket bağlantısı başarıyla kuruldu');
        reconnectAttempts.current = 0;
        setMessages(prev => [...prev, { type: 'system', text: 'Bağlantı kuruldu' }]);
      };

      ws.onmessage = async (event) => {
        console.log('WebSocket mesajı alındı:', event.data);
        try {
          const response = JSON.parse(event.data);
          
          setMessages(prev => {
            const newMessages = [...prev];
            if (newMessages.length > 0 && newMessages[newMessages.length - 1].text === 'Dinleniyor...') {
              newMessages.pop();
            }
            return newMessages;
          });

          if (response.recognized_text) {
            console.log('Tanınan ses:', response.recognized_text);
            setMessages(prev => [...prev, { type: 'user', text: response.recognized_text }]);
          }
          
          console.log('Bot yanıtı:', response.text);
          setMessages(prev => [...prev, { type: 'bot', text: response.text }]);
          
          if (response.audio) {
            console.log('Ses yanıtı çalınıyor');
            if (currentAudio) {
              currentAudio.pause();
              currentAudio.currentTime = 0;
            }
            const audio = new Audio(response.audio);
            setCurrentAudio(audio);
            await audio.play().catch(error => {
              console.error('Ses çalma hatası:', error);
            });
            audio.onended = () => {
              setCurrentAudio(null);
              startRecording();
            };
          }
          setIsProcessing(false);
        } catch (error) {
          console.error('WebSocket mesaj işleme hatası:', error);
          if (typeof event.data === 'string') {
            setMessages(prev => [...prev, { type: 'error', text: event.data }]);
          }
          setIsProcessing(false);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket bağlantı hatası:', error);
        console.error('WebSocket durumu:', ws.readyState);
        console.error('Bağlantı URL:', `${WS_URL}/ws`);
        console.error('Tam hata detayları:', {
          error,
          wsState: ws.readyState,
          wsUrl: `${WS_URL}/ws`,
          wsBufferedAmount: ws.bufferedAmount,
          wsExtensions: ws.extensions,
          wsProtocol: ws.protocol
        });
        setMessages(prev => [...prev, { type: 'error', text: 'Bağlantı hatası oluştu' }]);
      };

      ws.onclose = (event) => {
        console.log('WebSocket bağlantısı kapandı:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean
        });
        setConnection(null);
        
        if (event.code !== 1000 && event.code !== 1001) {
          if (reconnectAttempts.current < maxReconnectAttempts) {
            reconnectAttempts.current += 1;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
            console.log(`Yeniden bağlanılıyor (${reconnectAttempts.current}/${maxReconnectAttempts}) - ${delay}ms sonra`);
            setMessages(prev => [...prev, { type: 'system', text: `Yeniden bağlanılıyor (${reconnectAttempts.current}/${maxReconnectAttempts})...` }]);
            
            reconnectTimeout.current = setTimeout(() => {
              connectWebSocket();
            }, delay);
          } else {
            console.error('Maksimum yeniden bağlanma denemesi aşıldı');
            setMessages(prev => [...prev, { type: 'error', text: 'Bağlantı kurulamadı. Lütfen sayfayı yenileyin.' }]);
          }
        }
      };

      return ws;
    } catch (error) {
      console.error('WebSocket oluşturma hatası:', error);
      setMessages(prev => [...prev, { type: 'error', text: 'Bağlantı hatası: ' + error.message }]);
      return null;
    }
  };

  const startRecording = async () => {
    if (isRecording || isProcessing) return;
    
    if (!connection || connection.readyState !== WebSocket.OPEN) {
      console.log('WebSocket bağlantısı yok, yeniden bağlanılıyor...');
      await new Promise(resolve => setTimeout(resolve, 1000));
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
        setIsProcessing(true);
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
            setIsProcessing(false);
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
      setIsProcessing(false);
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
      if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
        mediaRecorder.current.stop();
        mediaRecorder.current.stream.getTracks().forEach(track => track.stop());
      }
      setIsRecording(false);
    } else {
      try {
        if (!navigator.mediaDevices) {
          throw new Error('Tarayıcınız mikrofon erişimini desteklemiyor. Lütfen HTTPS kullanın veya güvenilir bir bağlantı kurun.');
        }

        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 44100
          }
        });

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
        setMessages(prev => [...prev, { type: 'system', text: 'Kayıt başladı. Konuşabilirsiniz.' }]);

      } catch (error) {
        console.error('Mikrofon erişim hatası:', error);
        setMessages(prev => [...prev, { 
          type: 'error', 
          text: `Mikrofon hatası: ${error.message || 'Mikrofon erişimi sağlanamadı. Lütfen tarayıcı izinlerini kontrol edin.'}`
        }]);
        setIsRecording(false);
      }
    }
  };

  useEffect(() => {
    let mainWs = null;
    let welcomeWs = null;

    const initializeConnection = async () => {
      mainWs = connectWebSocket();
      
      welcomeWs = new WebSocket(`${WS_URL}/ws`);
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
        
        <Paper elevation={3} sx={{ p: 2, position: 'relative' }}>
          {isProcessing && (
            <Box sx={{ 
              position: 'absolute', 
              top: 0, 
              left: 0, 
              right: 0, 
              display: 'flex', 
              justifyContent: 'center', 
              p: 1 
            }}>
              <CircularProgress size={24} />
            </Box>
          )}
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 2 }}>
            <IconButton
              onClick={toggleRecording}
              disabled={isProcessing}
              color={isRecording ? 'secondary' : 'primary'}
            >
              {isRecording ? <StopIcon /> : <MicIcon />}
            </IconButton>
            <input
              accept=".json"
              style={{ display: 'none' }}
              id="upload-database"
              type="file"
              onChange={handleFileUpload}
              disabled={isProcessing}
            />
            <label htmlFor="upload-database">
              <IconButton
                component="span"
                disabled={isProcessing}
                color="primary"
              >
                <UploadFileIcon />
              </IconButton>
            </label>
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