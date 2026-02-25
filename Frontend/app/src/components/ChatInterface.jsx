import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
// ADDED: FileDown and Loader2 for the report generation button
import { Send, Upload, FileText, LogOut, ChevronRight, MessageSquare, Plus, X, Globe, ExternalLink, Bot, User, Activity, Mic, Volume2, Trash2, History, MoreHorizontal, StopCircle, Image as ImageIcon, FileDown, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import PDFViewer from './PDFViewer';

const API_BASE_URL = 'http://127.0.0.1:8000';

const ChatInterface = () => {
  const { token, logout, user } = useAuth();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [activeMedia, setActiveMedia] = useState(null);
  const [activeMediaType, setActiveMediaType] = useState(null);
  const [activeDocId, setActiveDocId] = useState(null);
  const [activePage, setActivePage] = useState(1);

  // ADDED: Loading state for report generation
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  const [documents, setDocuments] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [webSearchMode, setWebSearchMode] = useState(false);
  const [diagnosticMode, setDiagnosticMode] = useState(false);
  const [sidebarMode, setSidebarMode] = useState('documents');
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(null);

  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  useEffect(() => {
    fetchDocuments();
    fetchSessions();
  }, []);

  useEffect(() => {
    if (currentSessionId) {
      loadSessionMessages(currentSessionId);
    }
  }, [currentSessionId]);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/documents`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setDocuments(response.data);
    } catch (error) {
      console.error("Failed to fetch documents", error);
    }
  };

  const fetchSessions = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/history/sessions`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setSessions(response.data);
    } catch (error) {
      console.error("Failed to fetch sessions", error);
    }
  };

  const createNewSession = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/history/sessions`, {}, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setSessions(prev => [{ id: response.data.session_id, title: "New Chat", updated_at: new Date().toISOString() }, ...prev]);
      setCurrentSessionId(response.data.session_id);
      setMessages([]);
      setActiveMedia(null);
      setActiveMediaType(null);
      setActiveDocId(null);
    } catch (error) {
      console.error("Failed to create session", error);
    }
  };

  const loadSessionMessages = async (sessionId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/history/sessions/${sessionId}/messages`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setMessages(response.data);
    } catch (error) {
      console.error("Failed to load messages", error);
    }
  };

  const deleteSession = async (e, sessionId) => {
    e.stopPropagation();
    try {
      await axios.delete(`${API_BASE_URL}/history/sessions/${sessionId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      if (currentSessionId === sessionId) {
        setCurrentSessionId(null);
        setMessages([]);
      }
    } catch (error) {
      console.error("Failed to delete session", error);
    }
  };

  const saveMessageToHistory = async (role, content, citations = null) => {
    if (!currentSessionId) return;
    try {
      await axios.post(`${API_BASE_URL}/history/sessions/${currentSessionId}/messages`, {
        role, content, citations
      }, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      fetchSessions();
    } catch (error) {
      console.error("Failed to save message", error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/ingest`, formData, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });

      const docId = response.data.doc_id;
      await fetchDocuments();
      setActiveDocId(docId);
      setActiveMedia(file);
      setActiveMediaType(file.type);

      const sysMsg = `File "${file.name}" processed successfully.`;
      setMessages(prev => [...prev, { role: 'system', content: sysMsg }]);
      if (currentSessionId) saveMessageToHistory('system', sysMsg);

      setFile(null);
      setUploading(false);

    } catch (error) {
      console.error("Upload failed", error);
      setMessages(prev => [...prev, { role: 'system', content: 'Failed to upload file.' }]);
      setUploading(false);
    }
  };

  // ADDED: Report Generation Logic
  const handleGenerateReport = async () => {
    if (!activeDocId) return;
    setIsGeneratingReport(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/generate-report`, null, {
        params: { doc_id: activeDocId },
        headers: { 'Authorization': `Bearer ${token}` },
        responseType: 'blob' // Essential for downloading files like .docx
      });

      // Create a temporary link to trigger the browser download
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      // Extract filename from activeMedia or use default
      const originalName = activeMedia?.name ? activeMedia.name.split('.')[0] : `Doc_${activeDocId}`;
      link.setAttribute('download', `${originalName}_Metallurgy_Report.docx`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);

      const sysMsg = `Metallurgy report generated and downloaded successfully.`;
      setMessages(prev => [...prev, { role: 'system', content: sysMsg }]);
      if (currentSessionId) saveMessageToHistory('system', sysMsg);

    } catch (error) {
      console.error("Report generation failed", error);
      const errorMsg = error.response?.status === 404
        ? 'Failed to generate report. Make sure pages 1-2 contain valid metallurgical data.'
        : 'Error generating report. Please try again.';
      setMessages(prev => [...prev, { role: 'system', content: errorMsg }]);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Audio recording is not supported in this browser or context. Please use HTTPS or localhost.");
      return;
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const hasAudioInput = devices.some(device => device.kind === 'audioinput');

      if (!hasAudioInput) {
        alert("No microphone found. Please connect a microphone and try again.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('file', audioBlob, 'voice.wav');

        try {
          const response = await axios.post(`${API_BASE_URL}/voice/transcribe`, formData, {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'multipart/form-data'
            }
          });
          setInput(response.data.text);
        } catch (error) {
          console.error("Transcription failed", error);
          setMessages(prev => [...prev, { role: 'system', content: 'Voice transcription failed.' }]);
        }

        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone", error);
      if (error.name === 'NotFoundError') {
        alert("No microphone found. Please ensure a microphone is connected.");
      } else if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        alert("Microphone permission denied. Please allow microphone access in your browser settings.");
      } else {
        alert(`Error accessing microphone: ${error.message}. If you are not using localhost, ensure you are using HTTPS.`);
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSpeak = async (text, msgId) => {
    if (isSpeaking === msgId) {
      setIsSpeaking(null);
      return;
    }

    try {
      setIsSpeaking(msgId);
      const formData = new FormData();
      formData.append('text', text);

      const response = await axios.post(`${API_BASE_URL}/voice/speak`, formData, {
        headers: { 'Authorization': `Bearer ${token}` },
        responseType: 'blob'
      });

      const audioUrl = URL.createObjectURL(response.data);
      const audio = new Audio(audioUrl);
      audio.onended = () => setIsSpeaking(null);
      audio.play();
    } catch (error) {
      console.error("TTS failed", error);
      setIsSpeaking(null);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    let sessionId = currentSessionId;
    if (!sessionId) {
      try {
        const res = await axios.post(`${API_BASE_URL}/history/sessions`, {}, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        sessionId = res.data.session_id;
        setCurrentSessionId(sessionId);
        setSessions(prev => [{ id: sessionId, title: input.substring(0, 30), updated_at: new Date().toISOString() }, ...prev]);
      } catch (e) {
        console.error("Failed to auto-create session", e);
      }
    }

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    if (sessionId) saveMessageToHistory('user', input);

    const currentInput = input;
    setInput('');

    const loadingMessage = { role: 'assistant', content: 'Thinking...', isLoading: true };
    setMessages(prev => [...prev, loadingMessage]);
    scrollToBottom();

    try {
      let response;

      if (diagnosticMode) {
        response = await axios.post(`${API_BASE_URL}/diagnose`, null, {
          params: { question: currentInput },
          headers: { 'Authorization': `Bearer ${token}` }
        });

        const result = response.data.result;
        const type = response.data.type;
        let markdownResponse = "";

        if (type === "diagnostic") {
          markdownResponse += `### 🩺 Diagnostic Report\n\n`;
          if (result.likely_causes?.length > 0) {
            markdownResponse += `#### 🔍 Likely Causes\n`;
            result.likely_causes.forEach(cause => {
              markdownResponse += `- **${cause.cause}** (${(cause.probability * 100).toFixed(0)}%)\n  - *Solution:* ${cause.solution}\n  - *Evidence:* ${cause.evidence}\n`;
            });
            markdownResponse += `\n`;
          }
          if (result.immediate_actions?.length > 0) {
            markdownResponse += `#### ⚡ Immediate Actions\n`;
            result.immediate_actions.forEach(action => markdownResponse += `- ${action}\n`);
            markdownResponse += `\n`;
          }
          if (result.safety_warnings?.length > 0) {
            markdownResponse += `#### ⚠️ Safety Warnings\n`;
            result.safety_warnings.forEach(warning => markdownResponse += `- ${warning}\n`);
          }
        } else if (type === "compliance") {
          markdownResponse += `### 🛡️ Compliance & Safety Report\n\n**Assessment:** ${result.assessment}\n\n`;
          if (result.standards?.length > 0) {
            markdownResponse += `#### 📜 Applicable Standards\n`;
            result.standards.forEach(std => markdownResponse += `- ${std}\n`);
          }
          if (result.required_ppe?.length > 0) {
            markdownResponse += `\n#### 🦺 Required PPE\n`;
            result.required_ppe.forEach(ppe => markdownResponse += `- [ ] ${ppe}\n`);
          }
          if (result.risks?.length > 0) {
            markdownResponse += `\n#### ⚠️ Potential Risks\n`;
            result.risks.forEach(risk => markdownResponse += `- ${risk}\n`);
          }
        } else if (type === "training") {
          markdownResponse += `### 🎓 Training Module: ${result.module_title}\n\n`;
          if (result.learning_objectives?.length > 0) {
            markdownResponse += `#### 🎯 Learning Objectives\n`;
            result.learning_objectives.forEach(obj => markdownResponse += `- ${obj}\n`);
          }
          if (result.steps?.length > 0) {
            markdownResponse += `\n#### 📝 Procedure\n`;
            result.steps.forEach(step => {
              markdownResponse += `**Step ${step.step}:** ${step.instruction}\n`;
              if (step.warning) markdownResponse += `> ⚠️ *Warning: ${step.warning}*\n`;
              markdownResponse += `\n`;
            });
          }
          if (result.quiz?.length > 0) {
            markdownResponse += `#### 🧠 Knowledge Check\n`;
            result.quiz.forEach((q, idx) => {
              markdownResponse += `**Q${idx + 1}: ${q.question}**\n`;
              q.options.forEach(opt => markdownResponse += `- [${opt === q.correct_answer ? 'x' : ' '}] ${opt}\n`);
              markdownResponse += `\n`;
            });
          }
        } else {
          markdownResponse = result.message || "No structured data available.";
        }

        response = {
          data: {
            response: {
              answer: markdownResponse,
              citations: result.document_refs ? result.document_refs.map(ref => ({
                page: ref.page,
                domain: ref.filename
              })) : []
            }
          }
        };

      } else if (webSearchMode) {
        response = await axios.post(`${API_BASE_URL}/web-query`, null, {
          params: { question: currentInput },
          headers: { 'Authorization': `Bearer ${token}` }
        });
      } else {
        const params = { question: currentInput };
        if (activeDocId) params.doc_id = activeDocId;
        response = await axios.post(`${API_BASE_URL}/query`, null, {
          params: params,
          headers: { 'Authorization': `Bearer ${token}` }
        });
      }

      setMessages(prev => {
        const newMessages = prev.filter(msg => !msg.isLoading);
        const assistantMsg = {
          role: 'assistant',
          content: response.data.response.answer,
          citations: response.data.response.citations,
          isWebSearch: webSearchMode,
          isDiagnostic: diagnosticMode
        };
        if (sessionId) saveMessageToHistory('assistant', assistantMsg.content, assistantMsg.citations);
        return [...newMessages, assistantMsg];
      });
      scrollToBottom();
    } catch (error) {
      console.error("Query failed", error);
      setMessages(prev => {
        const newMessages = prev.filter(msg => !msg.isLoading);
        return [...newMessages, { role: 'system', content: 'Failed to get response. Please try again.' }];
      });
    }
  };

  const handleCitationClick = (page) => {
    setActivePage(page);
  };

  const selectDocument = async (doc) => {
    setActiveDocId(doc.id);
    try {
      const response = await axios.get(`${API_BASE_URL}/documents/${doc.id}/file`, {
        headers: { 'Authorization': `Bearer ${token}` },
        responseType: 'blob'
      });
      const contentType = response.data.type || 'application/pdf';
      const fileObj = new File([response.data], doc.filename, { type: contentType });
      setActiveMedia(fileObj);
      setActiveMediaType(contentType);
      setMessages(prev => [...prev, { role: 'system', content: `Switched context to "${doc.filename}".` }]);
    } catch (error) {
      console.error("Failed to load document", error);
    }
  };

  return (
    <div className="flex h-screen bg-white text-stone-800 overflow-hidden font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-gradient-to-b from-stone-50 to-white border-r border-stone-200 flex flex-col shadow-lg">
        <div className="p-4 flex items-center gap-3 border-b border-stone-200 bg-white">
          <div className="flex items-center gap-2 bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl p-1.5 shadow-md">
            <img src="./logo.png" className="w-8 h-8" alt="Logo" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-stone-800 tracking-tight">Custom AI Chatbot</h1>
            <p className="text-xs text-stone-500 font-medium">RAG-powered assistant</p>
          </div>
        </div>

        <div className="flex border-b border-stone-200">
          <button
            onClick={() => setSidebarMode('documents')}
            className={`flex-1 py-2 text-xs font-semibold ${sidebarMode === 'documents' ? 'text-stone-900 border-b-2 border-stone-900' : 'text-stone-400 hover:text-stone-600'}`}
          >
            Documents
          </button>
          <button
            onClick={() => setSidebarMode('history')}
            className={`flex-1 py-2 text-xs font-semibold ${sidebarMode === 'history' ? 'text-stone-900 border-b-2 border-stone-900' : 'text-stone-400 hover:text-stone-600'}`}
          >
            History
          </button>
        </div>

        <div className="p-3 flex-1 overflow-y-auto scroll-smooth">
          {sidebarMode === 'documents' ? (
            <>
              <div className="mb-6">
                <h3 className="text-[10px] font-bold text-stone-400 uppercase tracking-wider mb-3 pl-2">Upload Document</h3>
                <div className="flex flex-col gap-2">
                  <input type="file" accept=".pdf,.docx,.xlsx,.pptx,.png,.jpg,.jpeg,.tiff,.html" onChange={handleFileChange} className="hidden" id="file-upload" />
                  <label htmlFor="file-upload" className="flex items-center justify-center px-3 py-2.5 bg-white hover:bg-stone-50 rounded-xl cursor-pointer transition-all border border-stone-200 hover:border-stone-300 hover:shadow-md text-xs text-stone-600 shadow-sm group">
                    <Plus size={14} strokeWidth={2} className="mr-2 text-stone-400 group-hover:text-stone-600 transition-colors" />
                    {file ? <span className="font-medium text-stone-900">{file.name.substring(0, 15)}...</span> : 'Select File'}
                  </label>
                  <button onClick={handleUpload} disabled={!file || uploading} className={`flex items-center justify-center px-3 py-2.5 rounded-xl text-xs font-semibold transition-all shadow-sm ${!file || uploading ? 'bg-stone-100 text-stone-400 cursor-not-allowed' : 'bg-gradient-to-r from-stone-900 to-stone-800 hover:from-stone-800 hover:to-stone-700 text-white shadow-md hover:shadow-lg'}`}>
                    {uploading ? (
                      <span className="flex items-center gap-2">
                        <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                        Processing...
                      </span>
                    ) : (
                      <> <Upload size={14} strokeWidth={2} className="mr-2" /> Upload File </>
                    )}
                  </button>
                </div>
              </div>

              <div>
                <h3 className="text-[10px] font-bold text-stone-400 uppercase tracking-wider mb-3 pl-2">Your Documents</h3>
                <div className="space-y-1">
                  <button onClick={() => { setActiveDocId(null); setActiveMedia(null); setActiveMediaType(null); setMessages(prev => [...prev, { role: 'system', content: 'Switched to All Documents context.' }]); }} className={`w-full flex items-center px-3 py-2.5 rounded-xl text-xs transition-all ${activeDocId === null ? 'bg-gradient-to-r from-teal-50 to-blue-50 text-teal-700 shadow-sm border border-teal-200 font-semibold' : 'text-stone-500 hover:bg-white hover:text-stone-700 hover:shadow-sm'}`}>
                    <MessageSquare size={14} strokeWidth={2} className={`mr-2.5 ${activeDocId === null ? 'text-teal-600' : 'text-stone-400'}`} />
                    All Documents
                  </button>
                  {documents.map(doc => {
                    const isImg = doc.filename.match(/\.(jpg|jpeg|png|tiff)$/i);
                    return (
                      <button key={doc.id} onClick={() => selectDocument(doc)} className={`w-full flex items-center px-3 py-2.5 rounded-xl text-xs transition-all ${activeDocId === doc.id ? 'bg-gradient-to-r from-teal-50 to-blue-50 text-teal-700 shadow-sm border border-teal-200 font-semibold' : 'text-stone-500 hover:bg-white hover:text-stone-700 hover:shadow-sm'}`}>
                        {isImg ? (
                          <ImageIcon size={14} strokeWidth={2} className={`mr-2.5 ${activeDocId === doc.id ? 'text-teal-600' : 'text-stone-400'}`} />
                        ) : (
                          <FileText size={14} strokeWidth={2} className={`mr-2.5 ${activeDocId === doc.id ? 'text-teal-600' : 'text-stone-400'}`} />
                        )}
                        <span className="truncate text-left">{doc.filename}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </>
          ) : (
            <div className="space-y-2">
              <button onClick={createNewSession} className="w-full flex items-center justify-center px-3 py-2 bg-stone-900 text-white rounded-xl text-xs font-semibold shadow-md hover:bg-stone-800 transition-all mb-4">
                <Plus size={14} className="mr-2" /> New Chat
              </button>
              {sessions.map(session => (
                <div key={session.id} onClick={() => { setCurrentSessionId(session.id); loadSessionMessages(session.id); }} className={`group flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer transition-all ${currentSessionId === session.id ? 'bg-stone-100 border border-stone-200' : 'hover:bg-stone-50'}`}>
                  <div className="flex items-center gap-2 overflow-hidden">
                    <History size={14} className="text-stone-400" />
                    <div className="flex flex-col">
                      <span className={`text-xs font-medium truncate w-32 ${currentSessionId === session.id ? 'text-stone-900' : 'text-stone-600'}`}>{session.title}</span>
                      <span className="text-[10px] text-stone-400">{new Date(session.updated_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <button onClick={(e) => deleteSession(e, session.id)} className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-50 hover:text-red-500 rounded-lg transition-all">
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="p-3 border-t border-stone-200 bg-white">
          <button onClick={logout} className="flex items-center w-full px-3 py-2.5 text-xs text-stone-500 hover:text-red-600 hover:bg-red-50 rounded-xl transition-all font-medium">
            <LogOut size={14} strokeWidth={2} className="mr-2" /> Sign Out
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex relative bg-gradient-to-br from-stone-50 via-white to-stone-50">
        <div className={`flex-1 flex flex-col ${activeMedia ? 'w-1/2' : 'w-full'} transition-all duration-500 ease-in-out`}>
          <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-stone-400 animate-fade-in">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-purple-100 rounded-2xl mb-6 flex items-center justify-center border border-stone-200 shadow-lg">
                  <MessageSquare size={32} strokeWidth={1.5} className="text-stone-400" />
                </div>
                <p className="text-lg font-semibold text-stone-700 mb-2">Welcome to RAG Chatbot</p>
                <p className="text-sm text-stone-500 max-w-md text-center">Select a document, upload a file or image, or use web search to get started</p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'} group animate-fade-in`}>
                {msg.role === 'assistant' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-stone-900 flex items-center justify-center shadow-sm">
                    <Bot size={16} className="text-white" />
                  </div>
                )}

                <div className={`max-w-[85%] ${msg.role === 'user' ? 'order-1' : ''} ${msg.role === 'user' ? 'bg-gradient-to-br from-stone-900 to-stone-800 text-white rounded-2xl rounded-tr-sm shadow-lg' : msg.role === 'system' ? 'bg-stone-50 text-stone-500 text-xs border border-stone-100 w-full text-center py-2 rounded-lg' : 'bg-white text-stone-800 rounded-2xl rounded-tl-sm border border-stone-200 shadow-md hover:shadow-lg transition-shadow'}`}>
                  {msg.role === 'user' && (
                    <div className="flex items-center gap-2 px-4 pt-3 pb-2">
                      <div className="w-6 h-6 rounded-full bg-stone-700 flex items-center justify-center">
                        <User size={12} className="text-white" />
                      </div>
                      <span className="text-xs font-medium text-stone-300">You</span>
                    </div>
                  )}

                  <div className={`px-4 ${msg.role === 'user' ? 'pb-3 pt-0' : 'py-3'}`}>
                    {msg.role === 'assistant' ? (
                      msg.isLoading ? (
                        <div className="flex items-center gap-2 text-stone-500">
                          <span className="text-sm">Thinking...</span>
                        </div>
                      ) : (
                        <div className="prose prose-sm max-w-none">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                        </div>
                      )
                    ) : (
                      <p className="leading-relaxed whitespace-pre-wrap text-sm">{msg.content}</p>
                    )}

                    {msg.citations && msg.citations.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-stone-200">
                        <p className="text-[10px] font-bold text-stone-400 mb-2.5 uppercase tracking-widest">Sources</p>
                        <div className="flex flex-wrap gap-2">
                          {msg.citations.map((cit, cIdx) => (
                            <button key={cIdx} onClick={() => handleCitationClick(cit.page)} className="flex items-center px-2.5 py-1.5 bg-stone-50 hover:bg-stone-100 border border-stone-200 rounded-lg text-[10px] text-stone-600 transition-all hover:border-stone-300 hover:shadow-sm group">
                              <FileText size={11} className="mr-1.5 text-stone-500" />
                              <span className="font-medium">Page {cit.page}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  {msg.role === 'assistant' && !msg.isLoading && (
                    <div className="px-4 pb-2 flex justify-end">
                      <button onClick={() => handleSpeak(msg.content, idx)} className={`p-1.5 rounded-full hover:bg-stone-100 transition-colors ${isSpeaking === idx ? 'text-blue-500 animate-pulse' : 'text-stone-400'}`}>
                        <Volume2 size={14} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-4 bg-white/90 backdrop-blur-md border-t border-stone-200 shadow-lg">
            <div className="max-w-4xl mx-auto">
              {webSearchMode && <div className="mb-3 flex items-center gap-2 text-xs text-blue-600 bg-blue-50 px-3 py-1.5 rounded-lg border border-blue-200 animate-fade-in"><Globe size={14} className="animate-pulse" /><span className="font-semibold">Web Search Mode Active</span></div>}
              {diagnosticMode && <div className="mb-3 flex items-center gap-2 text-xs text-red-600 bg-red-50 px-3 py-1.5 rounded-lg border border-red-200 animate-fade-in"><Activity size={14} className="animate-pulse" /><span className="font-semibold">Diagnostic Mode Active</span></div>}

              <div className={`flex items-center gap-2 bg-white rounded-2xl p-2 border transition-all shadow-lg hover:shadow-xl ${webSearchMode ? 'border-blue-300 ring-2 ring-blue-100' : diagnosticMode ? 'border-red-300 ring-2 ring-red-100' : 'border-stone-200 focus-within:border-stone-400 focus-within:ring-2 focus-within:ring-stone-100'}`}>
                <input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()} placeholder={webSearchMode ? "Search the web..." : diagnosticMode ? "Describe the symptoms..." : "Ask a question..."} className="flex-1 bg-transparent border-none outline-none text-sm text-stone-900 px-4 py-2.5 placeholder:text-stone-400" />

                <button onClick={isRecording ? stopRecording : startRecording} className={`p-2.5 rounded-xl transition-all ${isRecording ? 'bg-red-500 text-white animate-pulse' : 'bg-stone-100 text-stone-500 hover:bg-stone-200'}`} title="Voice Input">
                  {isRecording ? <StopCircle size={18} /> : <Mic size={18} />}
                </button>

                <button onClick={() => { setDiagnosticMode(!diagnosticMode); if (!diagnosticMode) setWebSearchMode(false); }} className={`p-2.5 rounded-xl transition-all ${diagnosticMode ? 'bg-gradient-to-br from-red-100 to-orange-100 text-red-600 shadow-sm' : 'bg-stone-100 text-stone-500 hover:bg-stone-200'}`} title="Diagnostic Mode">
                  <Activity size={18} strokeWidth={2} />
                </button>

                <button onClick={() => { setWebSearchMode(!webSearchMode); if (!webSearchMode) setDiagnosticMode(false); }} className={`p-2.5 rounded-xl transition-all ${webSearchMode ? 'bg-gradient-to-br from-blue-100 to-indigo-100 text-blue-600 shadow-sm' : 'bg-stone-100 text-stone-500 hover:bg-stone-200'}`} title="Web Search">
                  <Globe size={18} strokeWidth={2} />
                </button>

                <button onClick={handleSend} disabled={!input.trim()} className="p-2.5 bg-gradient-to-br from-stone-900 to-stone-800 hover:from-stone-800 hover:to-stone-700 text-white rounded-xl transition-all disabled:opacity-30 disabled:cursor-not-allowed shadow-md hover:shadow-lg">
                  <Send size={18} strokeWidth={2} />
                </button>
              </div>
              <p className="text-[10px] text-stone-400 mt-2 text-center">Press Enter to send, Shift+Enter for new line</p>
            </div>
          </div>
        </div>

        {activeMedia && (
          <div className="w-1/2 border-l border-stone-200 bg-stone-50 flex flex-col shadow-xl z-10">
            <div className="p-3 border-b border-stone-200 flex justify-between items-center bg-white">
              <div className="flex items-center gap-2 overflow-hidden">
                <div className={`p-1 rounded ${activeMediaType?.startsWith('image/') ? 'bg-blue-50' : 'bg-red-50'}`}>
                  {activeMediaType?.startsWith('image/') ? (
                    <ImageIcon size={14} className="text-blue-500" />
                  ) : (
                    <FileText size={14} className="text-red-500" />
                  )}
                </div>
                <span className="text-xs font-semibold text-stone-700 truncate">{activeMedia.name}</span>
              </div>

              {/* ADDED: Report button logic here alongside the close button */}
              <div className="flex items-center gap-2">
                <button
                  onClick={handleGenerateReport}
                  disabled={isGeneratingReport}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 rounded-lg text-xs font-semibold transition-colors disabled:opacity-50 border border-indigo-100"
                  title="Generate Metallurgy Report"
                >
                  {isGeneratingReport ? <Loader2 size={14} className="animate-spin" /> : <FileDown size={14} />}
                  {isGeneratingReport ? 'Generating...' : 'Report'}
                </button>
                <button onClick={() => { setActiveMedia(null); setActiveMediaType(null); }} className="p-1.5 hover:bg-stone-100 rounded-full text-stone-400 hover:text-stone-600 transition-colors">
                  <X size={16} />
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-hidden relative bg-stone-100/50">
              <PDFViewer file={activeMedia} pageNumber={activePage} type={activeMediaType} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;