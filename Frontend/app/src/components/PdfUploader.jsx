import { useRef, useState } from 'react';
import axios from 'axios'; // Using axios to match your ChatInterface
import { Upload, Loader2, FileCheck, AlertCircle, FileText, Image as ImageIcon, FileBox } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

// List of supported formats for Docling
const SUPPORTED_FORMATS = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // docx
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',     // xlsx
  'application/vnd.openxmlformats-officedocument.presentationml.presentation', // pptx
  'image/png', 'image/jpeg', 'image/tiff', 'text/html'
];

export default function UniversalUploader({ onUploaded, token }) {
  const inputRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleUpload = async (file) => {
    // 1. Validation for All-in-One
    const isSupported = SUPPORTED_FORMATS.includes(file.type) ||
      file.name.match(/\.(pdf|docx|xlsx|pptx|png|jpg|jpeg|html)$/i);

    if (!isSupported) {
      setError('Format not supported. Please upload PDF, Office, or Images.');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('File size must be less than 50MB');
      return;
    }

    setLoading(true);
    setError(null);
    setSelectedFile(file);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Simulate progress UI
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 5, 95));
      }, 300);

      const response = await axios.post('http://127.0.0.1:8000/ingest', formData, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      setTimeout(() => {
        // Pass the file, the preview URL, and the doc_id from backend
        onUploaded(file, URL.createObjectURL(file), response.data.doc_id);
        setLoading(false);
      }, 500);
    } catch (err) {
      setError('Ingestion failed. Docling could not process the layout.');
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleUpload(e.dataTransfer.files[0]);
  };

  const getFileIcon = () => {
    if (!selectedFile) return <Upload size={32} className='mx-auto text-slate-400' />;
    if (selectedFile.type.includes('image')) return <ImageIcon size={32} className='mx-auto text-blue-500' />;
    if (selectedFile.type.includes('pdf')) return <FileText size={32} className='mx-auto text-red-500' />;
    return <FileBox size={32} className='mx-auto text-teal-500' />;
  };

  return (
    <div className='min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900'>
      <Card className='w-full max-w-md border-0 shadow-lg overflow-hidden'>
        <div className='p-8'>
          <div className='mb-8 text-center'>
            <div className='inline-flex h-20 w-20 items-center justify-center mb-4 bg-white rounded-2xl shadow-sm p-2'>
              <img src="/logo.png" alt="logo" className="object-contain" />
            </div>
            <h1 className='text-3xl font-bold text-slate-900 dark:text-white mb-2 tracking-tight'>
              Deep Ingestor
            </h1>
            <p className='text-sm text-slate-600 dark:text-slate-400'>
              Upload any document for layout-aware AI chat
            </p>
          </div>

          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`relative mb-6 rounded-2xl border-2 border-dashed transition-all duration-300 p-10 text-center cursor-pointer ${dragActive
              ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/50 scale-[1.02]'
              : 'border-slate-200 dark:border-slate-800 hover:border-slate-400'
              }`}
          >
            <input
              ref={inputRef}
              type='file'
              accept='.pdf,.docx,.xlsx,.pptx,.png,.jpg,.jpeg,.html'
              onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
              disabled={loading}
              className='hidden'
            />

            {loading ? (
              <div className='space-y-4'>
                <Loader2 className='mx-auto text-blue-500 animate-spin' size={32} />
                <div className='space-y-2'>
                  <p className='text-sm font-semibold text-slate-700 dark:text-slate-200'>
                    Analyzing Document Structure...
                  </p>
                  <Progress value={uploadProgress} className='h-1.5' />
                  <p className='text-[10px] text-slate-500 uppercase tracking-widest'>
                    {Math.round(uploadProgress)}% Processed
                  </p>
                </div>
              </div>
            ) : (
              <div className='space-y-3 animate-in fade-in duration-500'>
                {getFileIcon()}
                <div>
                  <p className='text-sm font-bold text-slate-900 dark:text-white'>
                    {selectedFile ? selectedFile.name : "Drop your file here"}
                  </p>
                  <p className='text-xs text-slate-500 mt-1'>
                    {selectedFile ? "Ready to re-upload" : "PDF, Word, Excel, or Images"}
                  </p>
                </div>
              </div>
            )}

            <button
              onClick={() => inputRef.current?.click()}
              disabled={loading}
              className='absolute inset-0 rounded-2xl'
            />
          </div>

          <Button
            onClick={() => inputRef.current?.click()}
            disabled={loading}
            className='w-full h-12 text-base font-bold text-white shadow-xl hover:shadow-2xl transition-all rounded-xl bg-slate-900 hover:bg-slate-800 dark:bg-blue-600 dark:hover:bg-blue-700'
          >
            {loading ? (
              <>
                <Loader2 size={18} className='mr-2 animate-spin' />
                Docling Ingesting...
              </>
            ) : (
              <>
                <FileCheck size={18} className='mr-2' />
                Select Any File
              </>
            )}
          </Button>

          {error && (
            <div className='mt-4 p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-900/30 flex gap-3 animate-in slide-in-from-top-2'>
              <AlertCircle size={18} className='text-red-500 flex-shrink-0' />
              <p className='text-xs font-medium text-red-700 dark:text-red-300'>{error}</p>
            </div>
          )}

          <div className="mt-8 grid grid-cols-4 gap-2 opacity-40 grayscale group-hover:grayscale-0 transition-all">
            <div className="text-[10px] font-bold text-center border rounded py-1">PDF</div>
            <div className="text-[10px] font-bold text-center border rounded py-1">DOCX</div>
            <div className="text-[10px] font-bold text-center border rounded py-1">XLSX</div>
            <div className="text-[10px] font-bold text-center border rounded py-1">IMG</div>
          </div>
        </div>
      </Card>
    </div>
  );
}