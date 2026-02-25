import { useRef, useState } from 'react';
import { uploadPDF } from './lib/api';
import { Upload, Loader2, FileCheck, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

export default function PdfUploader({ onUploaded }) {
  const inputRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleUpload = async (file) => {
    if (!file || !file.type.includes('pdf')) {
      setError('Please upload a valid PDF file');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('File size must be less than 50MB');
      return;
    }

    setLoading(true);
    setError(null);
    setUploadProgress(0);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + Math.random() * 30, 90));
      }, 200);

      await uploadPDF(file);

      clearInterval(progressInterval);
      setUploadProgress(100);

      setTimeout(() => {
        onUploaded(file, URL.createObjectURL(file));
        setLoading(false);
      }, 500);
    } catch (err) {
      setError('Failed to upload PDF. Please try again.');
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleUpload(files[0]);
    }
  };

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleUpload(file);
    }
  };

  return (
    <div className='min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900'>
      <Card className='w-full max-w-md border-0 shadow-lg'>
        <div className='p-8'>
          {/* Header */}
          <div className='mb-8 text-center'>
            <div className='inline-flex h-46 w-46 items-center justify-center  mb-4'>
              <img src="/logo.png" alt="logo" />
            </div>
            <h1 className='text-3xl font-bold text-slate-900 dark:text-white mb-2'>
              RAG Chatbot
            </h1>
            <p className='text-sm text-slate-600 dark:text-slate-400'>
              Upload your PDF and ask intelligent questions
            </p>
          </div>

          {/* Upload Area */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`relative mb-6 rounded-lg border-2 border-dashed transition-all duration-200 p-8 text-center cursor-pointer ${
              dragActive
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                : 'border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600'
            }`}
          >
            <input
              ref={inputRef}
              type='file'
              accept='.pdf'
              onChange={handleChange}
              disabled={loading}
              className='hidden'
            />

            {loading ? (
              <div className='space-y-3'>
                <Loader2 className='mx-auto text-blue-500 animate-spin' size={32} />
                <p className='text-sm font-medium text-slate-700 dark:text-slate-300'>
                  Processing your PDF...
                </p>
                <Progress value={uploadProgress} className='h-2' />
                <p className='text-xs text-slate-500 dark:text-slate-400'>
                  {Math.round(uploadProgress)}%
                </p>
              </div>
            ) : (
              <div className='space-y-2'>
                <Upload
                  size={32}
                  className='mx-auto text-slate-400 dark:text-slate-600'
                />
                <div>
                  <p className='text-sm font-semibold text-slate-900 dark:text-white'>
                    Drag & drop your PDF here
                  </p>
                  <p className='text-xs text-slate-500 dark:text-slate-400 mt-1'>
                    or click to browse
                  </p>
                </div>
              </div>
            )}

            <button
              onClick={() => inputRef.current?.click()}
              disabled={loading}
              className='absolute inset-0 rounded-lg'
            />
          </div>

          {/* Button */}
          <Button
            onClick={() => inputRef.current?.click()}
            disabled={loading}
            className='w-full h-11 text-base font-semibold text-white shadow-lg transition-all'
          >
            {loading ? (
              <>
                <Loader2 size={18} className='mr-2 animate-spin' />
                Uploading...
              </>
            ) : (
              <>
                <FileCheck size={18} className='mr-2' />
                Choose PDF
              </>
            )}
          </Button>

          {/* Error Message */}
          {error && (
            <div className='mt-4 p-3 rounded-lg bg-red-50 dark:bg-red-950 flex gap-3'>
              <AlertCircle size={18} className='text-red-500 flex-shrink-0 mt-0.5' />
              <p className='text-sm text-red-700 dark:text-red-300'>{error}</p>
            </div>
          )}

          {/* Info */}
          <p className='mt-6 text-xs text-slate-500 dark:text-slate-400 text-center'>
            Supported: PDF files up to 50MB
          </p>
        </div>
      </Card>
    </div>
  );
}