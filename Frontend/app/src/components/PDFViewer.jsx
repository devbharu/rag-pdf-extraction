import React, { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// Set worker source
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

// ADDED: 'type' prop to help identify if it's an image
const PDFViewer = ({ file, pageNumber, type }) => {
  const [numPages, setNumPages] = useState(null);
  const [containerWidth, setContainerWidth] = useState(null);

  // ADDED: Check if the file is an image
  const isImage = type?.startsWith('image/') || file?.type?.startsWith('image/');

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
  }

  useEffect(() => {
    const updateWidth = () => {
      // Changed ID slightly to reflect it handles both now
      const container = document.getElementById('media-container');
      if (container) {
        setContainerWidth(container.clientWidth - 48); // Subtract padding (p-6 = 24px * 2)
      }
    };

    window.addEventListener('resize', updateWidth);
    // Small delay to ensure container is rendered
    setTimeout(updateWidth, 100);

    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Scroll to page when pageNumber changes (Only for PDFs)
  useEffect(() => {
    if (!isImage && pageNumber) {
      const pageElement = document.getElementById(`page_${pageNumber}`);
      if (pageElement) {
        pageElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }, [pageNumber, isImage]);

  return (
    <div id="media-container" className="h-full overflow-y-auto p-6 bg-stone-100 flex flex-col items-center scroll-smooth">
      {isImage ? (
        // ADDED: Image rendering logic
        <div className="flex-1 flex items-center justify-center w-full h-full">
          <img
            src={file instanceof File || file instanceof Blob ? URL.createObjectURL(file) : file}
            alt="Uploaded Document"
            className="max-w-full max-h-full object-contain shadow-md rounded-lg"
          />
        </div>
      ) : (
        // KEPT: Your exact PDF rendering logic
        <Document
          file={file}
          onLoadSuccess={onDocumentLoadSuccess}
          className="flex flex-col gap-6"
          loading={<div className="text-stone-500 font-medium text-sm animate-pulse">Loading PDF...</div>}
          error={<div className="text-red-500 font-medium text-sm">Failed to load PDF.</div>}
        >
          {Array.from(new Array(numPages), (el, index) => (
            <div key={`page_${index + 1}`} id={`page_${index + 1}`} className="shadow-sm transition-shadow hover:shadow-md">
              <Page
                pageNumber={index + 1}
                width={containerWidth || 400}
                renderTextLayer={true}
                renderAnnotationLayer={true}
                className="bg-white"
              />
              <div className="text-center text-[10px] text-stone-400 mt-2 font-medium">Page {index + 1}</div>
            </div>
          ))}
        </Document>
      )}
    </div>
  );
};

export default PDFViewer;