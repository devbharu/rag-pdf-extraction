import React, { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// Set worker source
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

const PDFViewer = ({ file, pageNumber }) => {
  const [numPages, setNumPages] = useState(null);
  const [containerWidth, setContainerWidth] = useState(null);

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
  }

  useEffect(() => {
    const updateWidth = () => {
      const container = document.getElementById('pdf-container');
      if (container) {
        setContainerWidth(container.clientWidth - 48); // Subtract padding (p-6 = 24px * 2)
      }
    };

    window.addEventListener('resize', updateWidth);
    // Small delay to ensure container is rendered
    setTimeout(updateWidth, 100);

    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Scroll to page when pageNumber changes
  useEffect(() => {
    if (pageNumber) {
      const pageElement = document.getElementById(`page_${pageNumber}`);
      if (pageElement) {
        pageElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }, [pageNumber]);

  return (
    <div id="pdf-container" className="h-full overflow-y-auto p-6 bg-stone-100 flex flex-col items-center scroll-smooth">
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
    </div>
  );
};

export default PDFViewer;