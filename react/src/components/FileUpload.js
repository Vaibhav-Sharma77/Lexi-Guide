import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { 
  Upload, 
  FileText, 
  X, 
  CheckCircle, 
  AlertCircle,
  Loader
} from 'lucide-react';
import { fileUtils } from '../utils/fileUtils';

const FileUpload = ({ onTextExtracted, isLoading }) => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [extractingText, setExtractingText] = useState(false);

  const onDrop = useCallback(async (acceptedFiles, rejectedFiles) => {
    // Handle rejected files
    if (rejectedFiles.length > 0) {
      rejectedFiles.forEach(({ file, errors }) => {
        errors.forEach(error => {
          toast.error(`${file.name}: ${error.message}`);
        });
      });
      return;
    }

    const file = acceptedFiles[0];
    if (!file) return;

    try {
      // Validate file
      fileUtils.validateFile(file);
      setUploadedFile(file);
      setExtractingText(true);

      // Extract text from file
      const extractedText = await fileUtils.extractText(file);
      
      if (extractedText && extractedText.trim().length > 0) {
        onTextExtracted(extractedText);
        toast.success('File uploaded and text extracted successfully!');
      } else {
        throw new Error('No text content found in the file');
      }
    } catch (error) {
      toast.error(error.message);
      setUploadedFile(null);
    } finally {
      setExtractingText(false);
    }
  }, [onTextExtracted]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt']
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false,
    disabled: isLoading || extractingText
  });

  const removeFile = () => {
    setUploadedFile(null);
    onTextExtracted('');
  };

  const getDropzoneClasses = () => {
    let classes = 'dropzone p-8 text-center cursor-pointer transition-all duration-300 ';
    
    if (isDragActive && !isDragReject) {
      classes += 'active border-primary-500 bg-primary-50';
    } else if (isDragReject) {
      classes += 'reject border-red-500 bg-red-50';
    } else if (isLoading || extractingText) {
      classes += 'cursor-not-allowed bg-gray-50 border-gray-300';
    } else {
      classes += 'hover:border-primary-400 hover:bg-gray-50';
    }
    
    return classes;
  };

  return (
    <div className="w-full">
      {!uploadedFile ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div {...getRootProps()} className={getDropzoneClasses()}>
            <input {...getInputProps()} />
            
            <div className="flex flex-col items-center space-y-4">
              {extractingText ? (
                <>
                  <Loader className="w-12 h-12 text-primary-600 animate-spin" />
                  <div className="text-lg font-medium text-gray-700">
                    Extracting text from file...
                  </div>
                  <div className="text-sm text-gray-500">
                    Please wait while we process your document
                  </div>
                </>
              ) : (
                <>
                  <Upload className="w-12 h-12 text-primary-600" />
                  <div className="text-lg font-medium text-gray-700">
                    {isDragActive 
                      ? isDragReject 
                        ? 'File type not supported'
                        : 'Drop your contract here'
                      : 'Drag & drop your contract here'
                    }
                  </div>
                  <div className="text-sm text-gray-500">
                    or <span className="text-primary-600 font-medium">browse files</span>
                  </div>
                  <div className="flex items-center space-x-4 text-xs text-gray-400">
                    <span className="flex items-center">
                      <FileText className="w-4 h-4 mr-1" />
                      PDF
                    </span>
                    <span className="flex items-center">
                      <FileText className="w-4 h-4 mr-1" />
                      DOC/DOCX
                    </span>
                    <span className="flex items-center">
                      <FileText className="w-4 h-4 mr-1" />
                      TXT
                    </span>
                  </div>
                  <div className="text-xs text-gray-400">
                    Maximum file size: 10MB
                  </div>
                </>
              )}
            </div>
          </div>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="bg-green-50 border border-green-200 rounded-lg p-6"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <CheckCircle className="w-8 h-8 text-green-600" />
              <div>
                <div className="font-medium text-green-900">{uploadedFile.name}</div>
                <div className="text-sm text-green-700">
                  {fileUtils.formatFileSize(uploadedFile.size)} • Text extracted successfully
                </div>
              </div>
            </div>
            <button
              onClick={removeFile}
              className="text-green-600 hover:text-green-800 transition-colors"
              disabled={isLoading}
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </motion.div>
      )}

      {/* File Type Information */}
      <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start space-x-2">
          <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
          <div className="text-sm text-blue-800">
            <div className="font-medium mb-1">Supported file formats:</div>
            <ul className="space-y-1 text-xs">
              <li><strong>PDF:</strong> Requires manual copy-paste for now</li>
              <li><strong>DOCX:</strong> Fully supported with automatic text extraction</li>
              <li><strong>DOC:</strong> Please convert to DOCX or copy-paste manually</li>
              <li><strong>TXT:</strong> Fully supported</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;