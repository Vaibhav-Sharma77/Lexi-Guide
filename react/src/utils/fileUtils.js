import mammoth from 'mammoth';

// File processing utilities
export const fileUtils = {
  // Extract text from different file types
  extractText: async (file) => {
    const fileType = file.type || file.name.split('.').pop().toLowerCase();
    
    try {
      switch (fileType) {
        case 'application/pdf':
        case 'pdf':
          return await extractTextFromPDF(file);
        
        case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        case 'docx':
          return await extractTextFromDOCX(file);
        
        case 'application/msword':
        case 'doc':
          return await extractTextFromDOC(file);
        
        case 'text/plain':
        case 'txt':
          return await extractTextFromTXT(file);
        
        default:
          throw new Error(`Unsupported file type: ${fileType}`);
      }
    } catch (error) {
      throw new Error(`Failed to extract text from file: ${error.message}`);
    }
  },

  // Validate file
  validateFile: (file) => {
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    const ALLOWED_TYPES = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword',
      'text/plain'
    ];
    
    const ALLOWED_EXTENSIONS = ['pdf', 'docx', 'doc', 'txt'];
    
    if (file.size > MAX_FILE_SIZE) {
      throw new Error('File size must be less than 10MB');
    }
    
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const isValidType = ALLOWED_TYPES.includes(file.type) || ALLOWED_EXTENSIONS.includes(fileExtension);
    
    if (!isValidType) {
      throw new Error('File type not supported. Please upload PDF, DOC, DOCX, or TXT files.');
    }
    
    return true;
  },

  // Format file size
  formatFileSize: (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
};

// Extract text from PDF using PDF.js (browser-based)
const extractTextFromPDF = async (file) => {
  // Note: This requires PDF.js library to be loaded
  // For now, we'll return a placeholder and suggest manual copy-paste
  throw new Error('PDF text extraction requires additional setup. Please copy and paste the text manually for now.');
};

// Extract text from DOCX files
const extractTextFromDOCX = async (file) => {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    
    if (!result.value || result.value.trim().length === 0) {
      throw new Error('No text content found in the document');
    }
    
    return result.value;
  } catch (error) {
    throw new Error(`Failed to extract text from DOCX: ${error.message}`);
  }
};

// Extract text from DOC files (legacy format)
const extractTextFromDOC = async (file) => {
  // DOC files are more complex to parse in the browser
  // For now, we'll suggest converting to DOCX or copy-paste
  throw new Error('Legacy DOC files are not supported. Please convert to DOCX format or copy and paste the text manually.');
};

// Extract text from TXT files
const extractTextFromTXT = async (file) => {
  try {
    const text = await file.text();
    
    if (!text || text.trim().length === 0) {
      throw new Error('The text file appears to be empty');
    }
    
    return text;
  } catch (error) {
    throw new Error(`Failed to read text file: ${error.message}`);
  }
};

export default fileUtils;