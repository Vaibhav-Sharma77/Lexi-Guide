import axios from 'axios';

// Use environment variable for API URL, fallback to localhost for development
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// API Functions
export const apiService = {
  // Health check
  healthCheck: async () => {
    try {
      const response = await api.get('/');
      return response.data;
    } catch (error) {
      throw new Error('Backend service is not available');
    }
  },

  // Analyze contract
  analyzeContract: async (contractText, userRole = 'freelancer', country = 'India', userId = null) => {
    try {
      const response = await api.post('/analyze', {
        contract_text: contractText,
        user_role: userRole,
        country: country,
        user_id: userId
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        // Server responded with error status
        const errorMessage = error.response.data?.detail || 
                           error.response.data?.message || 
                           `Server error: ${error.response.status}`;
        console.error('API Error:', errorMessage);
        throw new Error(errorMessage);
      } else if (error.request) {
        // Request was made but no response received
        console.error('Network Error:', error.request);
        throw new Error('No response from server. Please check your connection.');
      } else {
        // Something else happened
        console.error('Unexpected Error:', error.message);
        throw new Error('An unexpected error occurred');
      }
    }
  }
};

export default api;