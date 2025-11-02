import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import { 
  Upload, 
  FileText, 
  Brain, 
  Shield, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Download,
  Share2,
  Copy,
  HelpCircle,
  ArrowLeft,
  Zap
} from 'lucide-react';
import FileUpload from '../components/FileUpload';
import AnalysisResults from '../components/AnalysisResults';
import { apiService } from '../utils/api';

const AnalysisPage = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [contractText, setContractText] = useState('');
  const [userRole, setUserRole] = useState('student');
  const [selectedCountry, setSelectedCountry] = useState('');
  const [customCountry, setCustomCountry] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [selectedClause, setSelectedClause] = useState(null);

  const steps = [
    { id: 1, title: 'Upload Contract', icon: Upload },
    { id: 2, title: 'Select Role & Country', icon: FileText },
    { id: 3, title: 'AI Analysis', icon: Brain },
    { id: 4, title: 'Results', icon: Shield }
  ];

  const userRoles = [
    {
      id: 'student',
      title: 'Student',
      description: 'Law student or academic researcher',
      icon: '🎓'
    },
    {
      id: 'freelancer',
      title: 'Freelancer/Contractor',
      description: 'Independent professional providing services',
      icon: '👨‍💻'
    },
    {
      id: 'client',
      title: 'Client/Company',
      description: 'Business hiring contractors or services',
      icon: '🏢'
    },
    {
      id: 'startup',
      title: 'Startup/MSME',
      description: 'Small to medium enterprise',
      icon: '🚀'
    },
    {
      id: 'vendor',
      title: 'Vendor/Supplier',
      description: 'Providing goods or services to businesses',
      icon: '📦'
    },
    {
      id: 'legal',
      title: 'Legal Professional',
      description: 'Lawyer or legal firm representative',
      icon: '⚖️'
    }
  ];

  const handleFileTextExtracted = (text) => {
    setContractText(text);
    if (text.trim().length > 0) {
      setCurrentStep(2);
    }
  };

  const handleManualTextSubmit = () => {
    if (contractText.trim().length < 50) {
      toast.error('Please enter at least 50 characters of contract text');
      return;
    }
    setCurrentStep(2);
  };

  const popularCountries = [
    { code: 'IN', name: 'India', flag: '🇮🇳', legalSystem: 'Common Law with Civil Law influences' },
    { code: 'US', name: 'United States', flag: '🇺🇸', legalSystem: 'Common Law' },
    { code: 'GB', name: 'United Kingdom', flag: '🇬🇧', legalSystem: 'Common Law' },
    { code: 'CA', name: 'Canada', flag: '🇨🇦', legalSystem: 'Common Law and Civil Law' },
    { code: 'AU', name: 'Australia', flag: '🇦🇺', legalSystem: 'Common Law' },
    { code: 'DE', name: 'Germany', flag: '🇩🇪', legalSystem: 'Civil Law' },
    { code: 'FR', name: 'France', flag: '🇫🇷', legalSystem: 'Civil Law' },
    { code: 'JP', name: 'Japan', flag: '🇯🇵', legalSystem: 'Civil Law' },
    { code: 'SG', name: 'Singapore', flag: '🇸🇬', legalSystem: 'Common Law' },
    { code: 'AE', name: 'UAE', flag: '🇦🇪', legalSystem: 'Civil Law with Islamic Law' },
    { code: 'CN', name: 'China', flag: '🇨🇳', legalSystem: 'Civil Law with Socialist characteristics' },
    { code: 'BR', name: 'Brazil', flag: '🇧🇷', legalSystem: 'Civil Law' },
    { code: 'ZA', name: 'South Africa', flag: '🇿🇦', legalSystem: 'Mixed Legal System' },
    { code: 'MY', name: 'Malaysia', flag: '🇲🇾', legalSystem: 'Common Law with Islamic Law' },
    { code: 'TH', name: 'Thailand', flag: '🇹🇭', legalSystem: 'Civil Law' },
    { code: 'OTHER', name: 'Other Country', flag: '🌍', legalSystem: 'Specify your country' }
  ];

  const handleRoleSelection = (role) => {
    setUserRole(role);
    // Don't proceed to next step yet - wait for country selection
  };

  const handleCountrySelection = (countryCode) => {
    setSelectedCountry(countryCode);
    if (countryCode !== 'OTHER') {
      setCustomCountry('');
      proceedToAnalysis(countryCode);
    }
  };

  const handleCustomCountrySubmit = () => {
    if (customCountry.trim().length > 0) {
      proceedToAnalysis('OTHER');
    }
  };

  const proceedToAnalysis = (country) => {
    setCurrentStep(3);
    handleAnalyze(userRole, country);
  };

  const handleAnalyze = async (selectedRole = userRole, selectedCountryCode = selectedCountry) => {
    if (!contractText.trim()) {
      toast.error('Please provide contract text first');
      return;
    }

    if (!selectedCountryCode) {
      toast.error('Please select a country first');
      return;
    }

    setIsAnalyzing(true);
    
    try {
      const countryName = selectedCountryCode === 'OTHER' 
        ? customCountry 
        : popularCountries.find(c => c.code === selectedCountryCode)?.name || 'Unknown';
        
      const result = await apiService.analyzeContract(contractText, selectedRole, countryName);
      setAnalysisData(result);
      setCurrentStep(4);
      toast.success('Analysis completed successfully!');
    } catch (error) {
      toast.error(error.message);
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setCurrentStep(1);
    setContractText('');
    setAnalysisData(null);
    setSelectedClause(null);
    setIsAnalyzing(false);
  };

  const handleBackToStep = (step) => {
    if (step < currentStep && !isAnalyzing) {
      setCurrentStep(step);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
            AI Contract Analysis
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Get instant insights into your contract's safety, risks, and negotiation opportunities
          </p>
        </motion.div>

        {/* Progress Steps */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="mb-8"
        >
          <div className="flex justify-center">
            <div className="flex items-center space-x-8 bg-white rounded-lg p-6 shadow-lg">
              {steps.map((step, index) => (
                <div key={step.id} className="flex items-center">
                  <div
                    className={`flex items-center justify-center w-12 h-12 rounded-full transition-all duration-300 cursor-pointer ${
                      currentStep >= step.id
                        ? 'bg-primary-600 text-white'
                        : 'bg-gray-200 text-gray-600'
                    } ${
                      currentStep > step.id && !isAnalyzing ? 'hover:bg-primary-700' : ''
                    }`}
                    onClick={() => handleBackToStep(step.id)}
                  >
                    <step.icon className="w-6 h-6" />
                  </div>
                  <div className="ml-3 hidden sm:block">
                    <div className={`text-sm font-medium ${
                      currentStep >= step.id ? 'text-primary-600' : 'text-gray-600'
                    }`}>
                      {step.title}
                    </div>
                  </div>
                  {index < steps.length - 1 && (
                    <div className={`w-8 h-0.5 mx-4 ${
                      currentStep > step.id ? 'bg-primary-600' : 'bg-gray-300'
                    }`} />
                  )}
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto">
          <AnimatePresence mode="wait">
            {/* Step 1: Upload Contract */}
            {currentStep === 1 && (
              <motion.div
                key="step1"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.4 }}
                className="bg-white rounded-xl shadow-lg p-8"
              >
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    Upload Your Contract
                  </h2>
                  <p className="text-gray-600">
                    Upload a file or paste your contract text to get started
                  </p>
                </div>

                {/* File Upload */}
                <div className="mb-8">
                  <FileUpload 
                    onTextExtracted={handleFileTextExtracted}
                    isLoading={isAnalyzing}
                  />
                </div>

                {/* Manual Text Input */}
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-300" />
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-4 bg-white text-gray-500">Or paste contract text</span>
                  </div>
                </div>

                <div className="mt-6">
                  <label htmlFor="contract-text" className="block text-sm font-medium text-gray-700 mb-2">
                    Contract Text
                  </label>
                  <textarea
                    id="contract-text"
                    value={contractText}
                    onChange={(e) => setContractText(e.target.value)}
                    placeholder="Paste your contract text here..."
                    className="w-full h-64 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
                    disabled={isAnalyzing}
                  />
                  <div className="flex justify-between items-center mt-2">
                    <div className="text-sm text-gray-500">
                      {contractText.length} characters (minimum 50 required)
                    </div>
                    <button
                      onClick={handleManualTextSubmit}
                      disabled={contractText.trim().length < 50 || isAnalyzing}
                      className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      Continue
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step 2: Select Role & Country */}
            {currentStep === 2 && (
              <motion.div
                key="step2"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.4 }}
                className="bg-white rounded-xl shadow-lg p-8"
              >
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    Select Your Role & Country
                  </h2>
                  <p className="text-gray-600">
                    Choose your role and country for personalized legal analysis
                  </p>
                </div>

                {/* Role Selection */}
                <div className="mb-8">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Your Role</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {userRoles.map((role) => (
                      <motion.button
                        key={role.id}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => handleRoleSelection(role.id)}
                        className={`p-4 border-2 rounded-xl text-left transition-all duration-200 ${
                          userRole === role.id
                            ? 'border-primary-600 bg-primary-50'
                            : 'border-gray-200 hover:border-primary-300 hover:bg-gray-50'
                        }`}
                      >
                        <div className="flex flex-col items-center text-center space-y-2">
                          <div className="text-3xl">{role.icon}</div>
                          <div>
                            <div className="font-semibold text-gray-900 text-sm">{role.title}</div>
                            <div className="text-xs text-gray-600">{role.description}</div>
                          </div>
                        </div>
                      </motion.button>
                    ))}
                  </div>
                </div>

                {/* Country Selection */}
                {userRole && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mb-6"
                  >
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Country/Jurisdiction</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-3">
                      {popularCountries.map((country) => (
                        <motion.button
                          key={country.code}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={() => handleCountrySelection(country.code)}
                          className={`p-3 border-2 rounded-lg text-center transition-all duration-200 ${
                            selectedCountry === country.code
                              ? 'border-primary-600 bg-primary-50'
                              : 'border-gray-200 hover:border-primary-300 hover:bg-gray-50'
                          }`}
                        >
                          <div className="text-2xl mb-1">{country.flag}</div>
                          <div className="text-xs font-medium text-gray-900">{country.name}</div>
                          <div className="text-xs text-gray-500 mt-1">{country.legalSystem}</div>
                        </motion.button>
                      ))}
                    </div>

                    {/* Custom Country Input */}
                    {selectedCountry === 'OTHER' && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        transition={{ duration: 0.3 }}
                        className="mt-4 p-4 bg-gray-50 rounded-lg"
                      >
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Enter Your Country
                        </label>
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={customCountry}
                            onChange={(e) => setCustomCountry(e.target.value)}
                            placeholder="Enter country name..."
                            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                          />
                          <button
                            onClick={handleCustomCountrySubmit}
                            disabled={!customCountry.trim()}
                            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                          >
                            Analyze
                          </button>
                        </div>
                        <p className="text-xs text-gray-500 mt-2">
                          We'll provide general legal analysis for your specified country
                        </p>
                      </motion.div>
                    )}
                  </motion.div>
                )}
              </motion.div>
            )}

            {/* Step 3: AI Analysis */}
            {currentStep === 3 && (
              <motion.div
                key="step3"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.4 }}
                className="bg-white rounded-xl shadow-lg p-8"
              >
                <div className="text-center">
                  <div className="w-24 h-24 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full flex items-center justify-center mx-auto mb-6">
                    <Brain className="w-12 h-12 text-white animate-pulse" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    AI is Analyzing Your Contract
                  </h2>
                  <p className="text-gray-600 mb-8">
                    Our advanced AI is reviewing every clause for risks and opportunities
                  </p>
                  
                  <div className="flex justify-center space-x-8 mb-8">
                    {[
                      { icon: Shield, label: 'Safety Analysis', delay: 0 },
                      { icon: AlertTriangle, label: 'Risk Detection', delay: 1 },
                      { icon: Zap, label: 'Smart Suggestions', delay: 2 }
                    ].map((item, index) => (
                      <motion.div
                        key={item.label}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: item.delay * 0.3 }}
                        className="text-center"
                      >
                        <item.icon className="w-8 h-8 text-primary-600 mx-auto mb-2" />
                        <div className="text-sm text-gray-600">{item.label}</div>
                      </motion.div>
                    ))}
                  </div>

                  <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: '100%' }}
                      transition={{ duration: 3, ease: 'easeInOut' }}
                      className="bg-gradient-to-r from-primary-500 to-secondary-500 h-2 rounded-full"
                    />
                  </div>

                  <div className="text-sm text-gray-500">
                    This usually takes 10-30 seconds...
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step 4: Results */}
            {currentStep === 4 && analysisData && (
              <motion.div
                key="step4"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.4 }}
              >
                <AnalysisResults
                  analysisData={analysisData}
                  contractText={contractText}
                  userRole={userRole}
                  selectedClause={selectedClause}
                  onClauseSelect={setSelectedClause}
                  onReset={resetAnalysis}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;